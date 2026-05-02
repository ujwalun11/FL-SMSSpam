"""
=============================================================================
  FedShield — Flower FL Server
  File: flwr_server.py

  Uses flwr (Flower) framework instead of hand-rolled aggregation.
  Everything else is IDENTICAL to server.py:
    - Same Flask chat API  (send_message, get_messages, correct_label)
    - Same disk persistence (messages.json, global_model_checkpoint.pt)
    - Same /register heartbeat for browser online indicators
    - Same /predict endpoint
    - float32 weights throughout — NO float16 precision loss

  HOW flwr REPLACES OUR MANUAL CODE:
    server.py              →  flwr_server.py
    aggregation_loop()     →  fl.server.start_server() handles rounds
    do_fedavg()            →  FedShieldStrategy(FedAvg) — override callbacks
    chunk_buffer           →  flwr gRPC transport handles weight transfer
    client_updates list    →  flwr collects per-round results internally
    fl_round counter       →  server_round param passed to callbacks

  RUN — TWO TERMINALS on Laptop A:
    Terminal 1:  python flwr_server.py                  (Flask API on :5000)
    Terminal 2:  python flwr_server.py --fl-only        (Flower gRPC on :8080)
    Terminal 3:  ngrok http 5000

  OR combined (one terminal):
    python flwr_server.py --combined

  INSTALL:
    pip install flwr torch transformers flask flask-cors numpy pandas
=============================================================================
"""

import os, json, time, threading, re, argparse
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    import torch
    import torch.nn as nn
    from transformers import DistilBertTokenizer, DistilBertModel
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[WARN] torch/transformers not installed")

try:
    import flwr as fl
    from flwr.server.strategy import FedAvg
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server import ServerConfig
    FLWR_OK = True
except ImportError:
    FLWR_OK = False
    print("[ERROR] flwr not installed — run:  pip install flwr")

# ─────────────────────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--fl-only",     action="store_true",
                    help="Run only the Flower gRPC server (no Flask API)")
parser.add_argument("--combined",    action="store_true",
                    help="Run Flask + Flower in one process")
parser.add_argument("--fl-port",     type=int, default=8080)
parser.add_argument("--api-port",    type=int, default=5000)
parser.add_argument("--rounds",      type=int, default=999,
                    help="FL rounds to run (default 999 = run until stopped)")
parser.add_argument("--min-clients", type=int, default=1,
                    help="Min clients before a round starts (default 1)")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH         = "best_model_dataset5971.pt"
CHECKPOINT_PATH    = "global_model_checkpoint.pt"
MESSAGES_PATH      = "messages.json"
DISTILBERT_NAME    = "distilbert-base-uncased"
NUM_FEAT_SIZE      = 17
NUM_CLASSES        = 3
MAX_LEN            = 160
LABEL_MAP          = {0: "ham", 1: "spam", 2: "smishing"}
CLIENT_TIMEOUT_SEC = 120

# ─────────────────────────────────────────────────────────────────────────────
# MODEL  (identical to server.py / flwr_client.py)
# ─────────────────────────────────────────────────────────────────────────────
if TORCH_OK:
    class DistilBertWithFeatures(nn.Module):
        def __init__(self, model_name, num_feat_size, num_classes, dropout=0.3):
            super().__init__()
            self.bert = DistilBertModel.from_pretrained(model_name)
            bert_h    = self.bert.config.hidden_size
            self.feat_proj = nn.Sequential(
                nn.Linear(num_feat_size, 64), nn.ReLU(), nn.Dropout(dropout))
            self.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(bert_h + 64, 256),
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes))

        def forward(self, input_ids, attention_mask, num_feats):
            cls = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask).last_hidden_state[:, 0, :]
            return self.classifier(torch.cat([cls, self.feat_proj(num_feats)], dim=1))

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(text):
    t = str(text); n = max(len(t), 1)
    return [
        len(t), len(t.split()),
        sum(c.isdigit() for c in t)/n, sum(c.isupper() for c in t)/n,
        int(bool(re.search(r"[£$€¥]", t))), t.count("!"),
        len(re.findall(r"http\S+|www\.\S+", t, re.I)),
        len(re.findall(r"\b\d{7,}\b", t)),
        int(bool(re.search(r"\bfree\b",      t, re.I))),
        int(bool(re.search(r"\bwin(ner)?\b", t, re.I))),
        int(bool(re.search(r"\bprize\b",     t, re.I))),
        int(bool(re.search(r"\burgent\b",    t, re.I))),
        int(bool(re.search(r"\bclick\b",     t, re.I))),
        int(bool(re.search(r"\bcall\b",      t, re.I))),
        int(bool(re.search(r"http\S+|www\.\S+", t, re.I))),
        int(bool(re.search(r"\S+@\S+\.\S+",    t, re.I))),
        int(bool(re.search(r"\b\d{7,}\b",       t))),
    ]

def clean_text(text):
    t = str(text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", t).strip()

def rule_based(text):
    tl = text.lower()
    smish = ["verify","bank","account","login","secure","http","confirm your"]
    spam  = ["free","win","prize","claim","reward","congratulations"]
    ss = sum(1 for w in smish if w in tl)
    sp = sum(1 for w in spam  if w in tl)
    if ss >= 2: return "smishing", round(min(50+ss*10, 95), 1)
    if sp >= 2: return "spam",     round(min(50+sp*10, 95), 1)
    return "ham", 88.0

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────
global_model     = None
tokenizer        = None
device           = None
messages_db      = []
browser_sessions = {}
fl_round_counter = 0          # updated by Flower callback
lock             = threading.Lock()

# Snapshot weights at round start — used to prove weights changed
_weights_before_round = None

# ─────────────────────────────────────────────────────────────────────────────
# MODEL INIT / HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    global global_model, tokenizer, device
    if not TORCH_OK:
        return
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Server] Device: {device}")
    tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_NAME)
    global_model = DistilBertWithFeatures(
        DISTILBERT_NAME, NUM_FEAT_SIZE, NUM_CLASSES).to(device)
    if os.path.exists(CHECKPOINT_PATH):
        global_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print(f"[Server] Resumed from {CHECKPOINT_PATH} ✓")
    elif os.path.exists(MODEL_PATH):
        global_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"[Server] Loaded {MODEL_PATH} ✓")
    else:
        print("[Server] WARNING: no .pt file found")
    global_model.eval()
    print(f"[Server] {sum(p.numel() for p in global_model.parameters()):,} params ✓")

def get_ndarrays():
    """Current global model weights as list of numpy float32 arrays."""
    return [v.cpu().numpy().astype(np.float32)
            for v in global_model.state_dict().values()]

def set_ndarrays(ndarrays):
    """
    Apply aggregated numpy arrays from Flower back into global_model.
    float32 throughout — no precision loss unlike our hand-rolled float16.
    """
    state = {k: torch.tensor(v, dtype=torch.float32)
             for k, v in zip(global_model.state_dict().keys(), ndarrays)}
    global_model.load_state_dict(state, strict=True)
    global_model.eval()

def save_checkpoint():
    if not TORCH_OK or global_model is None:
        return
    try:
        torch.save(global_model.state_dict(), CHECKPOINT_PATH)
        print(f"[Server] Checkpoint → {CHECKPOINT_PATH} ✓")
    except Exception as e:
        print(f"[Server] Checkpoint save failed: {e}")

def run_predict(text):
    if not TORCH_OK or global_model is None:
        label, conf = rule_based(text)
        return {"label": label, "confidence": conf, "probs": {}, "mode": "rule-based"}
    enc = tokenizer(clean_text(text), max_length=MAX_LEN,
                    padding="max_length", truncation=True, return_tensors="pt")
    nf  = torch.tensor([extract_features(text)], dtype=torch.float).to(device)
    global_model.eval()
    with torch.no_grad():
        logits = global_model(enc["input_ids"].to(device),
                              enc["attention_mask"].to(device), nf)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return {
        "label"     : LABEL_MAP[int(probs.argmax())],
        "confidence": round(float(probs.max()) * 100, 1),
        "probs"     : {"ham":     round(float(probs[0]) * 100, 1),
                       "spam":    round(float(probs[1]) * 100, 1),
                       "smishing":round(float(probs[2]) * 100, 1)},
        "mode"      : "distilbert"
    }

# ─────────────────────────────────────────────────────────────────────────────
# DISK PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
def save_messages():
    try:
        with open(MESSAGES_PATH, "w") as f:
            json.dump(messages_db, f)
    except Exception as e:
        print(f"[Server] Cannot save messages: {e}")

def load_messages():
    global messages_db
    if not os.path.exists(MESSAGES_PATH):
        return
    try:
        with open(MESSAGES_PATH) as f:
            messages_db = json.load(f)
        print(f"[Server] Loaded {len(messages_db)} messages ✓")
    except Exception as e:
        print(f"[Server] Cannot load messages: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# FLOWER STRATEGY
#
# FedShieldStrategy(FedAvg) overrides three hooks:
#
#   configure_fit()    — called before a round. We snapshot current weights
#                        here so we can measure change AFTER aggregation.
#
#   aggregate_fit()    — called after clients return trained weights.
#                        Parent FedAvg does weighted averaging.
#                        We then apply result to global_model + save checkpoint.
#                        We also print a weight-change proof so you can see
#                        the model actually updated.
#
#   aggregate_evaluate() — called after evaluation phase. We log accuracy.
# ─────────────────────────────────────────────────────────────────────────────
if FLWR_OK:
    class FedShieldStrategy(FedAvg):

        def configure_fit(self, server_round, parameters, client_manager):
            """Snapshot weights before the round so we can measure change."""
            global _weights_before_round
            if TORCH_OK and global_model is not None:
                # Save classifier layer for change measurement
                ck  = [k for k in global_model.state_dict() if "classifier" in k]
                _weights_before_round = {
                    k: global_model.state_dict()[k].cpu().numpy().copy()
                    for k in ck[:2]   # first two classifier tensors
                }
            return super().configure_fit(server_round, parameters, client_manager)

        def aggregate_fit(self, server_round, results, failures):
            global fl_round_counter

            if not results:
                print(f"[Flower] Round {server_round}: no client results")
                return None, {}

            # ── Flower's FedAvg does the weighted averaging here ─────────────
            # It averages each layer proportionally by num_examples per client
            # using float32 — no precision loss
            aggregated_params, metrics = super().aggregate_fit(
                server_round, results, failures)

            if aggregated_params is not None and TORCH_OK and global_model is not None:

                # ── Apply aggregated weights to the live model ───────────────
                ndarrays = parameters_to_ndarrays(aggregated_params)
                set_ndarrays(ndarrays)

                # ── PROOF: measure how much the weights actually changed ──────
                print(f"\n[Flower] ═══════ Round {server_round} complete ═══════")
                if _weights_before_round is not None:
                    for k, w_before in _weights_before_round.items():
                        w_after   = global_model.state_dict()[k].cpu().numpy()
                        diff      = w_after - w_before
                        l2_change = float(np.linalg.norm(diff))
                        max_change= float(np.abs(diff).max())
                        pct       = float((np.abs(diff) > 1e-7).mean() * 100)
                        print(f"[Flower]   Layer '{k}'")
                        print(f"[Flower]     L2 change  : {l2_change:.6f}  "
                              f"({'UPDATED' if l2_change > 1e-6 else 'NO CHANGE'})")
                        print(f"[Flower]     Max change : {max_change:.6f}")
                        print(f"[Flower]     % changed  : {pct:.1f}%")

                # ── Log round summary ────────────────────────────────────────
                total_samples = sum(r.num_examples for _, r in results)
                n_clients     = len(results)
                print(f"[Flower]   Clients    : {n_clients}")
                print(f"[Flower]   Samples    : {total_samples}")
                if failures:
                    print(f"[Flower]   Failures   : {len(failures)}")

                # ── Save checkpoint ──────────────────────────────────────────
                save_checkpoint()

                # ── Update round counter for Flask /status ───────────────────
                fl_round_counter = server_round
                print(f"[Flower] ════════════════════════════════════════\n")

            return aggregated_params, metrics

        def aggregate_evaluate(self, server_round, results, failures):
            """Log per-round evaluation accuracy."""
            if not results:
                return None, {}
            total   = sum(r.num_examples for _, r in results)
            acc_avg = sum(r.metrics.get("accuracy", 0) * r.num_examples
                          for _, r in results) / max(total, 1)
            print(f"[Flower] Round {server_round} eval accuracy: "
                  f"{acc_avg*100:.2f}% over {total} samples")
            return acc_avg, {"accuracy": acc_avg}

# ─────────────────────────────────────────────────────────────────────────────
# FLASK CHAT API  (identical to server.py)
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/status")
def status():
    now = time.time()
    with lock:
        active = {cid: info for cid, info in browser_sessions.items()
                  if now - info["last_seen"] < CLIENT_TIMEOUT_SEC}
    return jsonify({
        "ok"             : True,
        "fl_round"       : fl_round_counter,
        "fl_framework"   : "flower",
        "browser_clients": [{"client_id": cid, "name": info["name"]}
                             for cid, info in active.items()],
        "total_messages" : len(messages_db),
        "model_loaded"   : global_model is not None,
        "checkpoint"     : os.path.exists(CHECKPOINT_PATH),
        "fl_port"        : args.fl_port,
    })

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(force=True)
    cid  = data.get("client_id", "unknown")
    name = data.get("name", "User")
    with lock:
        browser_sessions[cid] = {"name": name, "last_seen": time.time()}
    return jsonify({"ok": True, "client_id": cid, "fl_round": fl_round_counter})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "empty"}), 400
    r = run_predict(text)
    r["fl_round"] = fl_round_counter
    return jsonify(r)

@app.route("/send_message", methods=["POST"])
def send_message():
    data   = request.get_json(force=True)
    sender = str(data.get("sender", "unknown")).strip()
    text   = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "empty"}), 400
    pred = run_predict(text)
    with lock:
        msg = {
            "id"        : len(messages_db) + 1,
            "sender"    : sender,
            "text"      : text,
            "label"     : pred["label"],
            "confidence": pred["confidence"],
            "probs"     : pred.get("probs", {}),
            "mode"      : pred.get("mode", ""),
            "corrected" : False,
            "timestamp" : time.time(),
            "fl_round"  : fl_round_counter,
        }
        messages_db.append(msg)
        save_messages()
    print(f"[Server] Msg #{msg['id']} {sender}: {text[:50]} → {pred['label']}")
    return jsonify(msg)

@app.route("/correct_label", methods=["POST"])
def correct_label():
    data    = request.get_json(force=True)
    msg_id  = int(data.get("msg_id", -1))
    correct = str(data.get("correct_label", "")).strip().lower()
    if correct not in ("ham", "spam", "smishing"):
        return jsonify({"error": "invalid label"}), 400
    with lock:
        for msg in messages_db:
            if msg["id"] == msg_id:
                old = msg["label"]
                msg["label"] = correct
                msg["corrected"] = True
                save_messages()
                print(f"[Server] Msg #{msg_id}: {old} → {correct}")
                return jsonify({"ok": True, "msg_id": msg_id,
                                "old_label": old, "new_label": correct})
    return jsonify({"error": f"msg_id {msg_id} not found"}), 404

@app.route("/get_messages")
def get_messages():
    since = int(request.args.get("since", 0))
    with lock:
        msgs = [m for m in messages_db if m["id"] > since]
    return jsonify({"messages": msgs, "fl_round": fl_round_counter,
                    "total": len(messages_db)})

# ─────────────────────────────────────────────────────────────────────────────
# FLOWER SERVER LAUNCHER
# ─────────────────────────────────────────────────────────────────────────────
def start_flower_server():
    if not FLWR_OK:
        print("[ERROR] flwr not installed"); return
    if not TORCH_OK or global_model is None:
        print("[ERROR] model not loaded"); return

    initial_params = ndarrays_to_parameters(get_ndarrays())

    strategy = FedShieldStrategy(
        fraction_fit          = 1.0,       # train on all available clients
        fraction_evaluate     = 1.0,       # evaluate on all clients
        min_fit_clients       = args.min_clients,
        min_evaluate_clients  = args.min_clients,
        min_available_clients = args.min_clients,
        initial_parameters    = initial_params,  # clients get model on first connect
    )

    addr = f"0.0.0.0:{args.fl_port}"
    print(f"\n[Flower] gRPC server starting on {addr}")
    print(f"[Flower] min_clients={args.min_clients}  rounds={args.rounds}")
    print(f"[Flower] Clients connect with:  --server localhost:{args.fl_port}\n")

    fl.server.start_server(
        server_address = addr,
        config         = ServerConfig(num_rounds=args.rounds),
        strategy       = strategy,
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_messages()
    load_model()

    print("\n[Server] ─────────────────────────────────────────────")
    print(f"[Server] Checkpoint : {CHECKPOINT_PATH}")
    print(f"[Server] Messages   : {MESSAGES_PATH}")
    print(f"[Server] Flask API  : port {args.api_port}")
    print(f"[Server] Flower gRPC: port {args.fl_port}")
    print(f"[Server] ngrok:  run  ngrok http {args.api_port}  in another terminal")
    print("[Server] ─────────────────────────────────────────────\n")

    if args.fl_only:
        start_flower_server()

    elif args.combined:
        flask_t = threading.Thread(
            target=lambda: app.run(
                host="0.0.0.0", port=args.api_port,
                debug=False, threaded=True, use_reloader=False),
            daemon=True)
        flask_t.start()
        print(f"[Server] Flask running on :{args.api_port}")
        start_flower_server()

    else:
        # Default: Flask only — start Flower separately with --fl-only
        print("[Server] Running Flask API only.")
        print(f"[Server] Start Flower in another terminal:")
        print(f"[Server]   python flwr_server.py --fl-only")
        print(f"[Server] Or run both together:")
        print(f"[Server]   python flwr_server.py --combined\n")
        app.run(host="0.0.0.0", port=args.api_port, debug=False, threaded=True)
