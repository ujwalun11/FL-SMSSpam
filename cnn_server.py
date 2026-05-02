"""
=============================================================================
  FedShield CNN — Multi-Channel CNN Server
  File: cnn_server.py  (SEPARATE from flwr_server.py — existing files untouched)

  Architecture:
    - TensorFlow/Keras Multi-Channel CNN (3/4/5-gram parallel Conv1D)
    - Embedding dim=128, vocab=10000, max_len=300
    - FedAvg via plain HTTP weight submission (no Flower/gRPC)
    - Synthetic smishing class generated from templates + nlpaug augmentation
    - Same REST API shape as flwr_server.py — index.html works unchanged

  RUN:
    python cnn_server.py                    (Flask on :5050)

  ADDITIONAL FL ENDPOINTS (beyond the standard chat API):
    GET  /get_global_weights    → FL clients pull current global weights
    POST /submit_weights        → FL clients push local weights after training
    POST /aggregate             → trigger FedAvg manually (or auto every 60s)
    GET  /get_tokenizer_config  → Keras tokenizer JSON (clients share same vocab)

  INSTALL:
    pip install tensorflow flask flask-cors numpy pandas scikit-learn nlpaug
=============================================================================
"""

import os, json, time, threading, re, argparse, pickle, base64
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_OK = True
except ImportError:
    TF_OK = False
    print("[WARN] TensorFlow not installed — run: pip install tensorflow")

try:
    import nlpaug.augmenter.char as nac
    NLPAUG_OK = True
except ImportError:
    NLPAUG_OK = False
    print("[WARN] nlpaug not installed — augmentation disabled. pip install nlpaug")

# ─────────────────────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="FedShield CNN Server")
parser.add_argument("--api-port",    type=int, default=5050)
parser.add_argument("--min-clients", type=int, default=3,
                    help="Number of clients that must submit before FedAvg runs (default 3)")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE      = "data_cleaned.csv"
WEIGHTS_PATH   = "data/cnn_global_weights.pkl"
TOKENIZER_PATH = "data/cnn_tokenizer.json"
MESSAGES_PATH  = "data/cnn_messages.json"
VOCAB_SIZE     = 10000
MAX_LEN        = 300
EMBED_DIM      = 128
NUM_CLASSES    = 3
LABEL_MAP      = {0: "ham", 1: "spam", 2: "smishing"}
CLIENT_TIMEOUT = 120
# Removed AUTO_AGG_SECS — aggregation is now triggered when min_clients have submitted

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────
global_model     = None
tokenizer_keras  = None
messages_db      = []
browser_sessions = {}
fl_round_counter = 0
# Each entry: {"weights": [...], "n_samples": int, "client_id": str}
client_updates   = []
lock             = threading.Lock()
# Event that fires after each successful FedAvg — clients wait on /wait_for_round
_round_done_event = threading.Event()
# Server-side corrections buffer — ALL FL clients pull from here each round
# so every human label correction trains ALL clients, not just one browser's local client.
# Dictionary: { "client_id": [{"text": str, "label": int}, ...] }
CORRECTIONS_FILE   = "data/cnn_corrections.json"
corrections_buffer = {}
corrections_lock   = threading.Lock()

def save_corrections():
    try:
        with open(CORRECTIONS_FILE, "w") as f:
            json.dump(corrections_buffer, f)
    except Exception as e:
        print(f"[CNN] Error saving corrections: {e}")

def load_corrections():
    global corrections_buffer
    if os.path.exists(CORRECTIONS_FILE):
        try:
            with open(CORRECTIONS_FILE) as f:
                corrections_buffer = json.load(f)
            print(f"[CNN] Loaded pending human corrections for {len(corrections_buffer)} clients")
        except Exception as e:
            print(f"[CNN] Error loading corrections: {e}")
    if not isinstance(corrections_buffer, dict):
        corrections_buffer = {}

# ─────────────────────────────────────────────────────────────────────────────
# MULTI-CHANNEL CNN MODEL  (exact architecture from user spec)
# ─────────────────────────────────────────────────────────────────────────────
def create_cnn_model():
    inp = layers.Input(shape=(MAX_LEN,))
    x   = layers.Embedding(VOCAB_SIZE, EMBED_DIM)(inp)

    # Three parallel Conv channels — trigram, 4-gram, 5-gram
    p3 = layers.GlobalMaxPooling1D()(layers.Conv1D(128, 3, activation="relu", padding="same")(x))
    p4 = layers.GlobalMaxPooling1D()(layers.Conv1D(128, 4, activation="relu", padding="same")(x))
    p5 = layers.GlobalMaxPooling1D()(layers.Conv1D(128, 5, activation="relu", padding="same")(x))

    merged = layers.Concatenate()([p3, p4, p5])
    out    = layers.Dense(NUM_CLASSES, activation="softmax")(
                 layers.Dropout(0.5)(
                     layers.Dense(64, activation="relu")(merged)))

    m = models.Model(inputs=inp, outputs=out)
    m.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m

# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT SERIALISATION  (pickle + base64 for HTTP transport)
# ─────────────────────────────────────────────────────────────────────────────
def weights_to_b64(w_list):
    return base64.b64encode(
        pickle.dumps([w.astype(np.float32) for w in w_list])
    ).decode()

def b64_to_weights(b64str):
    return pickle.loads(base64.b64decode(b64str))

def save_weights(w_list):
    with open(WEIGHTS_PATH, "wb") as f:
        pickle.dump([w.astype(np.float32) for w in w_list], f)
    print(f"[CNN] Weights saved → {WEIGHTS_PATH}")

def load_weights():
    if not os.path.exists(WEIGHTS_PATH):
        return None
    with open(WEIGHTS_PATH, "rb") as f:
        return pickle.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# TEXT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def clean(text):
    t = str(text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", t).strip().lower()

def rule_based(text):
    tl = text.lower()
    smish = ["verify","bank","account","login","secure","http","confirm","suspended","kyc"]
    spam  = ["free","win","prize","claim","reward","congratulations","winner","selected"]
    ss = sum(1 for w in smish if w in tl)
    sp = sum(1 for w in spam  if w in tl)
    if ss >= 2: return "smishing", round(min(50 + ss * 10, 95), 1)
    if sp >= 2: return "spam",     round(min(50 + sp * 10, 95), 1)
    return "ham", 88.0

def texts_to_padded(texts):
    seqs = tokenizer_keras.texts_to_sequences([str(t) for t in texts])
    return pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC SMISHING DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────
_SMISH_TMPL = [
    "urgent your {bank} account has been compromised verify at http://secure-{r}.com login now",
    "your {bank} account will be suspended click http://verify-{r}.net to confirm identity",
    "security alert unusual login attempt detected confirm at http://bank-{r}.com secure",
    "dear customer your account is at risk verify now http://login-{r}.org account",
    "final notice confirm your {bank} details or account will be closed http://{r}-secure.com",
    "your package could not be delivered update your address http://delivery-{r}.com track",
    "irs tax refund you are eligible for {amt} refund claim at http://irs-{r}.net",
    "action required verify your paypal account http://paypal-verify-{r}.com or lose access",
    "your upi is suspended confirm details at http://upi-{r}.in verify within 24 hours",
    "sbi alert debit card blocked reactivate http://sbi-{r}.com card activate immediately",
    "hdfc bank your account will be frozen login http://hdfc-{r}.net secure immediately",
    "congratulations you have been selected for kyc update visit http://{r}-kyc.com now",
    "alert your netbanking is temporarily blocked verify at http://{r}-netbank.com now",
    "dear user your {bank} credit card has been blocked call or click http://{r}.xyz",
]

def _gen_smishing(n=400):
    import random, string
    banks = ["SBI", "HDFC", "ICICI", "Axis", "PNB", "Kotak", "BOI", "PayTM"]
    out = []
    for _ in range(n):
        t  = random.choice(_SMISH_TMPL)
        r  = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        s  = t.format(r=r, bank=random.choice(banks), amt=f"${random.randint(100,9999)}")
        out.append(s)
    return out

def _augment_texts(texts, p=0.15):
    if not NLPAUG_OK or not texts:
        return texts
    try:
        aug = nac.RandomCharAug(action="substitute", aug_char_p=p)
        return [aug.augment(t)[0] for t in texts]
    except Exception as e:
        print(f"[CNN] nlpaug error: {e}")
        return texts

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING + TOKENIZER FITTING
# ─────────────────────────────────────────────────────────────────────────────
def load_and_prepare():
    global tokenizer_keras
    texts, labels = [], []

    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, sep=";", index_col=0)
        df = df.dropna(subset=["class", "message"]).reset_index(drop=True)
        lmap = {"ham": 0, "spam": 1, "smishing": 2}
        for _, row in df.iterrows():
            lbl = lmap.get(str(row["class"]).strip().lower(), 0)
            texts.append(clean(str(row["message"])))
            labels.append(lbl)
        print(f"[CNN] Loaded {len(df)} rows from {DATA_FILE}")
    else:
        print(f"[CNN] WARNING: {DATA_FILE} not found — using synthetic data only")

    # --- Synthetic smishing class (class 2) ---
    smish_base = _gen_smishing(400)
    spam_texts = [t for t, l in zip(texts, labels) if l == 1][:150]
    smish_aug  = _augment_texts(spam_texts, p=0.2)
    all_smish  = smish_base + smish_aug
    texts  += all_smish
    labels += [2] * len(all_smish)

    h = labels.count(0); sp = labels.count(1); sm = labels.count(2)
    print(f"[CNN] Dataset → ham:{h}  spam:{sp}  smishing:{sm}  total:{len(texts)}")

    # --- Tokenizer ---
    if TF_OK:
        if os.path.exists(TOKENIZER_PATH):
            with open(TOKENIZER_PATH) as f:
                tokenizer_keras = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
            print(f"[CNN] Tokenizer loaded from {TOKENIZER_PATH}")
        else:
            tokenizer_keras = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
            tokenizer_keras.fit_on_texts(texts)
            with open(TOKENIZER_PATH, "w") as f:
                f.write(tokenizer_keras.to_json())
            print(f"[CNN] Tokenizer fitted + saved → {TOKENIZER_PATH}")
    else:
        print("[CNN] Skipping tokenizer init — TensorFlow is missing.")

    return np.array(texts), np.array(labels, dtype=np.int32)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL INIT
# ─────────────────────────────────────────────────────────────────────────────
def init_model():
    global global_model
    if not TF_OK:
        print("[CNN] TF unavailable — rule-based fallback only")
        return
    global_model = create_cnn_model()
    saved = load_weights()
    if saved is not None:
        try:
            global_model.set_weights(saved)
            print(f"[CNN] Resumed weights from {WEIGHTS_PATH} ✓")
        except Exception as e:
            print(f"[CNN] Could not load weights ({e}) — fresh init")
    else:
        print("[CNN] No saved weights — model initialised randomly")
    print(f"[CNN] {global_model.count_params():,} parameters ✓")

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def run_predict(text):
    if not TF_OK or global_model is None or tokenizer_keras is None:
        label, conf = rule_based(text)
        p = {"ham": 0.0, "spam": 0.0, "smishing": 0.0}
        p[label] = conf
        return {"label": label, "confidence": conf, "probs": p, "mode": "rule-based"}
    seq   = texts_to_padded([clean(text)])
    probs = global_model.predict(seq, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return {
        "label"     : LABEL_MAP[idx],
        "confidence": round(float(probs[idx]) * 100, 1),
        "probs"     : {
            "ham"     : round(float(probs[0]) * 100, 1),
            "spam"    : round(float(probs[1]) * 100, 1),
            "smishing": round(float(probs[2]) * 100, 1),
        },
        "mode": "cnn",
    }

# ─────────────────────────────────────────────────────────────────────────────
# FEDAVG AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────
def do_fedavg():
    """
    Weighted FedAvg — each client's contribution is proportional to its
    number of training samples (standard FedAvg as in the McMahan 2017 paper).
    """
    global fl_round_counter, client_updates
    with lock:
        if not client_updates:
            return False, "no updates queued"
        snap = list(client_updates)
        client_updates = []

    total_samples = sum(u["n_samples"] for u in snap)
    if total_samples == 0:
        total_samples = len(snap)  # fallback: uniform weight

    # Weighted average: w_global = Σ (n_k / N) * w_k
    new_weights = []
    for layer_idx in range(len(snap[0]["weights"])):
        layer_avg = sum(
            (u["n_samples"] / total_samples) * u["weights"][layer_idx]
            for u in snap
        ).astype(np.float32)
        new_weights.append(layer_avg)

    if TF_OK and global_model is not None:
        global_model.set_weights(new_weights)

    with lock:
        fl_round_counter += 1
        rnd = fl_round_counter

    save_weights(new_weights)
    # Signal waiting clients that a new round is ready
    _round_done_event.set()
    _round_done_event.clear()

    client_names = [u['client_id'] for u in snap]
    print(f"\n[CNN FedAvg] ═══ Round {rnd} complete ═══")
    print(f"[CNN FedAvg]   Clients    : {client_names}")
    print(f"[CNN FedAvg]   Samples    : {total_samples}")
    print(f"[CNN FedAvg]   Weights saved → {WEIGHTS_PATH}\n")
    return True, f"round {rnd} — {len(snap)} clients — {total_samples} samples"

# ─────────────────────────────────────────────────────────────────────────────
# MESSAGE PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
def save_messages():
    try:
        with open(MESSAGES_PATH, "w") as f:
            json.dump(messages_db, f)
    except Exception as e:
        print(f"[CNN] Cannot save messages: {e}")

def load_messages():
    global messages_db
    if not os.path.exists(MESSAGES_PATH):
        return
    try:
        with open(MESSAGES_PATH) as f:
            messages_db = json.load(f)
        print(f"[CNN] Loaded {len(messages_db)} messages ✓")
    except Exception as e:
        print(f"[CNN] Cannot load messages: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# FLASK API  — same shape as flwr_server.py so index.html works unchanged
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Standard chat API ─────────────────────────────────────────────────────────

@app.route("/status")
def status():
    now = time.time()
    with lock:
        active  = {c: i for c, i in browser_sessions.items()
                   if now - i["last_seen"] < CLIENT_TIMEOUT}
        pending = len(client_updates)
    return jsonify({
        "ok"             : True,
        "fl_round"       : fl_round_counter,
        "fl_framework"   : "cnn_fedavg_http",
        "model"          : "multi_channel_cnn",
        "browser_clients": [{"client_id": c, "name": i["name"]} for c, i in active.items()],
        "total_messages" : len(messages_db),
        "model_loaded"   : global_model is not None,
        "checkpoint"     : os.path.exists(WEIGHTS_PATH),
        "pending_updates": pending,
        "pending_corrections": len(corrections_buffer),
    })

@app.route("/register", methods=["POST"])
def register():
    d = request.get_json(force=True)
    cid, name = d.get("client_id", "unknown"), d.get("name", "User")
    with lock:
        browser_sessions[cid] = {"name": name, "last_seen": time.time()}
    return jsonify({"ok": True, "client_id": cid, "fl_round": fl_round_counter})

@app.route("/get_clients")
def get_clients():
    with lock:
        now = time.time()
        active = []
        for cid, info in browser_sessions.items():
            status = "online" if now - info.get("last_seen", 0) < 60 else "offline"
            active.append({
                "client_id": cid,
                "name": info.get("name", "User"),
                "status": status
            })
    return jsonify({"clients": active})

@app.route("/predict", methods=["POST"])
def predict():
    d    = request.get_json(force=True)
    text = d.get("text", "").strip()
    if not text:
        return jsonify({"error": "empty"}), 400
    r = run_predict(text)
    r["fl_round"] = fl_round_counter
    return jsonify(r)

@app.route("/send_message", methods=["POST"])
def send_message():
    d      = request.get_json(force=True)
    sender = str(d.get("sender", "unknown")).strip()
    text   = str(d.get("text", "")).strip()
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
            "mode"      : pred.get("mode", "cnn"),
            "corrected" : False,
            "timestamp" : time.time(),
            "fl_round"  : fl_round_counter,
        }
        messages_db.append(msg)
        save_messages()
    print(f"[CNN] Msg #{msg['id']} {sender}: {text[:50]!r} → {pred['label']}")
    return jsonify(msg)

@app.route("/correct_label", methods=["POST"])
def correct_label():
    d       = request.get_json(force=True)
    msg_id  = int(d.get("msg_id", -1))
    cid     = d.get("client_id", "anonymous")
    correct = str(d.get("correct_label", "")).strip().lower()
    if correct not in ("ham", "spam", "smishing"):
        return jsonify({"error": "invalid label"}), 400
    lmap = {"ham": 0, "spam": 1, "smishing": 2}
    with lock:
        for msg in messages_db:
            if msg["id"] == msg_id:
                old  = msg["label"]
                text = msg["text"]
                msg["label"]     = correct
                msg["corrected"] = True
                save_messages()
                
                # ── Push correction into PER-CLIENT buffer ──────────────────────────
                with corrections_lock:
                    if cid not in corrections_buffer:
                        corrections_buffer[cid] = []
                    corrections_buffer[cid].append({
                        "text" : text,
                        "label": lmap[correct],
                    })
                save_corrections()
                print(f"[CNN] Msg #{msg_id} ({cid}): {old} → {correct}")
                return jsonify({"ok": True, "msg_id": msg_id, "new_label": correct})
    return jsonify({"error": f"msg_id {msg_id} not found"}), 404

@app.route("/get_messages")
def get_messages():
    since = int(request.args.get("since", 0))
    with lock:
        msgs = [m for m in messages_db if m["id"] > since]
    return jsonify({"messages": msgs, "fl_round": fl_round_counter,
                    "total": len(messages_db)})

@app.route("/get_corrections")
def get_corrections():
    """
    Returns corrections specific to a client_id.
    """
    cid  = request.args.get("client_id", "anonymous")
    peek = request.args.get("peek", "0") == "1"
    with corrections_lock:
        pending = list(corrections_buffer.get(cid, []))
        if not peek and cid in corrections_buffer:
            corrections_buffer[cid] = []
            save_corrections()
    print(f"[CNN] /get_corrections ({cid}): returned {len(pending)} item(s)")
    return jsonify({
        "corrections" : pending,
        "count"       : len(pending),
        "client_id"   : cid
    })

# ── FL-specific endpoints ─────────────────────────────────────────────────────

@app.route("/get_global_weights")
def get_global_weights():
    """FL clients call this to download the current global model weights."""
    if not TF_OK or global_model is None:
        return jsonify({"error": "model not ready"}), 503
    return jsonify({
        "weights_b64": weights_to_b64(global_model.get_weights()),
        "fl_round"   : fl_round_counter,
    })

@app.route("/submit_weights", methods=["POST"])
def submit_weights():
    """
    FL clients POST their locally-trained weights here.
    When queued count reaches args.min_clients, FedAvg is automatically
    triggered — no timer, no manual call needed.
    """
    d = request.get_json(force=True)
    b64       = d.get("weights_b64", "")
    cid       = d.get("client_id", "unknown")
    n_samples = int(d.get("n_samples", 1))   # sample count for weighted FedAvg

    if not b64:
        return jsonify({"error": "no weights"}), 400
    try:
        w = b64_to_weights(b64)
    except Exception as e:
        return jsonify({"error": f"deserialise failed: {e}"}), 400

    with lock:
        client_updates.append({"weights": w, "n_samples": n_samples, "client_id": cid})
        n = len(client_updates)

    print(f"[CNN] Weights received from {cid} (n_samples={n_samples}) "
          f"— queue: {n}/{args.min_clients}")

    # Auto-aggregate once ALL min_clients have submitted — no race condition
    if n >= args.min_clients:
        print(f"[CNN] All {args.min_clients} client(s) submitted — running FedAvg…")
        threading.Thread(target=do_fedavg, daemon=True).start()

    return jsonify({"ok": True, "queued": n,
                    "min_clients": args.min_clients,
                    "fl_round": fl_round_counter})

@app.route("/aggregate", methods=["POST"])
def aggregate():
    """Manually trigger FedAvg (UI button or override). Normally auto-triggered."""
    ok, msg = do_fedavg()
    if ok:
        return jsonify({"success": True, "fl_round": fl_round_counter, "message": msg})
    return jsonify({"success": False, "message": msg})

@app.route("/wait_for_round")
def wait_for_round():
    """
    Clients call this AFTER submitting weights to block until FedAvg completes.
    Returns immediately when the round_done event fires (or after timeout).
    This is the synchronization barrier that keeps all clients on the same round.
    """
    current = int(request.args.get("round", fl_round_counter))
    timeout = float(request.args.get("timeout", 300))   # default 5 min
    # Already advanced past the requested round — return immediately
    if fl_round_counter > current:
        return jsonify({"fl_round": fl_round_counter, "ready": True})
    # Block until aggregation finishes
    fired = _round_done_event.wait(timeout=timeout)
    return jsonify({"fl_round": fl_round_counter, "ready": fired})

@app.route("/get_tokenizer_config")
def get_tokenizer_config():
    """Return Keras tokenizer JSON so clients share the same vocabulary."""
    if tokenizer_keras is None:
        return jsonify({"error": "tokenizer not ready"}), 503
    return jsonify({"tokenizer_json": tokenizer_keras.to_json()})

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_messages()
    load_corrections()
    load_and_prepare()
    init_model()

    print("\n[CNN Server] ─────────────────────────────────────────")
    print(f"[CNN Server] Flask API  : http://0.0.0.0:{args.api_port}")
    print(f"[CNN Server] Weights    : {WEIGHTS_PATH}")
    print(f"[CNN Server] Tokenizer  : {TOKENIZER_PATH}")
    print(f"[CNN Server] Messages   : {MESSAGES_PATH}")
    print(f"[CNN Server] Model      : Multi-Channel CNN (3/4/5-gram)")
    print(f"[CNN Server] Min clients: {args.min_clients} (FedAvg triggers when all submit)")
    print(f"[CNN Server] FedAvg     : Weighted by n_samples (McMahan 2017)")
    print(f"[CNN Server] Sync       : Clients block on /wait_for_round between rounds")
    print(f"[CNN Server] Open cnn_index.html → Server URL: http://127.0.0.1:{args.api_port}")
    print("[CNN Server] ─────────────────────────────────────────\n")

    app.run(host="0.0.0.0", port=args.api_port, debug=False, threaded=True)
