"""
=============================================================================
  FedShield - Flower FL Client
  File: flwr_client.py

  INSTALL:
    pip install flwr torch transformers flask flask-cors numpy pandas

  RUN (fast mode):
    python flwr_client.py --id client_1 --server localhost:8080 --chat http://localhost:5000 --port 5001 --fast

  RUN (full mode):
    python flwr_client.py --id client_1 --server localhost:8080 --chat http://localhost:5000 --port 5001

  NOTE:
    --server  Flower server address (NO http://)  e.g. localhost:8080
    --chat    Flask chat API URL (WITH http://)   e.g. http://localhost:5000

  HOW IT DIFFERS FROM fl_client.py:
    Subclasses flwr.client.NumPyClient instead of manual HTTP
    get_parameters() returns weights to Flower server
    set_parameters() applies global weights from server (with proof)
    fit() trains locally and returns updated weights (with proof)
    Weights sent as float32 - no float16 precision loss
    No pull_weights/push_weights/encode_weights/decode_chunks needed
    Flower handles all communication and round coordination
    Mini Flask server for chat buffering is identical to fl_client.py
=============================================================================
"""

import argparse, os, re, sys, time, threading
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from transformers import (DistilBertTokenizer, DistilBertModel,
                              get_linear_schedule_with_warmup)
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError as e:
    print(f"[ERROR] Missing: {e}")
    print("Run: pip install torch transformers flask flask-cors pandas numpy")
    sys.exit(1)

try:
    import flwr as fl
    from flwr.common import NDArrays
except ImportError:
    print("[ERROR] flwr not installed. Run: pip install flwr")
    sys.exit(1)

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--id",     required=True, choices=["client_1","client_2","client_3"])
parser.add_argument("--server", required=True,
                    help="Flower server address e.g. localhost:8080")
parser.add_argument("--chat",   required=True,
                    help="Chat API URL e.g. http://localhost:5000")
parser.add_argument("--port",   type=int, default=None)
parser.add_argument("--fast",   action="store_true")
args = parser.parse_args()

# Config
MODEL_FILE      = "best_model_dataset5971.pt"
DATA_FILE       = "data_cleaned.csv"
DISTILBERT      = "distilbert-base-uncased"
NUM_FEAT_SIZE   = 17
NUM_CLASSES     = 3
CHECKPOINT_FILE = f"flwr_trained_{args.id}.pt"

if args.fast:
    MAX_LEN=64; BATCH_SIZE=8; LOCAL_EPOCHS=1; LR=2e-5; MAX_SAMPLES=200
    print("[Config] FAST MODE  MAX_LEN=64 BATCH=8 EPOCHS=1 SAMPLES=200")
else:
    MAX_LEN=160; BATCH_SIZE=16; LOCAL_EPOCHS=2; LR=2e-5; MAX_SAMPLES=None
    print("[Config] FULL MODE  MAX_LEN=160 BATCH=16 EPOCHS=2")

# Model (identical to fl_client.py)
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

def extract_features(text):
    t = str(text); n = max(len(t), 1)
    return [
        len(t), len(t.split()),
        sum(c.isdigit() for c in t)/n, sum(c.isupper() for c in t)/n,
        int(bool(re.search(r"[PS$EU+]", t))), t.count("!"),
        len(re.findall(r"http\S+|www\.\S+", t, re.I)),
        len(re.findall(r"\b\d{7,}\b", t)),
        int(bool(re.search(r"\bfree\b",      t, re.I))),
        int(bool(re.search(r"\bwin(ner)?\b", t, re.I))),
        int(bool(re.search(r"\bprize\b",     t, re.I))),
        int(bool(re.search(r"\burgent\b",    t, re.I))),
        int(bool(re.search(r"\bclick\b",     t, re.I))),
        int(bool(re.search(r"\bcall\b",      t, re.I))),
        int(bool(re.search(r"http\S+|www\.\S+", t, re.I))),
        int(bool(re.search(r"\S+@\S+\.\S+",   t, re.I))),
        int(bool(re.search(r"\b\d{7,}\b",       t))),
    ]

def clean_text(text):
    t = str(text).encode("ascii","ignore").decode("ascii")
    return re.sub(r"\s+"," ",t).strip()

class SMSDataset(Dataset):
    def __init__(self, texts, feats, labels, tok, max_len):
        self.texts=texts; self.feats=feats; self.labels=labels
        self.tok=tok; self.max_len=max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], max_length=self.max_len,
                       padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids":     enc["input_ids"].squeeze(0),
                "attention_mask":enc["attention_mask"].squeeze(0),
                "num_feats":     torch.tensor(self.feats[i], dtype=torch.float),
                "labels":        torch.tensor(self.labels[i], dtype=torch.long)}


class FedShieldClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for FedShield.
    Flower calls get_parameters / set_parameters / fit each round.
    We do NOT manually pull/push weights — Flower handles transport.
    """
    def __init__(self, client_id):
        self.id        = client_id
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok       = None
        self.model     = None
        self.buffer    = []
        self.buf_lock  = threading.Lock()
        self.round_num = 0
        self._init_model()
        self._load_data()

    def _init_model(self):
        print(f"\n[{self.id}] Loading model on {self.device}...")
        self.tok   = DistilBertTokenizer.from_pretrained(DISTILBERT)
        self.model = DistilBertWithFeatures(
            DISTILBERT, NUM_FEAT_SIZE, NUM_CLASSES).to(self.device)
        if os.path.exists(CHECKPOINT_FILE):
            self.model.load_state_dict(
                torch.load(CHECKPOINT_FILE, map_location=self.device))
            print(f"[{self.id}] Resumed from {CHECKPOINT_FILE}")
        elif os.path.exists(MODEL_FILE):
            self.model.load_state_dict(
                torch.load(MODEL_FILE, map_location=self.device))
            print(f"[{self.id}] Loaded {MODEL_FILE}")
        else:
            print(f"[{self.id}] WARNING: no .pt file found - random init")
        self.model.eval()
        n = sum(p.numel() for p in self.model.parameters())
        print(f"[{self.id}] Model ready - {n:,} parameters")

    def _load_data(self):
        if not os.path.exists(DATA_FILE):
            print(f"[{self.id}] {DATA_FILE} not found - chat msgs only")
            self.base_texts=[]; self.base_feats=np.empty((0,NUM_FEAT_SIZE),dtype=np.float32)
            self.base_labels=[]; return
        df = pd.read_csv(DATA_FILE, sep=";", index_col=0)
        df = df.dropna(subset=["class","message"]).reset_index(drop=True)
        n  = len(df)
        slices = {"client_1":(0,n//3),"client_2":(n//3,2*n//3),"client_3":(2*n//3,n)}
        s, e   = slices.get(self.id, (0, n))
        df     = df.iloc[s:e].reset_index(drop=True)
        if MAX_SAMPLES and len(df) > MAX_SAMPLES:
            df = df.sample(MAX_SAMPLES, random_state=42).reset_index(drop=True)
        df["label"] = df["class"].str.lower().map({"ham":0,"spam":1}).fillna(0).astype(int)
        self.base_texts  = [clean_text(t) for t in df["message"]]
        self.base_feats  = np.array([extract_features(t) for t in df["message"]], dtype=np.float32)
        self.base_labels = df["label"].tolist()
        print(f"[{self.id}] Data: {len(self.base_texts)} samples (rows {s}-{e})")

    # ── FLOWER METHOD 1 ───────────────────────────────────────────────────────
    def get_parameters(self, config):
        """Return current model weights as list of numpy float32 arrays."""
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    # ── FLOWER METHOD 2 ───────────────────────────────────────────────────────
    def set_parameters(self, parameters):
        """Apply global weights from server. Prints weight-change proof."""
        old_state = {k: v.cpu().numpy().copy()
                     for k, v in self.model.state_dict().items()}
        keys      = list(self.model.state_dict().keys())
        new_state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        self.model.load_state_dict(new_state)
        self.model.eval()

        # WEIGHT CHANGE PROOF
        total_l2 = sum(
            float(np.linalg.norm(new_state[k].numpy() - old_state[k]))
            for k in keys)
        changed  = sum(
            1 for k in keys
            if float(np.linalg.norm(new_state[k].numpy() - old_state[k])) > 1e-8)
        cls_keys = [k for k in keys if "classifier" in k and "weight" in k]
        if cls_keys:
            ck     = cls_keys[-1]
            cls_l2 = float(np.linalg.norm(new_state[ck].numpy() - old_state[ck]))
            cls_mx = float(np.abs(new_state[ck].numpy() - old_state[ck]).max())
        else:
            cls_l2 = cls_mx = 0.0
        print(f"\n[{self.id}] Global weights received from server:")
        print(f"[{self.id}]   Layers changed      : {changed}/{len(keys)}")
        print(f"[{self.id}]   Total L2 change     : {total_l2:.6f}")
        print(f"[{self.id}]   Classifier L2       : {cls_l2:.6f}")
        print(f"[{self.id}]   Classifier max dw   : {cls_mx:.8f}")
        if total_l2 < 1e-8:
            print(f"[{self.id}]   (Round 1 - same as local init)")
        else:
            print(f"[{self.id}]   Global model applied")

    # ── FLOWER METHOD 3 ───────────────────────────────────────────────────────
    def fit(self, parameters, config):
        """
        Flower calls this once per round.
        1. set_parameters: apply global weights (with proof)
        2. train locally
        3. save checkpoint
        4. return updated weights + num_samples
        """
        server_round   = config.get("server_round", self.round_num + 1)
        self.round_num = server_round
        print(f"\n[{self.id}] ====== Flower Round {server_round} ======")
        t0 = time.time()

        self.set_parameters(parameters)
        n_samples = self._train_local()
        if n_samples > 0:
            self._save_checkpoint()

        updated = self.get_parameters(config={})
        print(f"[{self.id}] ====== Round {server_round} done ({time.time()-t0:.0f}s) ======\n")
        return updated, n_samples, {"client_id": self.id}

    def _train_local(self):
        with self.buf_lock:
            buf_snap = list(self.buffer)
        texts  = self.base_texts  + [b["text"]  for b in buf_snap]
        labels = self.base_labels + [b["label"] for b in buf_snap]
        feats  = (np.concatenate([self.base_feats,
                                   np.array([b["feats"] for b in buf_snap], dtype=np.float32)])
                  if buf_snap else self.base_feats.copy())
        total  = len(texts)
        if total == 0:
            print(f"[{self.id}] No data - skipping. Send chat messages to add data.")
            return 0
        print(f"[{self.id}] Training on {total} samples (CSV:{len(self.base_texts)} chat:{len(buf_snap)})")

        ds     = SMSDataset(texts, feats, labels, self.tok, MAX_LEN)
        loader = DataLoader(ds, batch_size=min(BATCH_SIZE, total), shuffle=True, num_workers=0)
        opt    = AdamW([
            {"params": self.model.bert.parameters(),        "lr": LR},
            {"params": self.model.feat_proj.parameters(),   "lr": LR * 10},
            {"params": self.model.classifier.parameters(),  "lr": LR * 10},
        ], weight_decay=0.01)
        sched   = get_linear_schedule_with_warmup(opt, 0, len(loader) * LOCAL_EPOCHS)
        loss_fn = nn.CrossEntropyLoss()

        before  = {k: v.cpu().numpy().copy() for k, v in self.model.state_dict().items()}
        self.model.train()
        for ep in range(LOCAL_EPOCHS):
            ep_loss = 0.0
            for b in loader:
                ids = b["input_ids"].to(self.device)
                msk = b["attention_mask"].to(self.device)
                nf  = b["num_feats"].to(self.device)
                lbl = b["labels"].to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(ids, msk, nf), lbl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(); sched.step()
                ep_loss += loss.item()
            print(f"[{self.id}]   Epoch {ep+1}/{LOCAL_EPOCHS} loss={ep_loss/len(loader):.4f}")
        self.model.eval()

        # LOCAL TRAINING PROOF
        after   = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
        tot_l2  = sum(float(np.linalg.norm(after[k] - before[k])) for k in before)
        cls_k   = [k for k in before if "classifier" in k and "weight" in k][-1]
        cls_l2  = float(np.linalg.norm(after[cls_k] - before[cls_k]))
        cls_max = float(np.abs(after[cls_k] - before[cls_k]).max())
        print(f"[{self.id}] LOCAL TRAINING PROOF:")
        print(f"[{self.id}]   Total weight L2 change : {tot_l2:.6f}")
        print(f"[{self.id}]   Classifier L2 change   : {cls_l2:.6f}")
        print(f"[{self.id}]   Classifier max dw      : {cls_max:.8f}")
        if tot_l2 < 1e-8:
            print(f"[{self.id}]   WARNING: no weight change")
        else:
            print(f"[{self.id}]   Local weights UPDATED - sending to server")
        return total

    def _save_checkpoint(self):
        try:
            torch.save(self.model.state_dict(), CHECKPOINT_FILE)
            print(f"[{self.id}] Checkpoint saved to {CHECKPOINT_FILE}")
        except Exception as e:
            print(f"[{self.id}] Checkpoint save failed: {e}")

    def add_to_buffer(self, text, label_int):
        with self.buf_lock:
            self.buffer.append({
                "text" : clean_text(text),
                "feats": extract_features(text),
                "label": label_int})
        print(f"[{self.id}] Buffered (label={label_int}) buffer={len(self.buffer)}")

    def predict(self, text):
        enc = self.tok(clean_text(text), max_length=MAX_LEN,
                       padding="max_length", truncation=True, return_tensors="pt")
        nf  = torch.tensor([extract_features(text)], dtype=torch.float).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(enc["input_ids"].to(self.device),
                                enc["attention_mask"].to(self.device), nf)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        lmap  = {0:"ham", 1:"spam", 2:"smishing"}
        return {"label":lmap[int(probs.argmax())],
                "confidence":round(float(probs.max())*100, 1),
                "probs":{"ham":round(float(probs[0])*100,1),
                         "spam":round(float(probs[1])*100,1),
                         "smishing":round(float(probs[2])*100,1)}}


def run_mini_server(client, port):
    mini = Flask(f"mini_{client.id}")
    CORS(mini, resources={r"/*": {"origins": "*"}})

    @mini.route("/local_predict", methods=["POST"])
    def local_predict():
        data = request.get_json(force=True)
        text = data.get("text","").strip()
        if not text: return jsonify({"error":"empty"}), 400
        return jsonify(client.predict(text))

    @mini.route("/buffer_message", methods=["POST"])
    def buffer_message():
        data  = request.get_json(force=True)
        text  = data.get("text","")
        label = int(data.get("label",0))
        client.add_to_buffer(text, label)
        return jsonify({"ok":True, "buffer_size":len(client.buffer)})

    @mini.route("/client_status")
    def client_status():
        return jsonify({"client_id":client.id,"buffer_size":len(client.buffer),
                        "device":str(client.device),"round_num":client.round_num,
                        "checkpoint":os.path.exists(CHECKPOINT_FILE),
                        "fast_mode":args.fast, "fl_backend":"flower"})

    print(f"[{client.id}] Mini-server on http://0.0.0.0:{port}")
    mini.run(host="0.0.0.0", port=port, debug=False, threaded=True)


default_ports = {"client_1":5001,"client_2":5002,"client_3":5003}
port = args.port or default_ports[args.id]

print(f"\n{'='*60}")
print(f"  FedShield Flower Client - {args.id}")
print(f"  Flower server : {args.server}")
print(f"  Chat API      : {args.chat}")
print(f"  Local port    : {port}")
print(f"  Mode          : {'FAST' if args.fast else 'FULL'}")
print(f"  Checkpoint    : {CHECKPOINT_FILE}")
print(f"  FL backend    : Flower (flwr)")
print(f"{'='*60}\n")

flower_client = FedShieldClient(args.id)

mini_thread = threading.Thread(
    target=run_mini_server, args=(flower_client, port), daemon=True)
mini_thread.start()
time.sleep(1)

print(f"[{args.id}] Connecting to Flower server at {args.server}...")
fl.client.start_numpy_client(server_address=args.server, client=flower_client)
