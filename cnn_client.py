import os
import time
import json
import base64
import argparse
import threading
import requests
import numpy as np
import pandas as pd
from flask import Flask, request as freq, jsonify
from flask_cors import CORS

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_OK = True
except ImportError:
    TF_OK = False
    print("[WARN] TensorFlow not found.")

try:
    import nlpaug.augmenters.char as nac
    NLPAUG_OK = True
except ImportError:
    NLPAUG_OK = False
    print("[WARN] nlpaug not installed.")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & ARGS
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="FedShield CNN FL Client")
parser.add_argument("--id",     required=True, choices=["client_1","client_2","client_3"])
parser.add_argument("--server", default="http://localhost:5050")
parser.add_argument("--rounds", type=int, default=5)
parser.add_argument("--port",   type=int, default=None)
parser.add_argument("--fast",   action="store_true")
args = parser.parse_args()

DATA_FILE  = "data_cleaned.csv"
VOCAB_SIZE = 10000
MAX_LEN    = 300
EMBED_DIM  = 128
NUM_CLASSES= 3
LABEL_MAP  = {0: "ham", 1: "spam", 2: "smishing"}

if args.fast:
    LOCAL_EPOCHS = 1; BATCH_SIZE = 16; MAX_SAMPLES = 200
else:
    LOCAL_EPOCHS = 5; BATCH_SIZE = 32; MAX_SAMPLES = None

DEFAULT_PORTS = {"client_1": 5051, "client_2": 5052, "client_3": 5053}
LOCAL_PORT = args.port or DEFAULT_PORTS[args.id]

# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def clean(text):
    return str(text).lower().strip() if text else ""

def weights_to_b64(weights):
    return base64.b64encode(json.dumps([w.tolist() for w in weights]).encode()).decode()

def b64_to_weights(b64_str):
    return [np.array(w, dtype=np.float32) for w in json.loads(base64.b64decode(b64_str).decode())]

def create_cnn_model():
    """RESTORED: Multi-filter CNN Architecture (3, 4, 5)"""
    inp = layers.Input(shape=(MAX_LEN,))
    emb = layers.Embedding(VOCAB_SIZE, EMBED_DIM)(inp)
    
    # 3 Parallel Convolutional Filters
    c3 = layers.Conv1D(128, 3, activation="relu", padding="same")(emb)
    p3 = layers.GlobalMaxPooling1D()(c3)
    
    c4 = layers.Conv1D(128, 4, activation="relu", padding="same")(emb)
    p4 = layers.GlobalMaxPooling1D()(c4)
    
    c5 = layers.Conv1D(128, 5, activation="relu", padding="same")(emb)
    p5 = layers.GlobalMaxPooling1D()(c5)
    
    conc = layers.Concatenate()([p3, p4, p5])
    drop = layers.Dropout(0.3)(conc)
    d1   = layers.Dense(64, activation="relu")(drop)
    out  = layers.Dense(NUM_CLASSES, activation="softmax")(d1)
    
    return models.Model(inputs=inp, outputs=out)

def _gen_smishing(n=100):
    import random, string
    tmpl = ["urgent: your bank account suspended. verify at http://{r}.com", 
            "tax refund pending. claim at http://tax-{r}.net"]
    return [random.choice(tmpl).format(r="".join(random.choices(string.ascii_lowercase, k=5))) for _ in range(n)]

# ─────────────────────────────────────────────────────────────────────────────
# CLIENT CLASS
# ─────────────────────────────────────────────────────────────────────────────
class CNNFedClient:
    def __init__(self, client_id, server_url):
        self.id       = client_id
        self.server   = server_url.rstrip("/")
        self.model    = create_cnn_model()
        self.tok      = None
        self.buffer   = []
        self.buf_lock = threading.Lock()
        self.round    = 0
        self.history_path = f"corrections_{self.id}.json"
        self.all_corrections = self.load_history()

        self._sync_tokenizer()
        self._load_data()
        
        if self.model:
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                               loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def load_history(self):
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r") as f: return json.load(f)
            except: return []
        return []

    def save_history(self):
        with open(self.history_path, "w") as f: json.dump(self.all_corrections, f)

    def _sync_tokenizer(self):
        try:
            r = requests.get(f"{self.server}/get_tokenizer_config")
            if r.ok: self.tok = tf.keras.preprocessing.text.tokenizer_from_json(r.json()["tokenizer_json"])
        except: self.tok = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")

    def _load_data(self):
        if not os.path.exists(DATA_FILE):
            self.base_texts, self.base_labels = [], []
            return
        df = pd.read_csv(DATA_FILE, sep=";", index_col=0).dropna(subset=["class", "message"])
        n = len(df)
        s, e = {"client_1":(0, n//3), "client_2":(n//3, 2*n//3), "client_3":(2*n//3, n)}.get(self.id, (0,n))
        df = df.iloc[s:e]
        lmap = {"ham":0, "spam":1, "smishing":2}
        self.base_texts = [clean(t) for t in df["message"]]
        self.base_labels= [lmap.get(str(c).lower(), 0) for c in df["class"]]
        # Baseline Smishing knowledge
        smish = _gen_smishing(100)
        self.base_texts += [clean(t) for t in smish]
        self.base_labels += [2] * len(smish)

    def pull_weights(self):
        try:
            r = requests.get(f"{self.server}/get_global_weights")
            if r.ok:
                gw = b64_to_weights(r.json()["weights_b64"])
                lw = self.model.get_weights()
                # Backbone (Embedding + 3x Conv) = First 8 matrices (Weights + Bias x 4)
                for i in range(8): lw[i] = gw[i]
                self.model.set_weights(lw)
                print(f"[{self.id}] Pulled Global Backbone ✓")
        except: pass

    def push_weights(self, n):
        requests.post(f"{self.server}/submit_weights", 
                      json={"weights_b64": weights_to_b64(self.model.get_weights()), 
                            "client_id": self.id, "n_samples": n})

    def train_local(self, epochs=None):
        if epochs is None: epochs = LOCAL_EPOCHS
        mu = 0.1
        global_w = [tf.constant(w) for w in self.model.get_weights()]
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        # Merge new corrections
        try:
            r = requests.get(f"{self.server}/get_corrections?client_id={self.id}")
            sc = r.json().get("corrections", []) if r.ok else []
        except: sc = []
        
        texts = self.base_texts.copy()
        labels = self.base_labels.copy()
        with self.buf_lock:
            for c in (self.buffer + sc):
                texts.append(clean(c["text"])); labels.append(int(c["label"]))
            self.buffer = []

        X = pad_sequences(self.tok.texts_to_sequences(texts), maxlen=MAX_LEN, padding="post")
        y = np.array(labels, dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(len(X)).batch(BATCH_SIZE)

        print(f"[{self.id}] Training Round... ({len(texts)} samples)")
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in ds:
                with tf.GradientTape() as tape:
                    logits = self.model(xb, training=True)
                    loss = loss_fn(yb, logits)
                    prox = sum([tf.nn.l2_loss(lw - gw) for lw, gw in zip(self.model.trainable_variables, global_w)])
                    loss += (mu * prox)
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                total_loss += loss
            print(f"  Epoch {epoch+1}/{epochs} - Avg Loss: {total_loss/len(ds):.4f}")
        return len(texts)

    def run_round(self, rnd):
        self.pull_weights()
        n = self.train_local()
        if n > 0:
            self.push_weights(n)
            requests.get(f"{self.server}/wait_for_round", params={"round": rnd, "timeout": 300})

    def predict(self, text):
        X = pad_sequences(self.tok.texts_to_sequences([clean(text)]), maxlen=MAX_LEN, padding="post")
        probs = self.model.predict(X, verbose=0)[0]
        idx = int(np.argmax(probs))
        return {"label": LABEL_MAP[idx], "confidence": round(float(probs[idx])*100, 1),
                "probs": {LABEL_MAP[i]: round(float(probs[i])*100, 1) for i in range(3)}}

# ─────────────────────────────────────────────────────────────────────────────
# MINI SERVER
# ─────────────────────────────────────────────────────────────────────────────
def run_mini_server(client, port):
    mini = Flask(f"mini_{client.id}")
    CORS(mini)

    @mini.route("/local_predict", methods=["POST"])
    def lp(): return jsonify(client.predict(freq.get_json(force=True).get("text", "")))

    @mini.route("/manual_train", methods=["POST"])
    def mt():
        try:
            r = requests.get(f"{client.server}/get_corrections?client_id={client.id}")
            sc = r.json().get("corrections", []) if r.ok else []
        except: sc = []
        
        with client.buf_lock:
            for n in (client.buffer + sc):
                if not any(c["text"] == clean(n["text"]) for c in client.all_corrections):
                    client.all_corrections.append({"text": clean(n["text"]), "label": int(n["label"])})
            client.buffer = []; client.save_history()
            history = list(client.all_corrections)

        if not history: return jsonify({"ok": True, "msg": "No history"})

        # ── THE MEGA MICROPHONE ──
        # 200x weight for history + 20 epochs + 0.02 Learning Rate.
        # This gives you absolute control over your local brain.
        n_anchor = 1 
        h_idx = [i for i, l in enumerate(client.base_labels) if l == 0][:n_anchor]
        s_idx = [i for i, l in enumerate(client.base_labels) if l == 1][:n_anchor]
        m_idx = [i for i, l in enumerate(client.base_labels) if l == 2][:n_anchor]
        
        anchor_texts  = [client.base_texts[i] for i in (h_idx + s_idx + m_idx)]
        anchor_labels = [client.base_labels[i] for i in (h_idx + s_idx + m_idx)]

        # Personalize
        for layer in client.model.layers:
            name = layer.name.lower()
            layer.trainable = ("dense" in name or "embedding" in name)
        
        client.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
                             loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        
        # Combine history (weighted 200x) + anchors
        final_texts  = anchor_texts + ([c["text"] for c in history] * 200)
        final_labels = anchor_labels + ([c["label"] for c in history] * 200)
        
        cX = pad_sequences(client.tok.texts_to_sequences(final_texts), maxlen=MAX_LEN, padding="post")
        cy = np.array(final_labels, dtype=np.int32)
        
        print(f"[{client.id}] 🚀 MEGA MICROPHONE ({len(history)} items @ 200x weight)...")
        client.model.fit(cX, cy, epochs=20, verbose=1)
        
        for layer in client.model.layers: layer.trainable = True
        client.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                             loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return jsonify({"ok": True, "history_size": len(history)})

    @mini.route("/buffer_message", methods=["POST"])
    def bm():
        d = freq.get_json(force=True)
        client.add_to_buffer(d.get("text"), d.get("label"))
        return jsonify({"ok": True})

    @mini.route("/client_status")
    def cs():
        return jsonify({"client_id": client.id, "round": client.round, "history": len(client.all_corrections)})

    mini.run(host="127.0.0.1", port=port, threaded=True)

if __name__ == "__main__":
    print(f"\n--- FedShield Client {args.id} ---")
    c = CNNFedClient(args.id, args.server)
    threading.Thread(target=run_mini_server, args=(c, LOCAL_PORT), daemon=True).start()
    for r in range(1, args.rounds + 1): c.run_round(r)
    while True: time.sleep(60)
