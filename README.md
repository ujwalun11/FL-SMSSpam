# FedShield: Personalized Federated Learning for Smishing Detection

## 1. Project Overview
FedShield is a decentralized, privacy-preserving system designed to detect Smishing (SMS Phishing) attacks. It uses **Federated Learning (FL)** to train a global model on local user data without ever seeing the raw messages, ensuring complete user privacy.

## 2. Core Architecture: The "Personalized FL" (FedPer)
The system has moved from a standard global model to a **Personalized Federated Learning (FedPer)** architecture. This allows the system to balance a "Global Brain" (general security) with a "Personalized Head" (individual user preference).

### Layer Management (Freezing Logic)
*   **Frozen Backbone (Conv1D):** The convolutional layers are shared globally to learn general linguistic features of SPAM and HAM.
*   **Trainable Head (Embedding + Dense):** These layers are unfrozen during the personalization pass. Unfreezing the **Embedding** layer is critical as it allows the model to re-map specific OOV (Out of Vocabulary) terms (like a university URL or a work-related document link) based on the user's manual correction.

## 3. High-Performance CNN Model
We transitioned to a **Multi-Channel CNN** architecture, which proved more stable and faster than the previous Transformer model for local training.

### Model Specs:
*   **Channels:** 3 Parallel Convolutional Filters (Size 3, 4, and 5).
*   **Filters:** 128 per channel.
*   **Advantage:** Captures short-range word patterns (n-grams) simultaneously, making it highly effective at spotting URLs and "Urgent" keywords typical in Smishing.

## 4. Stability & Adaptation: The "Mega Microphone"
To ensure the model respects user corrections immediately, we implemented the **Mega Microphone** personalization strategy.

### Hyperparameters:
| Parameter | Value | Purpose |
| :--- | :--- | :--- |
| **History Weight** | **200x** | Forces the model to prioritize user corrections over global bias. |
| **Epochs** | **20** | Deep reinforcement of personalized labels. |
| **Learning Rate** | **0.02** | Rapid adaptation to specific local nuances. |
| **Anchor System** | **1 per class** | Maintains a minimal baseline of general knowledge while allowing total personalization. |
| **FedProx Mu ($\mu$)** | **0.1** | Prevents local models from drifting too far during global rounds. |

## 5. Technology Comparison
| Feature | Older Transformer Approach | Current CNN Approach |
| :--- | :--- | :--- |
| **Architecture** | DistilBERT / Transformer | **Multi-Channel Conv1D** |
| **Inference Speed** | Slow (Heavy) | **Very Fast (Lightweight)** |
| **Personalization** | Global-First | **User-Centric (FedPer)** |

## 6. Key Innovations
1.  **Long-Term Memory:** Implemented a persistent `corrections_{id}.json` history that re-feeds all historical corrections into every training pass.
2.  **Anchor System:** Balances personalization with a "moral compass" of baseline data to prevent the model from becoming biased toward a single label (e.g., everything becoming HAM).
3.  **Identity Sync:** Automatic synchronization between the Browser UI and Python client status to ensure corrections are attributed to the correct entity.

---
**Status:** Stable & Operational
**Last Updated:** April 2026
