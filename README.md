# GPT-2 Decoder from Scratch 🧠

A minimal, educational implementation of a **GPT-2 style decoder transformer** built entirely in PyTorch. This project strips away high-level abstractions to reveal the inner workings of modern Large Language Models.

---

### 🚀 Features
* **Core Architecture**: Multi-Head Self-Attention, GELU Feed-Forward Networks, and Residual Connections.
* **Embeddings**: Learned Token + Absolute Positional Embeddings.
* **Inference**: Autoregressive generation with **Temperature** and **Top-K** sampling.
* **Training**: Causal language modeling pipeline using **AdamW** and **Cross-Entropy Loss**.

---

### 🏗️ The Architecture
The model processes sequences through a standard decoder-only stack:
1.  **Input**: Tokenization via `tiktoken`.
2.  **Embedding**: $\text{Token} + \text{Position}$.
3.  **Transformer Blocks**: $N \times$ (Masked Multi-Head Attention → LayerNorm → MLP).
4.  **Output**: Linear head projecting to vocabulary logits.

---

### 📊 Tech Stack
| Component | Tool |
| :--- | :--- |
| **Language** | Python 3.x |
| **Framework** | PyTorch |
| **Tokenizer** | tiktoken (BPE) |
| **Visualization** | Matplotlib |

---

### 📝 Quick Start
**Example Generation:**
> **Prompt:** `Every effort moves you`  
> **Output:** `Every effort moves you closer toward forward effort moves you closer`

**File Structure:**
* `gpt2_decoder.ipynb`: The complete implementation & walkthrough.
* `the-verdict.txt`: Sample dataset for local training.

---

### 🛠️ Future Roadmap
- [ ] Implement **Top-P (Nucleus) Sampling**.
- [ ] Integrate **Flash Attention** for optimization.
- [ ] Scale to larger datasets (e.g., TinyStories).
