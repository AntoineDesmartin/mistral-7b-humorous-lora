# 🤖 Mistral-7B Humorous LoRA

A lightweight LoRA adapter that gives **Mistral-7B-Instruct** a humorous personality (wit, sarcasm).
The project includes:
- A local **Gradio** chat app (`app.py`) to try it on your machine
- A **Colab notebook** with the full training pipeline (in `notebook/`)
- A pointer to the **LoRA adapter on Hugging Face Hub**

> Adapter repo (LoRA weights): **[`tonioexe/mistral-7b-humorous-lora`](https://huggingface.co/tonioexe/mistral-7b-humorous-lora)**

---

## 🚀 Quickstart (Local)

### 1) Requirements
- **GPU with CUDA** recommended (e.g., RTX 3060/3080/T4, etc.).  
- **Hugging Face access** to the base model (gated):  
  Go to https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 and click **"Agree & access"**.

### 2) Create a virtual env and install deps
```bash
git clone https://github.com/tonioexe/mistral-humor-lora.git
cd mistral-humor-lora

# Python virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install PyTorch matching your CUDA (example: CUDA 12.1)
# Check: https://pytorch.org/get-started/locally/
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install the rest
pip install -r requirements.txt
```

3) Run the local chat
```python app.py```


# 🧪 Example prompt

```How do I politely cancel plans I never wanted in the first place ?```

```Easy: Just don’t. If they call, feign illness. If they text, ignore. If they show up, pretend you were in the shower. If they stay, ask for their phone number. Then block it. Congratulations, you’re now a lone wolf. Powered by caffeine and tears. Cheap, efficient, soul-crushing. Great until you remember mortality. Then it’s just sad.```


# 🧠 About the Model

- Base: mistralai/Mistral-7B-Instruct-v0.2 (Apache-2.0, gated access)

- Method: LoRA (PEFT + Hugging Face Transformers)

- Adapter size: ~53 MB

- Training: Google Colab (T4 GPU), ~800 instruction/response pairs, 3 epochs

- Eval: Perplexity ~7.75 on a small held-out set

The adapter changes style (humor) without forgetting the base model knowledge.


# 📚 Training (Notebook)

See notebook/LoRA-Mistral-Humor.ipynb for:

- Data loading (JSONL with fields: instruction, input (often empty), output)

- Tokenization with the official chat template

- 4-bit loading (bitsandbytes) + LoRA with PEFT

- Trainer loop (Transformers)

- Evaluation (perplexity + sample generations)

- Saving and pushing the adapter to the Hub