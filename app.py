import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login as hf_login

BASE = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
ADAPTER = os.environ.get("ADAPTER_ID", "tonioexe/mistral-7b-humorous-lora")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "200"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
USE_4BIT = os.environ.get("USE_4BIT", "1") == "1"  
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")  

if HF_TOKEN:
    try:
        hf_login(HF_TOKEN)
    except Exception:
        pass

print(f"Base model: {BASE}")
print(f"Adapter:    {ADAPTER}")
print(f"4-bit:      {USE_4BIT}")

quant_cfg = None
device_map = "auto"
if torch.cuda.is_available() and USE_4BIT:
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map=device_map,
    trust_remote_code=True,
    quantization_config=quant_cfg
)

model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()
model.config.use_cache = True

def chat_fn(message, history):
    if not message or not message.strip():
        return "Say something witty, I dare you."

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": message.strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
        )

    reply = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    if reply == "":
        reply = "(…deadpan silence…)"

    return reply

demo = gr.ChatInterface(
    chat_fn,
    title="🤖 Mistral-7B — Humorous LoRA",
    description="Chat locally with a Mistral-7B model fine-tuned (LoRA) for humor.",
)

if __name__ == "__main__":
    demo.launch(share=False)
