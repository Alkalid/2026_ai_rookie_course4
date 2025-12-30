# common_setup.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

set_seed(42)

# 根據需要可以從環境變數覆寫
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def has_cuda():
    return torch.cuda.is_available()

def print_env_info():
    print("PyTorch:", torch.__version__)
    print("GPU 可用:", has_cuda())
    if has_cuda():
        print("GPU 名稱:", torch.cuda.get_device_name(0))
        print("GPU 顯存(GB):", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))

def load_model_and_tokenizer(model_id: str = BASE_MODEL_ID, load_in_4bit: bool = True):
    """
    載入 tokenizer 與模型。
    預設使用 4-bit 量化（若 GPU 可用），以節省顯存。
    """
    print(f"載入模型: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    bnb_cfg = None
    device_map = "auto"
    if has_cuda() and load_in_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if has_cuda() else torch.float32,
        quantization_config=bnb_cfg,
        device_map=device_map,
    )
    model.eval()
    return tokenizer, model

if __name__ == "__main__":
    print_env_info()
    tok, m = load_model_and_tokenizer()
    print("Tokenizer vocab size:", tok.vocab_size)