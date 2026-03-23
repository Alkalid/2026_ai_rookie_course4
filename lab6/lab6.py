# lab6_ablation_and_packaging.py
import json
import textwrap
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
from ..lab5.lab5 import evaluate_one

_LAB6_DIR = Path(__file__).resolve().parent
_WORKDIR = _LAB6_DIR / "workdir"

def load_model_for_inference(base_model_id: str, adapter_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or base_model_id, use_fast=True)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model

def generate_correct(tokenizer, model, messages) -> str:
    """
    正確模板：使用 apply_chat_template + system。
    """
    # TODO 1: 用與 Lab5 相同方式實作
    return ""

def generate_wrong(tokenizer, model, messages) -> str:
    """
    錯誤模板：故意不用 chat template，只拼接 user 的內容。
    """
    # TODO 2: 將所有 user content 串起來，手寫 prompt，例如：
    # prompt = f"請用繁體中文回答：\n{user_text}\n回答："
    return ""

def run_template_ablation(test_examples: List[Dict[str, Any]], tokenizer, model, max_samples: int = 5):
    """
    對前幾筆測試資料做正確模板 vs 錯誤模板的比較。
    """
    for ex in test_examples[:max_samples]:
        good = generate_correct(tokenizer, model, ex["messages"])
        bad = generate_wrong(tokenizer, model, ex["messages"])

        eval_good = evaluate_one(ex, good)
        eval_bad = evaluate_one(ex, bad)

        print(f"\n[{ex['id']}] 正確模板 分數={eval_good['score']:.2f}, 錯誤模板 分數={eval_bad['score']:.2f}")
        print("正確模板回覆：", textwrap.shorten(good, width=200, placeholder=" ..."))
        print("錯誤模板回覆：", textwrap.shorten(bad, width=200, placeholder=" ..."))

def write_inference_script(base_model_id: str, adapter_dir: str, path: str):
    """
    產生一個簡單的推理腳本 inference.py，以方便 CLI 使用。
    """
    # TODO 3: 寫入一段 Python 程式碼到 path
    # 內容可包含 argparse 參數：--model_id, --adapter_dir, --system, --user
    code = """\
# TODO: 在這裡放入實際推理程式碼骨架
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

def write_readme(base_model_id: str, path: str):
    """
    產生一個 README_delivery.txt，說明如何使用模型。
    """
    # TODO 4: 說明基礎模型、LoRA 路徑、推理指令範例、注意事項
    readme = f"""\
專案說明（SFT 指令微調交付）
- 基礎模型: {base_model_id}
- LoRA/QLoRA 權重路徑: workdir/adapter

使用方式：
1) 在專案根目錄執行 `uv sync` 安裝依賴（含 transformers、peft、accelerate、bitsandbytes 等）。
2) 執行推理腳本，例如：
   python inference.py --model_id "{base_model_id}" --adapter_dir "workdir/adapter" --system "你是專業客服..." --user "我要查詢出貨進度"

注意事項：
- 確保推理時使用與訓練相同的 chat template（tokenizer.apply_chat_template）。
- 若顯存不足，可將 batch size 設為 1 且確保使用 4-bit 量化。
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(readme)

def main():
    adapter_dir = str(_WORKDIR / "adapter")
    tokenizer, model = load_model_for_inference(BASE_MODEL_ID, adapter_dir)

    # 讀取 test.jsonl
    test_examples = []
    with open(_WORKDIR / "test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            test_examples.append(json.loads(line))

    # TODO 5: 執行模板消融實驗
    run_template_ablation(test_examples, tokenizer, model, max_samples=5)

    # TODO 6: 寫出 inference.py 與 README_delivery.txt
    write_inference_script(BASE_MODEL_ID, adapter_dir, str(_WORKDIR / "inference.py"))
    write_readme(BASE_MODEL_ID, str(_WORKDIR / "README_delivery.txt"))

    print("已輸出 workdir/inference.py 與 workdir/README_delivery.txt")

if __name__ == "__main__":
    main()