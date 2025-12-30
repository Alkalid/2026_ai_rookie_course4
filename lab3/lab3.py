# lab3_data_cleaning.py
import json
import random
import re
import unicodedata
import hashlib
from typing import Any, Dict, List

from transformers import AutoTokenizer
from datasets import DatasetDict, load_dataset

from common_setup import BASE_MODEL_ID

# TODO 0: 嘗試匯入 opencc（如無則 fallback）
try:
    from opencc import OpenCC
    _cc_s2t = OpenCC("s2t")
    def to_trad(s: str) -> str:
        return _cc_s2t.convert(s)
except Exception:
    def to_trad(s: str) -> str:
        return s

BADWORDS = {"垃圾", "白癡", "去死"}  # 示意

def normalize_text(s: str) -> str:
    # TODO 1: 實作基礎清洗流程
    # 1) 簡轉繁
    # 2) Unicode 正規化
    # 3) 去掉多餘空白
    return s

def is_toxic(s: str) -> bool:
    # TODO 2: 如字串含 BADWORDS 中的任何詞就視為 toxic
    return False

def build_synthetic_examples(n: int = 60) -> List[Dict[str, Any]]:
    """
    建立一個模擬的客服訓練資料集，包含單輪與多輪，
    每筆資料格式：
    {
      "id": "ex0001",
      "messages": [ {role, content}, ... ],
      "topic": "退貨流程",
      "language": "zh-Hant"
    }
    """
    topics = [
        ("查詢出貨", "請幫我查詢訂單出貨進度，訂單號碼是 ABC123。"),
        ("退貨流程", "我想辦退貨，該怎麼做？"),
        ("退款時程", "退款通常需要幾天會到帳？"),
        ("修改地址", "下單後可以修改收件地址嗎？"),
        ("維修保固", "產品有保固嗎？如何送修？"),
        ("發票補發", "可以補發發票嗎？流程是？"),
    ]

    def synth_assistant(topic: str, user: str) -> str:
        # TODO 3: 回傳一個固定但看起來合理的客服回答（繁體中文）
        return "..."

    data = []
    for i in range(n):
        topic, user = random.choice(topics)
        # 隨機決定單輪或多輪
        if random.random() < 0.4:
            # 多輪
            messages = [
                # TODO 4: system + user + assistant + user + assistant
            ]
        else:
            # 單輪
            messages = [
                # TODO 5: system + user + assistant
            ]

        data.append(
            {
                "id": f"ex{i:04d}",
                "messages": messages,
                "topic": topic,
                "language": "zh-Hant",
            }
        )
    return data

def clean_dataset(
    examples: List[Dict[str, Any]],
    tokenizer,
    max_user_len: int = 512,
    max_total_tokens: int = 2048,
) -> List[Dict[str, Any]]:
    """
    - 對 messages 的 content 做 normalize / 繁簡轉換 / 毒性過濾
    - 過短/過長樣本刪除
    - 以 user 內容的 hash 去重
    - 用 chat template 估計 token 長度，超過 max_total_tokens 的刪除
    """
    cleaned = []
    seen_keys = set()

    for ex in examples:
        # TODO 6: 遍歷 messages，做 normalize_text & is_toxic 檢查
        # 若發現毒性，整筆丟棄
        messages = []

        # TODO 7: 計算所有 user content 的總長度，過短/過長丟棄
        # user_concat = ...

        # TODO 8: 用 chat_template 估 token 長度，超過 max_total_tokens 丟棄
        # chat_tokens = ...

        # TODO 9: 用 user_concat 做 hash，做去重
        # key = hashlib.md5(user_concat.encode("utf-8")).hexdigest()
        # 若 key 已存在於 seen_keys 則跳過

        # 否則加入 cleaned
        cleaned.append(ex)

    return cleaned

def save_jsonl(path: str, items: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)

    raw_examples = build_synthetic_examples(n=80)
    cleaned = clean_dataset(raw_examples, tokenizer, max_user_len=300, max_total_tokens=1024)
    random.shuffle(cleaned)

    n = len(cleaned)
    train = cleaned[: int(0.75 * n)]
    val = cleaned[int(0.75 * n) : int(0.9 * n)]
    test = cleaned[int(0.9 * n) :]

    save_jsonl("workdir/train.jsonl", train)
    save_jsonl("workdir/val.jsonl", val)
    save_jsonl("workdir/test.jsonl", test)

    print(f"資料筆數: train={len(train)}, val={len(val)}, test={len(test)}")

if __name__ == "__main__":
    main()