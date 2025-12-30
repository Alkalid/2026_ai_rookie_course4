# lab2_chat_template.py
from typing import Any, Dict, List

from transformers import AutoTokenizer

from common_setup import BASE_MODEL_ID

# 範例原始資料（實務中可從檔案載入）
RAW_EXAMPLES = [
    {
        "id": "ex1",
        "messages": [
            {"role": "system", "content": "你是專業客服助理，請用繁體中文，語氣禮貌。"},
            {"role": "user", "content": "我想取消訂單，流程是什麼？"},
        ],
    },
    {
        "id": "ex2",
        "messages": [
            # 故意省略 system，測試自動補上
            {"role": "user", "content": "請問有沒有學生優惠？"},
        ],
    },
]

def ensure_system_message(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    確保 messages 的第一則是 system；若沒有就自動補上一則預設 system。
    """
    # TODO 1: 如果第一則不是 system，就插入一則預設 system
    # 預設內容可以是： "你是專業客服助理，請用繁體中文，語氣禮貌。"
    fixed = messages
    return fixed

def to_chat_template_text(example: Dict[str, Any], tokenizer) -> str:
    """
    將一筆 example（含 messages）轉換成 chat template 的文字結果。
    不加 generation prompt（訓練用）。
    """
    # TODO 2: 取得 messages 並呼叫 ensure_system_message
    messages = example["messages"]

    # TODO 3: 用 tokenizer.apply_chat_template 轉成文字
    #   - tokenize=False
    #   - add_generation_prompt=False
    chat_text = ""

    return chat_text

def check_template_consistency(chat_text: str, tokenizer) -> Dict[str, Any]:
    """
    對產生出的 chat_text 做一些簡單的啟發式檢查：
    - BOS/EOS 是否重複太多次
    - 是否可能缺少 system 導引
    """
    issues = []

    # TODO 4: 檢查 BOS/EOS 出現次數（若 > 1，可能有問題）
    bos = tokenizer.bos_token or ""
    eos = tokenizer.eos_token or ""

    # 例：if bos and chat_text.count(bos) > 1: issues.append("BOS 出現次數異常")

    # TODO 5: 檢查是否有明顯的「請用繁體」或「你是」等 system 導引（簡單字串檢查）
    # 例：if "請用繁體" not in chat_text: issues.append("可能缺少繁體中文的 system 導引")

    return {
        "issues": issues,
        "length": len(chat_text),
    }

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)

    for ex in RAW_EXAMPLES:
        chat_text = to_chat_template_text(ex, tokenizer)
        report = check_template_consistency(chat_text, tokenizer)
        print(f"ID: {ex['id']}, 長度={report['length']}, 問題={report['issues']}")
        # 可以視需要印出 chat_text 片段
        # print(chat_text)

if __name__ == "__main__":
    main()