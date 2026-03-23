# lab2_chat_template.py
from typing import Any, Dict, List

from transformers import AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

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

def check_template_consistency(
    chat_text: str, messages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    對即將／已經餵給 apply_chat_template 的 messages 做簡單一致性檢查（建議傳入與
    apply_chat_template 相同的列表，例如已經過 ensure_system_message）：
    - 每則是否資料完整（必要欄位是否存在）
    - content 是否為空
    - role 順序是否合理（第一則為 system，之後 user／assistant 交替）
    """
    issues: List[str] = []

    # TODO 4: 資料完整性與 content 是否為空
    #   - 每則皆為 dict，且含 role、content
    #   - role、content 經 strip 後不應為空字串

    # TODO 5: role 順序
    #   - 第一則 role 必須為 system
    #   - 之後索引 1,3,5,... 應為 user；2,4,6,... 應為 assistant

    return {
        "issues": issues,
        "length": len(chat_text),
    }

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)

    for ex in RAW_EXAMPLES:
        messages_for_check = ensure_system_message(list(ex["messages"]))
        chat_text = to_chat_template_text(ex, tokenizer)
        report = check_template_consistency(chat_text, messages_for_check)
        print(f"ID: {ex['id']}, 長度={report['length']}, 問題={report['issues']}")
        # 可以視需要印出 chat_text 片段
        # print(chat_text)

if __name__ == "__main__":
    main()