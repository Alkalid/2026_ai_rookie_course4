# Lab1：Chat Template 轉換與一致性檢查

## 目的

微調跟推理時，模型吃的通常不是純 user 字串，而是經過 **chat template** 排好的字串（含 special tokens、角色標記）。這關要你：

- 讀 **JSON**（每筆有 `id`、`messages`，message 有 `role`、`content`）。
- 用 `AutoTokenizer.apply_chat_template` 轉成訓練／前處理用的文字。
- 寫簡單 **一致性檢查**，避免訓練、推理格式不一致，或 BOS/EOS 怪怪的、沒 system 導引等。

> **檔名對照**：`lab1.py` 檔頭註解寫 `lab2_chat_template.py`，內容就是 **Chat Template**；照這份說明和 `lab1.py` 裡的 `TODO` 做即可。

## 學習目標

- 能正確區分：`add_generation_prompt=False`（訓練／資料前處理）與 `True`（要接 `model.generate` 時）的差異。
- 能補齊或保留 **system** 角色，使資料格式穩定。
- 能對產生的字串做啟發式檢查，列出 `issues` 供除錯。

## 建議步驟（對照 `lab1.py` 的 TODO）

1. **環境**  
   在專案根目錄已執行過 `uv sync`。需能連線下載 tokenizer：`BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"`（僅載 tokenizer 即可，不必載完整 3B 權重做推理）。
2. **TODO 1 — `ensure_system_message`**  
   - 若 `messages` 第一則的 `role` 不是 `system`，請在**最前面**插入一則預設 system，例如：`"你是專業客服助理，請用繁體中文，語氣禮貌。"`  
   - 若已有 system，則不要重複插入。
3. **TODO 2～3 — `to_chat_template_text`**  
   - 從 `example["messages"]` 取出列表，先經 `ensure_system_message` 處理。  
   - 呼叫 `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)`，回傳字串。
4. **TODO 4～5 — `check_template_consistency`**  
   - 用 `tokenizer.bos_token`、`eos_token`（可能為 `None`）檢查在 `chat_text` 中是否出現**過多次**，若異常則加入 `issues`。  
   - 用簡單字串規則檢查是否**可能缺少**繁體／角色導引（例如是否包含「請用繁體」「你是」等，可依你們實際預設 system 調整關鍵字）。  
   - 回傳 `{"issues": [...], "length": len(chat_text)}`。

## 執行與自我檢核

```bash
cd lab1
uv run python lab1.py
```

- 對 `RAW_EXAMPLES` 裡的 `ex1`、`ex2` 都應印出長度與 `issues`。  
- `ex2` 刻意沒有 system：實作正確時，模板結果應仍帶有你補上的 system 導引；`issues` 中「缺少導引」類訊息應合理或為空（視你的規則而定）。

## 與後續 Lab 的關係

- **Lab3** 的 `messages`、**Lab4** 的 `formatting_samples`、**Lab5** 推理時的 `apply_chat_template`，最好跟這裡講的格式一致（同一模型族，或刻意對齊 template）。

## 完成定義

- `ensure_system_message`、`to_chat_template_text`、`check_template_consistency` 實作完成，`uv run python lab1.py` 可跑完且輸出合理。

## `apply_chat_template` 簡短示範

把 `messages`（每則含 `role`、`content`）轉成模型要吃的字串。下面範例跟 `lab1.py` 一樣用 `Qwen/Qwen2.5-3B-Instruct`：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", use_fast=True)
messages = [
    {"role": "system", "content": "你是專業客服助理，請用繁體中文。"},
    {"role": "user", "content": "我想取消訂單。"},
]

# 訓練／前處理：回傳字串，不要加「下一輪 assistant」開頭
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)

# 若要接 model.generate，把上面改成 add_generation_prompt=True
```

`tokenize=False` 表示回傳文字；若要直接拿 tensor，可改 `tokenize=True` 並加上 `return_tensors="pt"`。
