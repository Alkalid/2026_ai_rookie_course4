Lab2：Tokenizer 與成本估算
題目說明
目標：

## 比較不同模型（如 TinyLlama、Mistral、Qwen）對同一段繁體中文、英文、程式碼的 token 數差異。
實作一個簡單的「訓練預算估算器」，根據：
- 資料筆數
- 平均 prompt token 數
- 平均回覆 token 數
- 訓練 epoch 數

估計總訓練 token 與大致所需時間。