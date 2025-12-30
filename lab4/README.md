Lab4：指令微調（PEFT/QLoRA）
題目說明
目標：

使用 Lab3 產生的 train.jsonl / val.jsonl 作為指令微調資料。
- 使用 TRL 的 SFTTrainer + PEFT 的 LoRA / QLoRA 完成短程訓練：
- 啟用 4-bit 量化（QLoRA）
- 啟用 gradient checkpointing
- 設定 packing（序列打包）
- 產出一組 LoRA 權重（adapter），之後的推理會載入這些權重。


完成後你應該能夠：

跑通一次端對端的 SFT 訓練。
理解主要超參數的位置與用途（batch size、max_steps、learning_rate 等）。