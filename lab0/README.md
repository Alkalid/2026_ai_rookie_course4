# Lab0：環境檢查與第一次生成

## 目的

先確認電腦能從 Hugging Face 載入因果語言模型，用 **Chat Template** 組好提示詞並跑完一次 `generate`。環境沒問題，後面幾關才不會一直踩雷。

## 學習目標

1. 確認 **PyTorch** 版本與 **CUDA / GPU** 是否可用（若無 GPU，程式仍應能以 CPU 跑通，只是較慢）。
2. 成功用 `AutoTokenizer` / `AutoModelForCausalLM` 載入預設基礎模型（見 `lab0.py` 內 `BASE_MODEL_ID`，可用環境變數 `BASE_MODEL_ID` 覆寫）。
3. 將 `messages`（`system` + `user`）透過 `tokenizer.apply_chat_template(..., add_generation_prompt=True)` 轉成字串，再 tokenize 後呼叫 `model.generate`，並解碼得到模型回覆。

## 建議步驟（對照 `lab0.py`）

1. **建立環境與依賴**  
   在專案根目錄（有 `requirements.txt` 的那一層）建議先：
   - 建立虛擬環境並 `pip install -r requirements.txt`（若要用 4-bit / 後續 Lab，之後需再裝 `bitsandbytes`、`peft` 等；Lab0 僅需 `torch`、`transformers` 即可跑通基本流程）。
2. **執行腳本**  
   ```bash
   cd lab0
   python lab0.py
   ```
3. **自我檢核**
   - 終端機應印出：`PyTorch` 版本、`GPU 可用: True/False`（無 GPU 時為 `False` 屬正常）。
   - 應看到 `Chat template 文字:` 後面是一段依模型而定的對話格式字串。
   - 最後應有 `模型回應:` 與一段生成文字（內容每次可能不同，因有 `do_sample=True`）。

## 常見問題

| 現象 | 可能原因與處理 |
|------|----------------|
| 下載模型很慢或失敗 | 第一次會從 Hugging Face 拉權重；可設定鏡像、代理，或改用較小模型並設 `BASE_MODEL_ID`。 |
| `CUDA out of memory` | 換更小模型，或確認未同時開其他佔用顯存的程式；Lab0 的 `TinyLlama` 通常負擔較小。 |
| `apply_chat_template` 報錯 | 確認 tokenizer 來自 **Chat** 或 **Instruct** 類模型；純 base 模型可能沒有 chat template。 |

## 完成定義

- `python lab0.py` 可無錯誤跑完，且能看到一次完整的「模板 → generate → decode」輸出。
