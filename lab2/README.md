Lab2：Chat Template 轉換與一致性檢查
題目說明
目標：

- 讀入「原始對話 JSON 結構」（messages: role + content）。
- 用 AutoTokenizer.apply_chat_template 轉成模型所需的文字格式。
- 實作簡單的檢查函式，偵測常見錯誤：
    - 遺漏 system 訊息
    - 重複加入 BOS/EOS
    - 長度是否異常等

完成後你應該能夠：
- 正確用 apply_chat_template 把資料變成模型訓練/推理的輸入。
- 有一個「模板一致性檢查」函式，避免訓練與推理時用到不同結構。