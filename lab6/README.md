Lab6：小型消融與交付封裝

## 題目說明
目標：

- 進行「模板一致性」消融實驗：
    - 正確使用 chat template vs 錯誤的 prompt（不含 system / 不用 chat template）
- 比較兩者在 Lab5 評估指標上的差異。
- （選作）對比 packing=True / False 的訓練 loss（可縮短 steps）。
- 將推理邏輯封裝成一個獨立腳本與 README，方便團隊直接使用。


完成後你應該能夠：
- 實際觀察「模板錯／對」的效果差異。
- 能把微調成果打包成可以交付給同事使用的簡單 CLI。