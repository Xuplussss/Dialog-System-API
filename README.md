# Dialog-System-API

!["our proposed system frameworks"](https://github.com/Xuplussss/Dialog-System-API/blob/main/SystemFrameworks.png?raw=true)

## Requirements
```
- uvicorn
- fastapi
- transformers
```
## 簡介:
```
本專案用於建立一個供呼叫的對話系統API，可支援輸入為語音及文字輸入。
語音輸入先進行ASR轉換為文字，文字則直接進入回應生成模型進行系統回復。 
系統回覆後進行段句再return。
```
## 伺服器建立
需先準備對話系統模型於Dialog_model目錄以及ASR模型或其他ASR套件替換dialog_server_userID.py中的speech2text。

本腳本以DialoGPT為基底撰寫。
```
uvicorn dialog_server_userID:app --host localhost --port 8087
```
## 客戶端呼叫
### 文字輸入
```
http://localhost:8087/dialog_text/?user_id={使用者ID}&item_id={輸入的內容}
```
### 語音輸入
```
curl -i -X POST  http://localhost:8087/dialog_audio/{使用者ID}/ -F "file=@{語音檔名}.wav"

or

curl -i -H 'Content-type:audio/wav' -X POST  http://localhost:8087/dialog_audio/{使用者ID}/ -F "file=@{語音檔名}.wav"
```
## Reference
This package provides training code for the BD APP paper. If you use this codebase in your experiments please cite: 

```

```
