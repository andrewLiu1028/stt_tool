# 語音轉文字應用

這是一個使用 Streamlit 和 Whisper 開發的語音轉文字應用，支援多種音頻格式的轉換。

## 功能特點

- 支援 MP3、WAV、M4A、FLAC 格式
- 提供純文本和 SRT 字幕輸出
- 多種模型選擇，滿足不同需求
- 自動適應運行環境（本地/雲端）

## 部署說明

1. 安裝系統依賴：
```bash
apt-get update && apt-get install ffmpeg
```

2. 安裝 Python 依賴：
```bash
pip install -r requirements.txt
```

3. 運行應用：
```bash
streamlit run app.py
```

## 注意事項

- 本地運行支援 GPU 加速
- Streamlit Cloud 環境使用 CPU 模式
- 檔案大小限制：雲端 50MB / 本地 100MB