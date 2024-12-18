# import streamlit as st
# import whisper
# import torch
# import warnings
# from pathlib import Path
# import tempfile
# import logging
# import os
# from typing import Optional, Tuple
# import time

# # 設置環境變數和警告控制
# os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
# os.environ["TORCH_ENABLE_CPU_FALLBACK"] = "1"
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
# logging.getLogger("whisper").setLevel(logging.ERROR)

# class AudioConverter:
#     SUPPORTED_MODELS = {
#         "tiny": "最小模型 - 最快但準確度較低",
#         "base": "基礎模型 - 平衡速度和準確度",
#         "small": "小型模型 - 較好的準確度",
#         "medium": "中型模型 - 高準確度",
#         "large": "大型模型 - 最高準確度但較慢"
#     }
    
#     def __init__(self):
#         self.model = None
#         self.model_name = None
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.cache_dir = Path.home() / ".cache" / "whisper"
#         self.cache_dir.mkdir(parents=True, exist_ok=True)
    
#     def load_model(self, model_name: str) -> bool:
#         """載入模型，如果已載入相同模型則跳過"""
#         try:
#             if self.model is None or self.model_name != model_name:
#                 with st.spinner(f"載入 {model_name} 模型中..."):
#                     self.model = whisper.load_model(
#                         model_name,
#                         device=self.device,
#                         download_root=str(self.cache_dir)
#                     )
#                     self.model_name = model_name
#                 st.success(f"模型 {model_name} 載入成功！")
#             return True
#         except Exception as e:
#             st.error(f"模型載入失敗: {str(e)}")
#             return False

#     def validate_audio_file(self, file) -> bool:
#         """驗證音頻檔案"""
#         if file is None:
#             st.error("請上傳音頻檔案")
#             return False
        
#         max_size = 100 * 1024 * 1024  # 100MB
#         if file.size > max_size:
#             st.error("檔案大小不能超過 100MB")
#             return False
            
#         return True

#     def process_audio(self, audio_file, progress_bar) -> Tuple[Optional[str], Optional[str]]:
#         """處理音頻檔案並返回轉錄結果"""
#         if not self.validate_audio_file(audio_file):
#             return None, None

#         try:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp_file:
#                 tmp_path = Path(tmp_file.name)
#                 # 寫入臨時檔案
#                 progress_bar.progress(0.2, "準備音頻檔案...")
#                 tmp_file.write(audio_file.getvalue())

#             try:
#                 # 轉錄處理
#                 progress_bar.progress(0.4, "開始轉錄...")
#                 result = self.model.transcribe(
#                     str(tmp_path),
#                     language="zh",
#                     fp16=False,
#                     initial_prompt="以繁體中文轉錄內容"
#                 )
                
#                 progress_bar.progress(0.7, "生成輸出...")
                
#                 # 生成文本輸出
#                 plain_text = self._generate_plain_text(result["segments"])
                
#                 # 生成 SRT 輸出
#                 srt_content = self._generate_srt(result["segments"])
                
#                 progress_bar.progress(1.0, "處理完成！")
#                 return plain_text, srt_content
                
#             finally:
#                 # 清理臨時檔案
#                 if tmp_path.exists():
#                     tmp_path.unlink()
                    
#         except Exception as e:
#             st.error(f"處理失敗: {str(e)}")
#             return None, None

#     def _generate_plain_text(self, segments) -> str:
#         """生成純文本輸出"""
#         return "\n".join([
#             segment['text'].strip() 
#             for segment in segments
#         ])

#     def _generate_srt(self, segments) -> str:
#         """生成 SRT 格式輸出"""
#         srt_content = []
#         for i, segment in enumerate(segments, 1):
#             start_time = self._format_timestamp(segment['start'])
#             end_time = self._format_timestamp(segment['end'])
#             srt_content.append(
#                 f"{i}\n{start_time} --> {end_time}\n{segment['text'].strip()}\n"
#             )
#         return "\n".join(srt_content)

#     def _format_timestamp(self, seconds: float) -> str:
#         """格式化時間戳"""
#         hours = int(seconds // 3600)
#         minutes = int((seconds % 3600) // 60)
#         seconds = int(seconds % 60)
#         milliseconds = int((seconds % 1) * 1000)
#         return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

# def main():
#     st.set_page_config(
#         page_title="音頻轉文字平台",
#         page_icon="🎵",
#         layout="wide"
#     )
    
#     st.title("🎵 音頻轉文字平台")
#     st.caption("支援 MP3、WAV、M4A、FLAC 格式音頻檔案轉換為文字和字幕")
    
#     # 初始化 session state
#     if 'converter' not in st.session_state:
#         st.session_state.converter = AudioConverter()
    
#     # 側邊欄設置
#     with st.sidebar:
#         st.subheader("⚙️ 設置")
#         selected_model = st.selectbox(
#             "選擇模型",
#             options=list(AudioConverter.SUPPORTED_MODELS.keys()),
#             format_func=lambda x: f"{x} - {AudioConverter.SUPPORTED_MODELS[x]}",
#             index=1  # 預設選擇 base 模型
#         )
#         st.info(f"目前使用的設備: {st.session_state.converter.device}")
    
#     # 檔案上傳區域
#     audio_file = st.file_uploader(
#         "上傳音頻檔案",
#         type=["mp3", "wav", "m4a", "flac"],
#         help="支援的格式：MP3、WAV、M4A、FLAC，檔案大小限制：100MB"
#     )
    
#     if audio_file:
#         st.audio(audio_file)
        
#         col1, col2 = st.columns([2, 1])
#         with col1:
#             process_button = st.button(
#                 "開始處理",
#                 type="primary",
#                 use_container_width=True
#             )
            
#         if process_button:
#             # 載入選擇的模型
#             if st.session_state.converter.load_model(selected_model):
#                 # 顯示進度條
#                 progress_bar = st.progress(0, "準備處理...")
                
#                 # 處理音頻
#                 plain_text, srt_content = st.session_state.converter.process_audio(
#                     audio_file, progress_bar
#                 )
                
#                 if plain_text and srt_content:
#                     st.success("處理完成！")
                    
#                     # 顯示結果
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         st.subheader("📝 逐字稿")
#                         st.text_area(
#                             "純文本輸出",
#                             plain_text,
#                             height=300
#                         )
#                         st.download_button(
#                             "下載逐字稿 (TXT)",
#                             plain_text,
#                             f"{Path(audio_file.name).stem}_transcript.txt",
#                             mime="text/plain"
#                         )
                            
#                     with col2:
#                         st.subheader("🎬 字幕檔")
#                         st.text_area(
#                             "SRT 格式輸出",
#                             srt_content,
#                             height=300
#                         )
#                         st.download_button(
#                             "下載字幕檔 (SRT)",
#                             srt_content,
#                             f"{Path(audio_file.name).stem}_subtitle.srt",
#                             mime="text/plain"
#                         )
#     else:
#         st.info("👆 請上傳音頻檔案開始處理")

# if __name__ == "__main__":
#     main()


import streamlit as st
import whisper
import torch
import warnings
from pathlib import Path
import tempfile
import logging
import os
from typing import Optional, Tuple

# 設置環境變數和警告控制
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
os.environ["TORCH_ENABLE_CPU_FALLBACK"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("whisper").setLevel(logging.ERROR)

class AudioConverter:
    # 根據執行環境決定可用的模型
    SUPPORTED_MODELS = {
        "tiny": "最小模型 - 最快但準確度較低 (150MB)",
        "base": "基礎模型 - 平衡速度和準確度 (290MB)",
        "small": "小型模型 - 較好的準確度 (960MB)",
    }
    
    # 本地環境額外提供的模型
    LOCAL_MODELS = {
        "medium": "中型模型 - 高準確度 (1.5GB)",
        "large": "大型模型 - 最高準確度但較慢 (2.9GB)",
    }
    
    def __init__(self):
        """初始化轉換器"""
        self.model = None
        self.model_name = None
        # 檢查是否在 Streamlit Cloud 環境
        self.is_cloud = self._is_streamlit_cloud()
        # 根據環境選擇設備
        self.device = self._get_device()
        # 根據環境選擇可用模型
        self.available_models = self._get_available_models()
        
    def _is_streamlit_cloud(self) -> bool:
        """判斷是否在 Streamlit Cloud 環境"""
        return os.environ.get('STREAMLIT_CLOUD') == 'true'
    
    def _get_device(self) -> str:
        """根據環境選擇運算設備"""
        if self.is_cloud:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def _get_available_models(self) -> dict:
        """根據環境返回可用的模型列表"""
        if self.is_cloud:
            return self.SUPPORTED_MODELS
        return {**self.SUPPORTED_MODELS, **self.LOCAL_MODELS}
    
    @st.cache_resource(show_spinner=False)
    def load_model(_self, model_name: str):
        """
        使用 st.cache_resource 來快取模型
        這樣在 Streamlit 重新運行時不會重複載入模型
        """
        return whisper.load_model(model_name, device=_self.device)

    def get_model(self, model_name: str) -> bool:
        """獲取模型，如果已快取則直接返回"""
        try:
            if self.model_name != model_name:
                with st.spinner(f"載入 {model_name} 模型中..."):
                    self.model = self.load_model(model_name)
                    self.model_name = model_name
                st.success(f"模型 {model_name} 載入成功！")
            return True
        except Exception as e:
            st.error(f"模型載入失敗: {str(e)}")
            return False

    def validate_audio_file(self, file) -> bool:
        """驗證音頻檔案"""
        if file is None:
            st.error("請上傳音頻檔案")
            return False
        
        # 根據環境設置大小限制
        max_size = 50 * 1024 * 1024 if self.is_cloud else 100 * 1024 * 1024
        if file.size > max_size:
            st.error(f"檔案大小不能超過 {'50MB' if self.is_cloud else '100MB'}")
            return False
            
        return True

    def process_audio(self, audio_file, progress_bar) -> Tuple[Optional[str], Optional[str]]:
        """處理音頻檔案並返回轉錄結果"""
        if not self.validate_audio_file(audio_file):
            return None, None

        temp_dir = None
        temp_path = None
        
        try:
            # 創建臨時目錄
            temp_dir = tempfile.mkdtemp()
            # 在臨時目錄中創建檔案
            temp_path = Path(temp_dir) / f"audio{Path(audio_file.name).suffix}"
            
            progress_bar.progress(0.2, "準備音頻檔案...")
            # 寫入音頻數據
            with open(temp_path, 'wb') as f:
                f.write(audio_file.getvalue())
            
            # 確保檔案寫入完成
            progress_bar.progress(0.4, "開始轉錄...")
            
            # 轉錄處理
            result = self.model.transcribe(
                str(temp_path),
                language="zh",
                fp16=torch.cuda.is_available() and not self.is_cloud,  # 在本地 GPU 環境啟用 fp16
                initial_prompt="以繁體中文轉錄內容"
            )
            
            progress_bar.progress(0.7, "生成輸出...")
            
            # 生成輸出
            plain_text = self._generate_plain_text(result["segments"])
            srt_content = self._generate_srt(result["segments"])
            
            progress_bar.progress(1.0, "處理完成！")
            return plain_text, srt_content
                    
        except Exception as e:
            st.error(f"處理失敗: {str(e)}")
            return None, None
            
        finally:
            # 清理臨時檔案
            try:
                if temp_path and temp_path.exists():
                    temp_path.unlink()
                if temp_dir and Path(temp_dir).exists():
                    Path(temp_dir).rmdir()
            except Exception as e:
                print(f"清理臨時檔案時發生錯誤: {e}")

    def _generate_plain_text(self, segments) -> str:
        """生成純文本輸出"""
        return "\n".join([
            segment['text'].strip() 
            for segment in segments
        ])

    def _generate_srt(self, segments) -> str:
        """生成 SRT 格式輸出"""
        srt_content = []
        for i, segment in enumerate(segments, 1):
            start_time = self._format_timestamp(segment['start'])
            end_time = self._format_timestamp(segment['end'])
            srt_content.append(
                f"{i}\n{start_time} --> {end_time}\n{segment['text'].strip()}\n"
            )
        return "\n".join(srt_content)

    def _format_timestamp(self, seconds: float) -> str:
        """格式化時間戳"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def main():
    st.set_page_config(
        page_title="語音轉文字平台",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 語音轉文字")
    st.caption("支援 MP3、WAV、M4A、FLAC 格式音頻檔案轉換為文字和字幕")
    
    # 初始化 session state
    if 'converter' not in st.session_state:
        st.session_state.converter = AudioConverter()
    
    # 側邊欄設置
    with st.sidebar:
        st.subheader("⚙️ 設置")
        
        # 顯示運行環境資訊
        env_info = "🖥️ 本地環境" if not st.session_state.converter.is_cloud else "☁️ 雲端環境"
        device_info = f"使用 {'GPU' if st.session_state.converter.device == 'cuda' else 'CPU'} 運算"
        st.info(f"{env_info} - {device_info}")
        
        selected_model = st.selectbox(
            "選擇模型",
            options=list(st.session_state.converter.available_models.keys()),
            format_func=lambda x: f"{x} - {st.session_state.converter.available_models[x]}",
            index=1  # 預設選擇 base 模型
        )
        
        st.markdown(f"""
        ### 📝 使用說明
        1. 選擇合適的模型
        2. 上傳音頻檔案（限制 {'50MB' if st.session_state.converter.is_cloud else '100MB'} 以下）
        3. 點擊處理按鈕開始轉換
        4. 下載轉換結果
        
        ### ℹ️ 模型說明
        - tiny: 適合簡短、清晰的音頻
        - base: 一般用途，適合大多數場景
        - small: 較高準確度，但處理較慢
        {'- medium: 高準確度，適合較長音頻' if not st.session_state.converter.is_cloud else ''}
        {'- large: 最高準確度，適合複雜音頻' if not st.session_state.converter.is_cloud else ''}
        """)
    
    # 檔案上傳區域
    audio_file = st.file_uploader(
        "上傳音頻檔案",
        type=["mp3", "wav", "m4a", "flac"],
        help=f"支援的格式：MP3、WAV、M4A、FLAC，檔案大小限制：{'50MB' if st.session_state.converter.is_cloud else '100MB'}"
    )
    
    if audio_file:
        st.audio(audio_file)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            process_button = st.button(
                "開始處理",
                type="primary",
                use_container_width=True
            )
            
        if process_button:
            # 載入選擇的模型
            if st.session_state.converter.get_model(selected_model):
                # 顯示進度條
                progress_bar = st.progress(0, "準備處理...")
                
                # 處理音頻
                plain_text, srt_content = st.session_state.converter.process_audio(
                    audio_file, progress_bar
                )
                
                if plain_text and srt_content:
                    st.success("處理完成！")
                    
                    # 顯示結果
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📝 逐字稿")
                        st.text_area(
                            "純文本輸出",
                            plain_text,
                            height=300
                        )
                        st.download_button(
                            "下載逐字稿 (TXT)",
                            plain_text,
                            f"{Path(audio_file.name).stem}_transcript.txt",
                            mime="text/plain"
                        )
                            
                    with col2:
                        st.subheader("🎬 字幕檔")
                        st.text_area(
                            "SRT 格式輸出",
                            srt_content,
                            height=300
                        )
                        st.download_button(
                            "下載字幕檔 (SRT)",
                            srt_content,
                            f"{Path(audio_file.name).stem}_subtitle.srt",
                            mime="text/plain"
                        )
    else:
        st.info("👆 請上傳音頻檔案開始處理")

if __name__ == "__main__":
    main()