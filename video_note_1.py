import gradio as gr
import cv2
import numpy as np
# from moviepy.editor import VideoFileClip
from moviepy import *
import whisper
import torch
import json
import os
from datetime import timedelta
import tempfile
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoTokenizer
import re
from typing import List, Tuple, Dict
import gc
from pydub import AudioSegment
import logging
import ollama  # 添加这一行

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.video_path = None
        self.roi = None
        self.whisper_model = None
        self.clip_model = None
        self.clip_processor = None
        self.gemma_model = None
        self.gemma_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    def extract_audio(self, video_path):
        """从视频中提取音频"""
        try:
            video = VideoFileClip(video_path)
            audio = video.audio
            
            # 保存音频文件
            audio_path = tempfile.mktemp(suffix=".wav")
            audio.write_audiofile(audio_path)
            
            video.close()
            return audio_path
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise
    
    def split_audio_chunks(self, audio_path, chunk_duration=3600, overlap=60):
        """将音频分割成1小时的块，带重叠"""
        try:
            audio = AudioSegment.from_wav(audio_path)
            chunks = []
            
            chunk_length = chunk_duration * 1000  # 转换为毫秒
            overlap_length = overlap * 1000
            
            start = 0
            chunk_num = 0
            
            while start < len(audio):
                end = min(start + chunk_length, len(audio))
                chunk = audio[start:end]
                
                chunk_path = tempfile.mktemp(suffix=f"_chunk_{chunk_num}.wav")
                chunk.export(chunk_path, format="wav")
                chunks.append((chunk_path, start/1000))  # 返回路径和开始时间（秒）
                
                start += chunk_length - overlap_length
                chunk_num += 1
                
            return chunks
        except Exception as e:
            logger.error(f"Error splitting audio: {e}")
            raise
    
    def transcribe_audio(self, audio_chunks):
        """使用Whisper转录音频"""
        try:
            if self.whisper_model is None:
                logger.info("Loading Whisper model...")
                self.whisper_model = whisper.load_model("base")
            
            all_segments = []
            
            for chunk_path, start_time in audio_chunks:
                logger.info(f"Transcribing chunk starting at {start_time}s...")
                result = self.whisper_model.transcribe(chunk_path)
                
                # 调整时间戳
                for segment in result["segments"]:
                    segment["start"] += start_time
                    segment["end"] += start_time
                    all_segments.append(segment)
            
            # 保存到JSON
            json_path = tempfile.mktemp(suffix=".json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_segments, f, ensure_ascii=False, indent=2)
            
            # 卸载模型以释放内存
            logger.info("Offloading Whisper model...")
            del self.whisper_model
            self.whisper_model = None
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return json_path, all_segments
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise
    
    def get_frame_at_time(self, video_path, time_seconds):
        """获取指定时间的视频帧"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(fps * time_seconds)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def extract_frames_with_roi(self, video_path, interval_seconds, roi):
        """根据ROI提取视频帧"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        frames = []
        frame_numbers = []
        
        logger.info(f"Extracting frames every {interval_seconds}s from {duration:.1f}s video...")
        
        for i in range(0, int(duration), interval_seconds):
            frame_num = int(i * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if roi:
                    x1, y1, x2, y2 = roi
                    frame = frame[y1:y2, x1:x2]
                frames.append(frame)
                frame_numbers.append(frame_num)
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames")
        return frames, frame_numbers
    
    def remove_duplicate_frames(self, frames, frame_numbers, threshold=0.95):
        """使用CLIP移除重复帧"""
        try:
            if self.clip_model is None:
                logger.info("Loading CLIP model from Hugging Face...")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            
            unique_frames = []
            unique_numbers = []
            
            if len(frames) == 0:
                return unique_frames, unique_numbers
            
            # 转换所有帧为PIL图像
            pil_images = [Image.fromarray(frame) for frame in frames]
            
            # 批量处理图像获取特征
            logger.info("Computing CLIP features for all frames...")
            all_features = []
            batch_size = 8  # 根据GPU内存调整
            
            for i in range(0, len(pil_images), batch_size):
                batch = pil_images[i:i+batch_size]
                inputs = self.clip_processor(images=batch, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    all_features.append(image_features)
            
            all_features = torch.cat(all_features, dim=0)
            
            # 比较相似度并去重
            unique_indices = [0]  # 第一帧总是保留
            
            for i in range(1, len(frames)):
                is_duplicate = False
                current_feature = all_features[i:i+1]
                
                for j in unique_indices:
                    similarity = torch.cosine_similarity(current_feature, all_features[j:j+1])
                    if similarity.item() > threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_indices.append(i)
            
            # 收集唯一帧
            unique_frames = [frames[i] for i in unique_indices]
            unique_numbers = [frame_numbers[i] for i in unique_indices]
            
            logger.info(f"Removed {len(frames) - len(unique_frames)} duplicate frames")
            return unique_frames, unique_numbers
            
        except Exception as e:
            logger.error(f"Error removing duplicates: {e}")
            raise
    
    def extract_ppt_content(self, frame, frame_number, fps):
        """使用Ollama和Gemma提取PPT内容（简化版本）"""
        try:
            # 保存帧到临时文件
            temp_frame_path = tempfile.mktemp(suffix=".png")
            pil_frame = Image.fromarray(frame)
            pil_frame.save(temp_frame_path)
            
            # 构建提示词，要求模型分析图像中的PPT内容
            time_seconds = frame_number / fps
            time_str = str(timedelta(seconds=int(time_seconds)))
            
            prompt = f"""OCR the provided image and extract any presentation (PPT) content you can identify. \n\nPlease structure your response in markdown format"""
            # :\n1. A timestamp header indicating when in the video this frame appears (Time: {time_str})\n2. A 'Main Content' section describing the key points\n3. Any 'Text Detected' in the image\n4. Any 'Formulas' written in LaTeX format\n5. A 'Key Points' section as a bullet list of important concepts\n\nIf you cannot identify any presentation content, please respond with \"No presentation content identified at time {time_str}.\"
            # 使用Ollama调用Gemma模型分析图像
            # 使用gemma2:27b模型，它是Gemma的27B版本，功能更强大
            response = ollama.generate(
                model='gemma3n:e4b',
                prompt=prompt,
                images=[temp_frame_path],
                options={
                    'temperature': 0.1,  # 低温度值确保更一致的输出
                    'num_ctx': 2048      # 增加上下文长度以处理详细内容
                }
            )
            
            # 提取生成的内容
            content = response['response'].strip()
            
            # 如果没有返回内容，提供默认格式
            if not content:
                content = f"### Time: {time_str}\n\nNo presentation content identified.\n\n"
            
            # 清理临时文件
            os.remove(temp_frame_path)
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting PPT content: {e}")
            time_seconds = frame_number / fps
            time_str = str(timedelta(seconds=int(time_seconds)))
            return f"### Time: {time_str}\n\nError extracting content: {str(e)}\n\n"

# Gradio界面
class GradioInterface:
    def __init__(self):
        self.processor = VideoProcessor()
        self.preview_frame = None
        self.clicks = []
        self.video_path = None
        
    def upload_video(self, video):
        """处理视频上传"""
        if video is None:
            return None, None, None
        
        self.video_path = video
        
        # 获取第2分钟的预览帧
        frame = self.processor.get_frame_at_time(video, 120)  # 2分钟 = 120秒
        
        if frame is not None:
            self.preview_frame = frame
            self.clicks = []  # 重置点击
            return frame, "视频上传成功！请在预览帧上点击两次来选择感兴趣区域。", None
        else:
            # 如果视频不足2分钟，获取中间帧
            cap = cv2.VideoCapture(video)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            cap.release()
            
            middle_time = duration / 2
            frame = self.processor.get_frame_at_time(video, middle_time)
            if frame is not None:
                self.preview_frame = frame
                self.clicks = []
                return frame, f"视频长度不足2分钟，显示第{middle_time:.1f}秒的帧。请点击两次选择区域。", None
            
            return None, "无法读取视频帧", None
    
    def handle_click(self, evt: gr.SelectData):
        """处理鼠标点击事件"""
        if self.preview_frame is None:
            return self.preview_frame, "请先上传视频", None
        
        x, y = evt.index[0], evt.index[1]
        self.clicks.append((x, y))
        
        # 在图像上绘制点击位置
        img = self.preview_frame.copy()
        for click in self.clicks:
            cv2.circle(img, click, 5, (255, 0, 0), -1)
        
        if len(self.clicks) == 2:
            # 绘制矩形
            x1, y1 = self.clicks[0]
            x2, y2 = self.clicks[1]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 保存ROI
            self.processor.roi = (
                min(x1, x2), min(y1, y2),
                max(x1, x2), max(y1, y2)
            )
            
            status = f"已选择区域: ({min(x1,x2)}, {min(y1,y2)}) 到 ({max(x1,x2)}, {max(y1,y2)})"
            self.clicks = []  # 重置点击
        else:
            status = f"已点击第{len(self.clicks)}个点，请再点击一次完成选择"
        
        return img, status, None
    
    def extract_audio_only(self, video):
        """仅提取音频"""
        if video is None:
            return None, "请先上传视频"
        
        try:
            audio_path = self.processor.extract_audio(video)
            return audio_path, "音频提取成功！"
        except Exception as e:
            return None, f"音频提取失败: {str(e)}"
    
    def process_video_full(self, video, interval, threshold, progress=gr.Progress()):
        """完整处理视频"""
        if video is None:
            return None, None, None, "请先上传视频"
        
        try:
            progress(0.1, desc="提取音频...")
            audio_path = self.processor.extract_audio(video)
            
            progress(0.2, desc="分割音频...")
            audio_chunks = self.processor.split_audio_chunks(audio_path)
            
            progress(0.3, desc="转录音频（这可能需要较长时间）...")
            transcript_json, segments = self.processor.transcribe_audio(audio_chunks)
            
            progress(0.5, desc="提取视频帧...")
            frames, frame_numbers = self.processor.extract_frames_with_roi(
                video, interval, self.processor.roi
            )
            
            progress(0.7, desc="去除重复帧...")
            unique_frames, unique_numbers = self.processor.remove_duplicate_frames(
                frames, frame_numbers, threshold
            )
            
            progress(0.8, desc="提取PPT内容...")
            
            # 获取视频信息
            cap = cv2.VideoCapture(video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # 创建markdown内容
            markdown_content = "# 视频PPT内容提取报告\n\n"
            markdown_content += f"- 总帧数: {len(frames)}\n"
            markdown_content += f"- 唯一帧数: {len(unique_frames)}\n"
            markdown_content += f"- 音频段数: {len(audio_chunks)}\n"
            markdown_content += f"- 转录段数: {len(segments)}\n\n"
            markdown_content += "---\n\n"
            
            # 添加转录内容摘要
            markdown_content += "## 音频转录摘要\n\n"
            for i, segment in enumerate(segments[:5]):  # 只显示前5段
                markdown_content += f"- [{segment['start']:.1f}s - {segment['end']:.1f}s]: {segment['text']}\n"
            if len(segments) > 5:
                markdown_content += f"\n*...还有 {len(segments)-5} 段转录内容，详见JSON文件*\n"
            
            markdown_content += "\n---\n\n"
            markdown_content += "## PPT内容提取\n\n"
            
            # 处理每一帧
            for i, (frame, frame_num) in enumerate(zip(unique_frames, unique_numbers)):
                progress(0.8 + 0.2 * i / len(unique_frames), desc=f"处理第 {i+1}/{len(unique_frames)} 帧...")
                
                markdown_content += f"\n### 帧 {i+1}\n\n"
                ppt_content = self.processor.extract_ppt_content(frame, frame_num, fps)
                markdown_content += ppt_content
                markdown_content += "\n---\n"
            
            # 保存markdown文件
            md_path = tempfile.mktemp(suffix=".md")
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            # 清理临时音频文件
            os.remove(audio_path)
            for chunk_path, _ in audio_chunks:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            
            progress(1.0, desc="处理完成！")
            
            # 重新创建音频文件供下载
            final_audio_path = self.processor.extract_audio(video)
            
            return final_audio_path, transcript_json, md_path, "处理完成！所有文件已准备好下载。"
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return None, None, None, f"处理出错: {str(e)}"

# 创建Gradio应用
def create_app():
    interface = GradioInterface()
    
    with gr.Blocks(title="视频PPT提取系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎥 视频PPT内容提取系统")
        gr.Markdown("""
        ### 功能说明：
        1. **音频提取**：从视频中提取音频，支持1小时分块处理
        2. **语音转录**：使用Whisper模型转录音频内容
        3. **帧提取**：按指定间隔提取视频帧，支持ROI选择
        4. **智能去重**：使用CLIP模型去除相似帧
        5. **内容提取**：分析PPT内容并生成Markdown文档
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="上传视频", height=300)
                
                with gr.Row():
                    interval_input = gr.Number(
                        label="帧提取间隔（秒）",
                        value=5,
                        minimum=1,
                        maximum=60,
                        step=1
                    )
                    threshold_input = gr.Slider(
                        label="相似度阈值（越高越严格）",
                        minimum=0.8,
                        maximum=1.0,
                        value=0.95,
                        step=0.01
                    )
                
                with gr.Row():
                    extract_audio_btn = gr.Button("仅提取音频", variant="secondary")
                    process_btn = gr.Button("完整处理", variant="primary")
                
                status_text = gr.Textbox(label="状态信息", lines=3, interactive=False)
                
            with gr.Column(scale=1):
                preview_image = gr.Image(
                    label="预览帧（点击两次选择ROI区域）",
                    interactive=True,
                    height=400
                )
        
        with gr.Row():
            with gr.Column():
                audio_output = gr.File(label="📁 提取的音频", interactive=False)
            with gr.Column():
                transcript_output = gr.File(label="📁 转录结果 (JSON)", interactive=False)
            with gr.Column():
                markdown_output = gr.File(label="📁 PPT内容 (Markdown)", interactive=False)
        
        # 使用示例
        gr.Markdown("""
        ### 使用步骤：
        1. 上传视频文件
        2. 在预览帧上点击两次选择感兴趣的区域（可选）
        3. 设置帧提取间隔和相似度阈值
        4. 点击"完整处理"开始处理
        5. 等待处理完成后下载结果文件
        
        ### 注意事项：
        - 首次运行需要下载模型，请耐心等待
        - 处理大文件可能需要较长时间
        - 建议使用GPU加速处理
        """)
        
        # 事件处理
        video_input.change(
            interface.upload_video,
            inputs=[video_input],
            outputs=[preview_image, status_text, audio_output]
        )
        
        preview_image.select(
            interface.handle_click,
            inputs=[],
            outputs=[preview_image, status_text, audio_output]
        )
        
        extract_audio_btn.click(
            interface.extract_audio_only,
            inputs=[video_input],
            outputs=[audio_output, status_text]
        )
        
        process_btn.click(
            interface.process_video_full,
            inputs=[video_input, interval_input, threshold_input],
            outputs=[audio_output, transcript_output, markdown_output, status_text]
        )
    
    return demo

# 运行应用
if __name__ == "__main__":
    app = create_app()

    # 修改为：
    app.launch(
    share=True,
    max_threads=10
    )