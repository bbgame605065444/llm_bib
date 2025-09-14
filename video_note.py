import gradio as gr
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import whisper
import torch
import json
import os
from datetime import timedelta
import tempfile
from PIL import Image
import clip
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from typing import List, Tuple, Dict
import gc

class VideoProcessor:
    def __init__(self):
        self.video_path = None
        self.roi = None
        self.whisper_model = None
        self.clip_model = None
        self.clip_preprocess = None
        self.gemma_model = None
        self.gemma_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def extract_audio(self, video_path):
        """从视频中提取音频"""
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # 保存音频文件
        audio_path = tempfile.mktemp(suffix=".wav")
        audio.write_audiofile(audio_path, verbose=False, logger=None)
        
        video.close()
        return audio_path
    
    def split_audio_chunks(self, audio_path, chunk_duration=360, overlap=20):
        """将音频分割成1小时的块，带重叠"""
        from pydub import AudioSegment
        
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
    
    def transcribe_audio(self, audio_chunks):
        """使用Whisper转录音频"""
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model("base")
        
        all_segments = []
        
        for chunk_path, start_time in audio_chunks:
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
        del self.whisper_model
        self.whisper_model = None
        gc.collect()
        torch.cuda.empty_cache()
        
        return json_path, all_segments
    
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
        
        frames = []
        frame_numbers = []
        
        for i in range(0, total_frames, int(fps * interval_seconds)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if roi:
                    x1, y1, x2, y2 = roi
                    frame = frame[y1:y2, x1:x2]
                frames.append(frame)
                frame_numbers.append(i)
        
        cap.release()
        return frames, frame_numbers
    
    def remove_duplicate_frames(self, frames, frame_numbers, threshold=0.95):
        """使用CLIP移除重复帧"""
        if self.clip_model is None:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        unique_frames = []
        unique_numbers = []
        
        if len(frames) == 0:
            return unique_frames, unique_numbers
        
        # 预处理所有帧
        processed_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            processed = self.clip_preprocess(img).unsqueeze(0).to(self.device)
            processed_frames.append(processed)
        
        # 计算特征
        with torch.no_grad():
            features = []
            for pf in processed_frames:
                feature = self.clip_model.encode_image(pf)
                features.append(feature)
        
        # 比较相似度并去重
        unique_frames.append(frames[0])
        unique_numbers.append(frame_numbers[0])
        
        for i in range(1, len(frames)):
            is_duplicate = False
            
            for j in range(len(unique_frames)):
                # 计算余弦相似度
                sim = torch.cosine_similarity(features[i], features[unique_numbers.index(unique_numbers[j])], dim=1)
                if sim.item() > threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_frames.append(frames[i])
                unique_numbers.append(frame_numbers[i])
        
        return unique_frames, unique_numbers
    
    def extract_ppt_content(self, frame):
        """使用Gemma提取PPT内容"""
        if self.gemma_model is None:
            model_name = "google/gemma-2-2b-it"  # 使用较小的模型
            self.gemma_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.gemma_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        # 将图像转换为base64或使用OCR
        # 这里简化处理，实际应该使用OCR或多模态模型
        prompt = """请分析这个PPT页面的内容，提取：
1. 标题
2. 主要文本内容
3. 公式（使用LaTeX格式）
4. 要点列表

请用Markdown格式输出。"""
        
        inputs = self.gemma_tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.gemma_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
        
        content = self.gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的内容
        content = content.split(prompt)[-1].strip()
        
        return content
    
    def process_video(self, video_path, extract_interval, similarity_threshold):
        """完整的视频处理流程"""
        self.video_path = video_path
        
        # 1. 提取音频
        audio_path = self.extract_audio(video_path)
        
        # 2. 分割音频并转录
        audio_chunks = self.split_audio_chunks(audio_path)
        transcript_json, segments = self.transcribe_audio(audio_chunks)
        
        # 3. 提取帧
        frames, frame_numbers = self.extract_frames_with_roi(video_path, extract_interval, self.roi)
        
        # 4. 去除重复帧
        unique_frames, unique_numbers = self.remove_duplicate_frames(frames, frame_numbers, similarity_threshold)
        
        # 5. 提取PPT内容
        markdown_content = "# 视频PPT内容提取\n\n"
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        for i, (frame, frame_num) in enumerate(zip(unique_frames, unique_numbers)):
            time_seconds = frame_num / fps
            time_str = str(timedelta(seconds=int(time_seconds)))
            
            markdown_content += f"\n## 帧 {i+1} - 时间: {time_str}\n\n"
            
            # 提取PPT内容
            ppt_content = self.extract_ppt_content(frame)
            markdown_content += ppt_content + "\n\n"
            markdown_content += "---\n"
        
        # 保存markdown文件
        md_path = tempfile.mktemp(suffix=".md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # 清理音频文件
        os.remove(audio_path)
        for chunk_path, _ in audio_chunks:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
        
        return audio_path, transcript_json, md_path

# Gradio界面
class GradioInterface:
    def __init__(self):
        self.processor = VideoProcessor()
        self.preview_frame = None
        self.clicks = []
        
    def upload_video(self, video):
        """处理视频上传"""
        if video is None:
            return None, None, None
        
        # 获取第2分钟的预览帧
        frame = self.processor.get_frame_at_time(video, 120)  # 2分钟 = 120秒
        
        if frame is not None:
            self.preview_frame = frame
            return frame, "视频上传成功！请在预览帧上点击两次来选择感兴趣区域。", None
        else:
            return None, "无法读取视频帧", None
    
    def handle_click(self, evt: gr.SelectData):
        """处理鼠标点击事件"""
        if self.preview_frame is None:
            return self.preview_frame, "请先上传视频", None
        
        x, y = evt.index
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
    
    def process_video_full(self, video, interval, threshold, progress=gr.Progress()):
        """完整处理视频"""
        if video is None:
            return None, None, None, "请先上传视频"
        
        try:
            progress(0.1, desc="提取音频...")
            audio_path = self.processor.extract_audio(video)
            
            progress(0.3, desc="转录音频...")
            audio_chunks = self.processor.split_audio_chunks(audio_path)
            transcript_json, _ = self.processor.transcribe_audio(audio_chunks)
            
            progress(0.5, desc="提取视频帧...")
            frames, frame_numbers = self.processor.extract_frames_with_roi(
                video, interval, self.processor.roi
            )
            
            progress(0.7, desc="去除重复帧...")
            unique_frames, unique_numbers = self.processor.remove_duplicate_frames(
                frames, frame_numbers, threshold
            )
            
            progress(0.9, desc="提取PPT内容...")
            # 创建markdown内容
            markdown_content = "# 视频PPT内容提取\n\n"
            
            cap = cv2.VideoCapture(video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            for i, (frame, frame_num) in enumerate(zip(unique_frames, unique_numbers)):
                time_seconds = frame_num / fps
                time_str = str(timedelta(seconds=int(time_seconds)))
                
                markdown_content += f"\n## 帧 {i+1} - 时间: {time_str}\n\n"
                markdown_content += "PPT内容将在这里显示...\n\n"
                markdown_content += "---\n"
            
            # 保存文件
            md_path = tempfile.mktemp(suffix=".md")
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            progress(1.0, desc="处理完成！")
            
            # 清理
            os.remove(audio_path)
            for chunk_path, _ in audio_chunks:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            
            return audio_path, transcript_json, md_path, "处理完成！"
            
        except Exception as e:
            return None, None, None, f"处理出错: {str(e)}"

# 创建Gradio应用
def create_app():
    interface = GradioInterface()
    
    with gr.Blocks(title="视频PPT提取系统") as demo:
        gr.Markdown("# 视频PPT内容提取系统")
        gr.Markdown("上传视频，提取音频和PPT内容，生成Markdown文档")
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="上传视频")
                
                with gr.Row():
                    interval_input = gr.Number(
                        label="帧提取间隔（秒）",
                        value=5,
                        minimum=1,
                        maximum=60
                    )
                    threshold_input = gr.Slider(
                        label="相似度阈值",
                        minimum=0.8,
                        maximum=1.0,
                        value=0.95,
                        step=0.01
                    )
                
                process_btn = gr.Button("开始处理", variant="primary")
                
            with gr.Column():
                preview_image = gr.Image(
                    label="预览帧（第2分钟）- 点击两次选择区域",
                    interactive=True
                )
                status_text = gr.Textbox(label="状态", interactive=False)
        
        with gr.Row():
            audio_output = gr.File(label="提取的音频")
            transcript_output = gr.File(label="转录结果(JSON)")
            markdown_output = gr.File(label="PPT内容(Markdown)")
        
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
        
        process_btn.click(
            interface.process_video_full,
            inputs=[video_input, interval_input, threshold_input],
            outputs=[audio_output, transcript_output, markdown_output, status_text]
        )
    
    return demo

# 运行应用
if __name__ == "__main__":
    app = create_app()
    app.launch(share=True)