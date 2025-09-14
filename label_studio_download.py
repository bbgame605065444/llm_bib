# 4fc08b3f06e5a93cef76028a2ba2a21c58e30c8e
import requests
import json
import os
from pathlib import Path
from urllib.parse import urljoin, urlparse
import time

class LabelStudioFileDownloader:
    def __init__(self, label_studio_url, api_token):
        self.base_url = label_studio_url.rstrip('/')
        self.api_token = api_token
        self.headers = {'Authorization': f'Token {api_token}'}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def download_project_files(self, project_id, output_dir, rename_with_original=True):
        """
        下载项目中的所有媒体文件
        
        Args:
            project_id: 项目ID
            output_dir: 输出目录
            rename_with_original: 是否使用原始文件名重命名
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取项目中的所有任务
        tasks_url = f"{self.base_url}/api/projects/{project_id}/tasks"
        response = self.session.get(tasks_url)
        response.raise_for_status()
        
        tasks = response.json()
        downloaded_files = []
        
        print(f"找到 {len(tasks)} 个任务")
        
        for i, task in enumerate(tasks, 1):
            print(f"\n处理任务 {i}/{len(tasks)} (ID: {task['id']})")
            
            # 获取文件信息
            original_filename = task.get('file_upload', '')
            file_path = task['data'].get('video', '') or task['data'].get('image', '') or task['data'].get('audio', '')
            
            if not file_path:
                print(f"  跳过：未找到媒体文件")
                continue
            
            # 构建下载URL
            if file_path.startswith('/data/'):
                # Label Studio内部路径
                download_url = f"{self.base_url}{file_path}"
            else:
                # 可能是完整URL
                download_url = file_path
            
            try:
                # 下载文件
                file_info = self.download_file(
                    download_url, 
                    output_dir, 
                    original_filename if rename_with_original else None,
                    task['id']
                )
                
                if file_info:
                    file_info.update({
                        'task_id': task['id'],
                        'original_filename': original_filename,
                        'ls_file_path': file_path
                    })
                    downloaded_files.append(file_info)
                    print(f"  ✓ 已下载: {file_info['saved_filename']}")
                else:
                    print(f"  ✗ 下载失败")
                    
            except Exception as e:
                print(f"  ✗ 下载出错: {str(e)}")
                continue
            
            # 避免请求过快
            time.sleep(0.1)
        
        # 保存下载记录
        self.save_download_record(downloaded_files, output_dir)
        
        return downloaded_files
    
    def download_file(self, file_url, output_dir, preferred_filename=None, task_id=None):
        """下载单个文件"""
        response = self.session.get(file_url, stream=True)
        response.raise_for_status()
        
        # 确定文件名
        if preferred_filename:
            filename = preferred_filename
        else:
            # 从URL或响应头中获取文件名
            filename = self.get_filename_from_response(response, file_url)
        
        # 如果有任务ID，添加到文件名前
        if task_id and not preferred_filename:
            name, ext = os.path.splitext(filename)
            filename = f"task{task_id}_{name}{ext}"
        
        # 处理文件名冲突
        output_path = os.path.join(output_dir, filename)
        counter = 1
        original_name, ext = os.path.splitext(filename)
        
        while os.path.exists(output_path):
            new_filename = f"{original_name}_{counter}{ext}"
            output_path = os.path.join(output_dir, new_filename)
            counter += 1
        
        # 下载文件
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        return {
            'saved_filename': os.path.basename(output_path),
            'saved_path': output_path,
            'file_size': os.path.getsize(output_path),
            'download_url': file_url
        }
    
    def get_filename_from_response(self, response, url):
        """从响应或URL中提取文件名"""
        # 尝试从Content-Disposition头获取
        content_disposition = response.headers.get('Content-Disposition', '')
        if 'filename=' in content_disposition:
            filename = content_disposition.split('filename=')[1].strip('"\'')
            return filename
        
        # 从URL路径获取
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        if filename and '.' in filename:
            return filename
        
        # 默认文件名
        content_type = response.headers.get('Content-Type', '')
        if 'video' in content_type:
            return 'video.mp4'
        elif 'image' in content_type:
            return 'image.jpg'
        else:
            return 'file.bin'
    
    def save_download_record(self, downloaded_files, output_dir):
        """保存下载记录"""
        record_file = os.path.join(output_dir, 'download_record.json')
        
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump({
                'download_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_files': len(downloaded_files),
                'files': downloaded_files
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n下载记录已保存到: {record_file}")
    
    def download_from_exported_json(self, json_file_path, output_dir, rename_with_original=True):
        """
        从导出的JSON文件中下载所有引用的媒体文件
        
        Args:
            json_file_path: 导出的JSON文件路径
            output_dir: 输出目录
            rename_with_original: 是否使用原始文件名
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        os.makedirs(output_dir, exist_ok=True)
        downloaded_files = []
        
        print(f"从JSON文件中找到 {len(data)} 个任务")
        
        for i, item in enumerate(data, 1):
            print(f"\n处理任务 {i}/{len(data)} (ID: {item.get('id', 'unknown')})")
            
            original_filename = item.get('file_upload', '')
            file_path = item.get('data', {}).get('video', '') or \
                       item.get('data', {}).get('image', '') or \
                       item.get('data', {}).get('audio', '')
            
            if not file_path:
                print(f"  跳过：未找到媒体文件路径")
                continue
            
            # 构建完整的下载URL
            if file_path.startswith('/'):
                download_url = f"{self.base_url}{file_path}"
            else:
                download_url = file_path
            
            try:
                file_info = self.download_file(
                    download_url,
                    output_dir,
                    original_filename if rename_with_original else None,
                    item.get('id')
                )
                
                if file_info:
                    file_info.update({
                        'task_id': item.get('id'),
                        'original_filename': original_filename,
                        'ls_file_path': file_path
                    })
                    downloaded_files.append(file_info)
                    print(f"  ✓ 已下载: {file_info['saved_filename']}")
                    
            except Exception as e:
                print(f"  ✗ 下载失败: {str(e)}")
                continue
            
            time.sleep(0.1)  # 避免请求过快
        
        self.save_download_record(downloaded_files, output_dir)
        return downloaded_files

# 使用示例
def main():
    # 配置Label Studio连接信息
    LABEL_STUDIO_URL = "http://localhost:8080"  # 替换为你的Label Studio地址
    API_TOKEN = "4fc08b3f06e5a93cef76028a2ba2a21c58e30c8e"           # 替换为你的API token
    
    downloader = LabelStudioFileDownloader(LABEL_STUDIO_URL, API_TOKEN)
    
    # 方法1: 直接下载项目中的所有文件
    print("=== 方法1: 通过项目ID下载 ===")
    project_id = 4  # 替换为你的项目ID
    output_dir = "./downloaded_videos"
    
    try:
        files = downloader.download_project_files(
            project_id=project_id,
            output_dir=output_dir,
            rename_with_original=True  # 使用原始文件名
        )
        print(f"\n成功下载 {len(files)} 个文件到 {output_dir}")
    except Exception as e:
        print(f"下载失败: {e}")
    
    # 方法2: 从导出的JSON文件下载
    print("\n=== 方法2: 从导出JSON下载 ===")
    json_file = "exported_annotations.json"  # 你的导出文件
    output_dir2 = "./downloaded_from_json"
    
    try:
        files2 = downloader.download_from_exported_json(
            json_file_path=json_file,
            output_dir=output_dir2,
            rename_with_original=True
        )
        print(f"\n成功下载 {len(files2)} 个文件到 {output_dir2}")
    except Exception as e:
        print(f"从JSON下载失败: {e}")

if __name__ == "__main__":
    main()