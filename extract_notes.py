import json

def extract_notes_to_md(json_file_path, md_file_path):
    """
    从JSON文件中提取语音转文本的笔记内容，并保存为Markdown文件
    
    Args:
        json_file_path (str): JSON文件路径
        md_file_path (str): 输出的Markdown文件路径
    """
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 提取所有文本内容
    notes = []
    for segment in data:
        if 'text' in segment:
            notes.append(segment['text'])
    
    # 将文本内容写入Markdown文件
    with open(md_file_path, 'w', encoding='utf-8') as file:
        # 写入标题
        file.write("# 语音转文本笔记\n\n")
        
        # 写入内容
        for i, note in enumerate(notes, 1):
            file.write(f"{note}\n\n")
    
    print(f"笔记已成功提取到 {md_file_path}")

if __name__ == "__main__":
    # 设置文件路径
    json_file = "/home/campus.ncl.ac.uk/c1041562/Desktop/codes/article_agent/tmpnzqqe8e4.json"
    md_file = "/home/campus.ncl.ac.uk/c1041562/Desktop/codes/article_agent/speech_notes.md"
    
    # 提取笔记
    extract_notes_to_md(json_file, md_file)