import fitz  # PyMuPDF
import re
import spacy
import json
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 创建存储路径
output_dir = "./processed_data"
os.makedirs(output_dir, exist_ok=True)

# Step 1: 提取 PDF 文本并清洗
def extract_and_clean_text(pdf_path, output_path):
    """
    从 PDF 文件提取纯文本内容，清洗后保存为 txt 文件
    :param pdf_path: PDF 文件路径
    :param output_path: 输出 txt 文件路径
    """
    with fitz.open(pdf_path) as pdf:
        text = ""
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text()  # 提取每页文本
            text += page_text
    # 清洗文本：去除多余换行符和空白符号
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    # 保存清洗后的文本
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(cleaned_text)
    print(f"清洗后的文本已保存到 {output_path}")

# Step 2: 按场景分割文本
def split_by_scene(text, max_scene=18):
    """
    按场景分割文本并返回场景列表
    :param text: 清洗后的文本内容
    :param max_scene: 最大场景编号
    :return: 场景的结构化列表
    """
    # 动态生成正则表达式用于匹配场景
    scene_pattern = r'\bScene (One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|' \
                    r'Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen)\b'
    scenes = re.split(scene_pattern, text)
    structured_scenes = []
    for i in range(1, len(scenes), 2):  # 每两个元素一组：场景标题和内容
        title = f"Scene {scenes[i].capitalize()}"
        content = scenes[i + 1].strip()
        structured_scenes.append({"title": title, "content": content})
    return structured_scenes

def save_scenes_to_json(scenes, output_path):
    """
    将场景列表保存为 JSON 文件
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(scenes, file, ensure_ascii=False, indent=4)
    print(f"分割后的场景已保存到 {output_path}")

# Step 3: 数据分段为 Part One 和 Part Two
def split_into_parts(scenes):
    """
    将场景划分为 Part One 和 Part Two
    :param scenes: 场景列表
    :return: 按部分组织的嵌套字典
    """
    parts = {"Part One": {}, "Part Two": {}}
    for i, scene in enumerate(scenes):
        part = "Part One" if i < 7 else "Part Two"
        parts[part][scene["title"]] = scene["content"]
    return parts

def save_parts_to_json(parts, output_path):
    """
    将分段后的数据保存为 JSON 文件
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(parts, file, ensure_ascii=False, indent=4)
    print(f"分段后的数据已保存到 {output_path}")

# Step 4: 句子切分
def split_into_sentences(text):
    """
    按标点符号切分文本为句子
    :param text: 场景内容
    :return: 切分后的句子列表
    """
    sentences = re.split(r'[.!?]\s*', text.strip())
    return [sentence.strip() for sentence in sentences if sentence]

def save_sentences_to_json(parts, output_path):
    """
    将分段后的每部分的句子切分结果保存为 JSON 文件
    """
    sentences_data = {}
    for part, scenes in parts.items():
        sentences_data[part] = {}
        for title, content in scenes.items():
            sentences_data[part][title] = split_into_sentences(content)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(sentences_data, file, ensure_ascii=False, indent=4)
    print(f"切分后的句子数据已保存到 {output_path}")

# Step 5: 词汇标准化
def normalize_text(sentences, stopwords):
    """
    对句子列表进行小写化、词形还原和停用词过滤
    :param sentences: 切分后的句子列表
    :param stopwords: 停用词集合
    :return: 标准化后的句子列表
    """
    nlp = spacy.load("en_core_web_sm")
    normalized = []
    for sentence in sentences:
        doc = nlp(sentence.lower())
        tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stopwords]
        normalized.append(" ".join(tokens))
    return normalized

def save_normalized_sentences(parts, stopwords, output_path):
    """
    将标准化后的句子数据保存为 JSON 文件
    """
    normalized_data = {}
    for part, scenes in parts.items():
        normalized_data[part] = {}
        for title, content in scenes.items():
            sentences = split_into_sentences(content)
            normalized_data[part][title] = normalize_text(sentences, stopwords)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(normalized_data, file, ensure_ascii=False, indent=4)
    print(f"标准化后的句子数据已保存到 {output_path}")

# 主流程
if __name__ == "__main__":
    # 配置路径
    pdf_path = "./raw_data/PrimaFacie_text.pdf"
    cleaned_text_path = os.path.join(output_dir, "cleaned_text.txt")
    scenes_json_path = os.path.join(output_dir, "scenes.json")
    parts_json_path = os.path.join(output_dir, "parts.json")
    sentences_json_path = os.path.join(output_dir, "sentences.json")
    normalized_json_path = os.path.join(output_dir, "normalized_sentences.json")
     # 停用词列表
    custom_stopwords = set([
    # 通用英文停用词
    'the', 'a', 'an', 'of', 'in', 'on', 'to', 'with', 'at', 'for',
    'so', 'and', 'or', 'but', 'because', 'although', 'if', 'while',
    'this', 'that', 'these', 'those', 'as', 'by', 'from', 'about',
    'it', 'its', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'not', 'no', 'yes',
    'i', 'me', 'my', 'mine', 'you', 'your', 'we', 'us', 'they', 'them',
    
    # 表演常用填充词（剧场语言特征，无明确含义）
    'oh', 'okay', 'yeah', 'hmm', 'like', 'sort', 'um', 'uh', 'really', 'actually',
    'thing', 'stuff', 'bit', 'maybe', 'just', 'quite', 'right', 'still', 'always', 'never',

    # 剧本中高频无信息的词
    'blah', 'blahblah', 'scene', 'part', 'mr', 'sir', 'one', 'two', 'three', 'm', 'm.',

    # 时间词（不影响语义建模）
    'day', 'night', 'today', 'week', 'month', 'year', 'now', 'then', 'last', 'next'
    ])
    stopwords = custom_stopwords
    '''
    base_stopwords = set([
        "the", "and", "is", "in", "at", "of", "to", "a", "an", "on", "for", "with",
        "as", "by", "it", "this", "that", "these", "those", "be", "was", "were",
        "are", "am", "has", "have", "had", "do", "does", "did"
    ])
    custom_stopwords = set(["scene", "part", "act", "prima", "facie", "character", "dialogue"])
    '''

    

    # 执行各步骤
    extract_and_clean_text(pdf_path, cleaned_text_path)
    with open(cleaned_text_path, "r", encoding="utf-8") as file:
        text = file.read()
    scenes = split_by_scene(text)
    save_scenes_to_json(scenes, scenes_json_path)
    parts = split_into_parts(scenes)
    save_parts_to_json(parts, parts_json_path)
    save_sentences_to_json(parts, sentences_json_path)
    save_normalized_sentences(parts, stopwords, normalized_json_path)