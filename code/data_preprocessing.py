import fitz  # PyMuPDF
import re
import spacy
import json
import os
import sys
import io
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter

# 创建一个download_nltk_data.py文件
import nltk
nltk.download('punkt_tab')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 创建存储路径
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

# Step 1: 提取 PDF 文本并清洗
def extract_and_clean_text(pdf_path, output_path):
    """
    从 PDF 文件提取纯文本内容，清洗后保存为 txt 文件
    :param pdf_path: PDF 文件路径
    :param output_path: 输出 txt 文件路径
    """
    try:
        with fitz.open(pdf_path) as pdf:
            text = ""
            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text()
                text += page_text
                
        # 清洗文本：去除页眉页脚和特殊格式符号
        # 移除页码
        text = re.sub(r'\b\d+\b\s*$', '', text, flags=re.MULTILINE)
        # 标准化标点符号
        text = re.sub(r'["""]', '"', text)  # 标准化引号
        text = re.sub(r"[''']", "'", text)  # 标准化单引号
        text = re.sub(r'—|–', '-', text)    # 标准化破折号
        # 去除多余换行符和空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 保存清洗后的文本
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(text)
        print(f"清洗后的文本已保存到 {output_path}")
        return text
    except Exception as e:
        print(f"提取文本时出错: {e}")
        return ""

# Step 2: 按场景分割文本，增强对场景的识别
def split_by_scene(text):
    """
    按场景分割文本并返回场景列表
    :param text: 清洗后的文本内容
    :return: 场景的结构化列表
    """
    # 使用正则表达式识别场景标记
    scene_pattern = r'\bScene\s+(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|' \
                    r'Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen)\b'
    
    # 找到所有场景标题的位置
    scene_matches = list(re.finditer(scene_pattern, text, re.IGNORECASE))
    
    # 提取各场景内容
    scenes = []
    for i in range(len(scene_matches)):
        start_pos = scene_matches[i].start()
        title = text[scene_matches[i].start():scene_matches[i].end()]
        
        # 确定场景结束位置
        if i < len(scene_matches) - 1:
            end_pos = scene_matches[i+1].start()
        else:
            end_pos = len(text)
            
        content = text[scene_matches[i].end():end_pos].strip()
        
        # 处理舞台指示信息，标记括号内容
        content = re.sub(r'\(([^)]+)\)', r' STAGE_DIRECTION: \1 ', content)
        
        scenes.append({"title": title, "content": content, "scene_number": i + 1})
        
    return scenes

# Step 3: 数据分段为 Part One 和 Part Two
def split_into_parts(scenes):
    """
    将场景划分为 Part One 和 Part Two
    :param scenes: 场景列表
    :return: 按部分组织的嵌套字典
    """
    parts = {"Part One": {}, "Part Two": {}}
    
    for scene in scenes:
        scene_number = scene["scene_number"]
        part = "Part One" if scene_number <= 7 else "Part Two"
        parts[part][f"Scene {scene_number}"] = scene["content"]
        
    return parts

# Step 4: 句子切分，使用更可靠的方法
def split_into_sentences(text):
    """
    使用NLTK的sent_tokenize提高句子分割准确性
    :param text: 场景内容
    :return: 切分后的句子列表
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # 使用NLTK进行句子分割，更好处理引号和缩写
    sentences = sent_tokenize(text)
    
    # 过滤空句子
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# 识别法律相关术语，用于语料库划分
def is_legal_discourse(sentence, legal_terms):
    """
    判断句子是否属于法律话语
    :param sentence: 待判断的句子
    :param legal_terms: 法律术语集合
    :return: 布尔值，表示是否属于法律话语
    """
    # 转小写后匹配法律术语
    sentence_lower = sentence.lower()
    # 计算法律术语出现次数
    term_count = sum(1 for term in legal_terms if term in sentence_lower)
    # 如果包含一定数量的法律术语，判定为法律话语
    return term_count >= 1

# Step 5: 构建法律话语和创伤叙事语料库
def build_corpora(parts, legal_terms):
    """
    构建法律话语和创伤叙事语料库
    :param parts: 按部分组织的场景内容
    :param legal_terms: 法律术语集合
    :return: 两个语料库的字典
    """
    legal_discourse_corpus = []
    trauma_narrative_corpus = []
    
    # 遍历Part One，主要是法律话语
    for scene, content in parts["Part One"].items():
        sentences = split_into_sentences(content)
        for sentence in sentences:
            # Part One仍需判断，有些可能是创伤叙事
            if is_legal_discourse(sentence, legal_terms):
                legal_discourse_corpus.append({"text": sentence, "source": scene, "part": "Part One"})
            else:
                trauma_narrative_corpus.append({"text": sentence, "source": scene, "part": "Part One"})
    
    # 遍历Part Two，主要是创伤叙事，但仍可能含有法律话语
    for scene, content in parts["Part Two"].items():
        sentences = split_into_sentences(content)
        for sentence in sentences:
            if is_legal_discourse(sentence, legal_terms):
                legal_discourse_corpus.append({"text": sentence, "source": scene, "part": "Part Two"})
            else:
                trauma_narrative_corpus.append({"text": sentence, "source": scene, "part": "Part Two"})
    
    return {
        "legal_discourse": legal_discourse_corpus,
        "trauma_narrative": trauma_narrative_corpus
    }

# Step 6: 词汇标准化
def normalize_text(sentences, stopwords):
    """
    对句子列表进行小写化、词形还原和停用词过滤
    :param sentences: 切分后的句子列表
    :param stopwords: 停用词集合
    :return: 标准化后的句子列表
    """
    nlp = spacy.load("en_core_web_sm")
    normalized = []
    
    for sentence_obj in sentences:
        sentence = sentence_obj["text"]
        doc = nlp(sentence.lower())
        
        # 保留情感词、法律术语、人称代词等
        tokens = []
        for token in doc:
            # 保留alpha字符，排除停用词，但保留特定类型
            if token.is_alpha and token.text not in stopwords:
                tokens.append(token.lemma_)
                
        # 添加标准化后的文本和原始元数据
        normalized.append({
            "text": " ".join(tokens),
            "original": sentence,
            "source": sentence_obj["source"],
            "part": sentence_obj["part"]
        })
    
    return normalized

def save_corpora_to_json(corpora, output_path):
    """
    将语料库保存为JSON文件
    :param corpora: 语料库字典
    :param output_path: 输出路径
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(corpora, file, ensure_ascii=False, indent=4)
    print(f"语料库已保存到 {output_path}")

def save_normalized_corpora(normalized_corpora, output_path):
    """
    保存标准化后的语料库
    :param normalized_corpora: 标准化后的语料库字典
    :param output_path: 输出路径
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(normalized_corpora, file, ensure_ascii=False, indent=4)
    print(f"标准化语料库已保存到 {output_path}")

def get_corpus_stats(corpus):
    """
    计算语料库统计信息
    :param corpus: 语料库
    :return: 统计信息字典
    """
    all_words = []
    for item in corpus:
        words = item["text"].split()
        all_words.extend(words)
    
    return {
        "sentence_count": len(corpus),
        "word_count": len(all_words),
        "unique_words": len(set(all_words)),
        "top_words": Counter(all_words).most_common(20)
    }
def save_scenes_to_json(scenes, output_path):
    """
    将场景列表保存为 JSON 文件
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(scenes, file, ensure_ascii=False, indent=4)
    print(f"分割后的场景已保存到 {output_path}")

def save_parts_to_json(parts, output_path):
    """
    将分段后的数据保存为 JSON 文件
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(parts, file, ensure_ascii=False, indent=4)
    print(f"分段后的数据已保存到 {output_path}")
# 主流程
if __name__ == "__main__":
    # 配置路径
    pdf_path = "code/raw_data/PrimaFacie_text.pdf"
    cleaned_text_path = os.path.join(output_dir, "cleaned_text.txt")
    scenes_json_path = os.path.join(output_dir, "scenes.json")
    parts_json_path = os.path.join(output_dir, "parts.json")
    corpora_json_path = os.path.join(output_dir, "corpora.json")
    normalized_corpora_json_path = os.path.join(output_dir, "normalized_corpora.json")
    stats_json_path = os.path.join(output_dir, "corpus_stats.json")
    
    # 法律术语集，用于识别法律话语
    legal_terms = {
        "court", "judge", "jury", "witness", "evidence", "testimony", "cross-examination",
        "prosecution", "defense", "defendant", "plaintiff", "barrister", "solicitor",
        "counsel", "objection", "sustained", "overruled", "verdict", "guilty", "acquittal",
        "reasonable doubt", "sworn", "oath", "exhibit", "testify", "motion", "precedent",
        "statute", "legal", "law", "allegation", "alleged", "charge", "criminal", "civil",
        "affidavit", "appeal", "jurisdiction", "conviction", "cross-examine", "examine",
        "case", "hearing", "trial", "judgment", "ruling", "miranda", "rights", "lawyer",
        "attorney", "prosecutor", "sexual assault", "disclosure", "bench", "advocate", 
        "procedural", "substantive", "prima facie", "burden of proof", "section",
        "act", "legal aid", "police", "arrest", "statement", "affirmation", "penalty",
        "offence", "offense", "bail", "juvenile", "rape", "assault", "summon", "subpoena"
    }
     
    # 停用词列表 - 保留情感词、否定词和人称代词
    custom_stopwords = set([
        # 通用英文停用词
        'the', 'a', 'an', 'of', 'in', 'on', 'to', 'with', 'at', 'for',
        'so', 'because', 'although', 'if', 'while',
        'this', 'that', 'these', 'those', 'as', 'by', 'from', 'about',
        'its', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did',
        
        # 表演常用填充词（剧场语言特征，无明确含义）
        'oh', 'okay', 'yeah', 'hmm', 'like', 'sort', 'um', 'uh', 'really', 'actually',
        'thing', 'stuff', 'bit', 'maybe', 'just', 'quite', 'right', 'still',
        
        # 剧本中高频无信息的词
        'blah', 'blahblah', 'scene', 'part', 'mr', 'sir', 'one', 'two', 'three', 'm', 'm.',
        
        # 时间词（不影响语义建模）
        'day', 'night', 'today', 'week', 'month', 'year', 'now', 'then', 'last', 'next'
    ])
    
    # 保留词（不加入停用词列表）：情感词、否定词、人称代词等
    preserved_words = {
        # 否定词
        'not', "don't", "didn't", "doesn't", "isn't", "aren't", "wasn't", "weren't", "no",
        # 情感表达
        'yes', 'angry', 'sad', 'happy', 'afraid', 'scared', 'pain', 'hurt', 'fear', 'shame',
        # 人称代词
        'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'we', 'us', 'our', 'they', 'them', 'their',
        'he', 'him', 'his', 'she', 'her', 'hers',
        # 情态动词和助动词（可能表示不确定性或情感）
        'will', 'would', 'can', 'could', 'shall', 'should', 'may', 'might', 'must',
        # 关键连词（可能表示逻辑关系）
        'but', 'and', 'or', 'yet', 'nor'
    }
    
    # 从停用词中移除保留词
    for word in preserved_words:
        if word in custom_stopwords:
            custom_stopwords.remove(word)
    
    # 执行各步骤
    text = extract_and_clean_text(pdf_path, cleaned_text_path)
    if not text:  # 如果提取失败，尝试读取已保存的文件
        with open(cleaned_text_path, "r", encoding="utf-8") as file:
            text = file.read()
    
    scenes = split_by_scene(text)
    save_scenes_to_json(scenes, scenes_json_path)
    
    parts = split_into_parts(scenes)
    save_parts_to_json(parts, parts_json_path)
    
    # 构建语料库
    corpora = build_corpora(parts, legal_terms)
    save_corpora_to_json(corpora, corpora_json_path)
    
    # 标准化语料库
    normalized_corpora = {
        "legal_discourse": normalize_text(corpora["legal_discourse"], custom_stopwords),
        "trauma_narrative": normalize_text(corpora["trauma_narrative"], custom_stopwords)
    }
    save_normalized_corpora(normalized_corpora, normalized_corpora_json_path)
    
    # 计算并保存语料库统计信息
    stats = {
        "legal_discourse": get_corpus_stats(normalized_corpora["legal_discourse"]),
        "trauma_narrative": get_corpus_stats(normalized_corpora["trauma_narrative"])
    }
    with open(stats_json_path, "w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=4)
    print(f"语料库统计信息已保存到 {stats_json_path}")
    
    # 打印语料库大小信息
    print(f"法律话语语料库: {len(corpora['legal_discourse'])} 句, 约 {stats['legal_discourse']['word_count']} 词")
    print(f"创伤叙事语料库: {len(corpora['trauma_narrative'])} 句, 约 {stats['trauma_narrative']['word_count']} 词")
