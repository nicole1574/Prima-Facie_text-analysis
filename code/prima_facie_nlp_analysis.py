# prima_facie_nlp_analysis.py
# 剧本文本结构分析脚本 —— 用于分析《Prima Facie》中法律话语与女性身体经验的张力
# 前提条件：确保已安装以下库并准备数据文件（见说明）

import os
import json
from collections import Counter
from nltk import bigrams
from textblob import TextBlob
from keybert import KeyBERT
from bertopic import BERTopic
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# ====== 一、加载数据 ======

with open('./processed_data/normalized_sentences.json', 'r', encoding='utf-8') as f:
    normalized_sentences = json.load(f)

with open('./processed_data/parts.json', 'r', encoding='utf-8') as f:
    parts = json.load(f)

# 根据 parts.json 计算 Part One 和 Part Two 的句子总数（可选，仅用于打印检查）
part_one_sentence_count = sum(len(sents) for sents in parts["Part One"].values())
part_two_sentence_count = sum(len(sents) for sents in parts["Part Two"].values())

print(f"Part One total scenes: {len(parts['Part One'])}, total sentences approx: {part_one_sentence_count}")
print(f"Part Two total scenes: {len(parts['Part Two'])}, total sentences approx: {part_two_sentence_count}")

# 将 normalized_sentences 按 Part One 和 Part Two 的结构，合并成两个句子列表
def flatten_part_sentences(normalized_data, part_name):
    all_sents = []
    scenes = normalized_data.get(part_name, {})
    # 按场景标题排序保证顺序一致（如果场景名是有序的）
    for scene_title in sorted(scenes.keys()):
        all_sents.extend(scenes[scene_title])
    return all_sents

part1_sentences = flatten_part_sentences(normalized_sentences, "Part One")
part2_sentences = flatten_part_sentences(normalized_sentences, "Part Two")

print(f"Part One sentences count: {len(part1_sentences)}")
print(f"Part Two sentences count: {len(part2_sentences)}")



# ====== 二、功能函数定义 ======

def word_freq(sentences):
    all_words = [word for sent in sentences for word in sent.split()]
    return Counter(all_words).most_common(30)

def top_bigrams_with_pmi(sentences, top_n=15):
    all_words = [word for sent in sentences for word in sent.split()]
    finder = BigramCollocationFinder.from_words(all_words)
    scored = finder.score_ngrams(BigramAssocMeasures.pmi)
    return sorted(scored, key=lambda x: -x[1])[:top_n]

def sentiment_analysis(sentences):
    return [
        {
            "sentence": s,
            "polarity": TextBlob(s).sentiment.polarity,
            "subjectivity": TextBlob(s).sentiment.subjectivity
        } for s in sentences
    ]

def extract_keywords(sentences):
    kw_model = KeyBERT()
    return kw_model.extract_keywords(" ".join(sentences), keyphrase_ngram_range=(1, 2), stop_words='english', top_n=30)

def topic_modeling(sentences, html_output):
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(sentences)

     # 提取主题信息
    topic_info = topic_model.get_topic_info()
    print("主题建模结果：")
    for index, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id != -1:  # 排除“不相关”主题
            keywords = topic_model.get_topic(topic_id)
            print(f"Topic {topic_id}: {', '.join([kw[0] for kw in keywords])}")
    # 可视化主题
    topic_model.visualize_barchart(top_n_topics=5).write_html(html_output)

def plot_sentiment_trend(sentiments, part_label):
    df = pd.DataFrame(sentiments)
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=range(len(df)), y="polarity", data=df, label="Polarity")
    sns.lineplot(x=range(len(df)), y="subjectivity", data=df, label="Subjectivity")
    plt.title(f"Sentiment Trends - {part_label}")
    plt.xlabel("Sentence Index")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(f"./output/sentiment_trend_{part_label}.png")
    plt.close()

def generate_wordcloud(keywords, filename):
    freq_dict = dict(keywords)
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_dict)
    wc.to_file(f"./output/{filename}")

def plot_word_freq(freq, part_label):
    words, counts = zip(*freq)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(words), y=list(counts))
    plt.title(f"Top 30 Word Frequencies - {part_label}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"./output/word_freq_{part_label}.png")
    plt.close()

# ====== 三、创建输出目录 ======
os.makedirs("./output", exist_ok=True)

# ====== 四、分析流程执行 ======

# --- Part 1 ---
print("分析 Part 1...")
freq1 = word_freq(part1_sentences)
print("词频数据 Part 1:", freq1)
bigrams1 = top_bigrams_with_pmi(part1_sentences)
print("大词对数据 Part 1:", bigrams1)
sentiments1 = sentiment_analysis(part1_sentences)
print("情感分析结果 Part 1:", sentiments1)
keywords1 = extract_keywords(part1_sentences)
print("关键词 Part 1:", keywords1)
topic_modeling(part1_sentences, "./output/topic_barchart_part1.html")
plot_sentiment_trend(sentiments1, "part1")
generate_wordcloud(keywords1, "wordcloud_part1.png")
plot_word_freq(freq1, "part1")



# --- Part 2 ---
print("分析 Part 2...")
freq2 = word_freq(part2_sentences)
print("词频数据 Part 2:", freq2)
bigrams2 = top_bigrams_with_pmi(part2_sentences)
print("大词对数据 Part 2:", bigrams2)
sentiments2 = sentiment_analysis(part2_sentences)
print("情感分析结果 Part 2:", sentiments2)
keywords2 = extract_keywords(part2_sentences)
print("关键词 Part 2:", keywords2)
topic_modeling(part2_sentences, "./output/topic_barchart_part2.html")
plot_sentiment_trend(sentiments2, "part2")
generate_wordcloud(keywords2, "wordcloud_part2.png")
plot_word_freq(freq2, "part2")
# ====== 五、保存结果为 JSON 文件 ======
with open("./output/word_freq_part1.json", "w", encoding="utf-8") as f:
    json.dump(freq1, f, ensure_ascii=False, indent=2)

with open("./output/word_freq_part2.json", "w", encoding="utf-8") as f:
    json.dump(freq2, f, ensure_ascii=False, indent=2)

with open("./output/bigrams_part1.json", "w", encoding="utf-8") as f:
    json.dump(bigrams1, f, ensure_ascii=False, indent=2)

with open("./output/bigrams_part2.json", "w", encoding="utf-8") as f:
    json.dump(bigrams2, f, ensure_ascii=False, indent=2)

print("✅ 分析完成！所有文件已保存在 ./output 目录中。")
