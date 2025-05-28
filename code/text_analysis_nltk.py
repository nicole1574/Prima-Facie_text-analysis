import pandas as pd
import nltk
from nltk.probability import FreqDist
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim import corpora, models
from config import config
# 下载 NLTK 必需的数据包
nltk.download('punkt')
nltk.download('stopwords')

# 读取清洗后的 CSV 文件
def load_processed_data(csv_file):
    df = pd.read_csv(csv_file)
    words = df['Word'].tolist()  # 提取单词列
    return words

# 1. 词频统计 (Word Frequency)
def word_frequency_analysis(words, top_n=10):
    freq_dist = FreqDist(words)
    most_common = freq_dist.most_common(top_n)
    print("Word Frequency)")
    for word, count in most_common:
        print(f"{word}: {count}")
    return freq_dist

# 2. 关键词提取 (Keyword Extraction)
def keyword_extraction(words, threshold=5):
    freq_dist = FreqDist(words)
    keywords = [word for word, freq in freq_dist.items() if freq > threshold]
    print("\nKeyword Extraction:")
    print(keywords)
    return keywords

# 3. N-Gram 分析
def ngram_analysis(words, n=2, top_n=10):
    n_grams = list(ngrams(words, n))
    freq_dist = FreqDist(n_grams)
    most_common = freq_dist.most_common(top_n)
    print(f"\n{n}(N-Gram Analysis):")
    for ngram, count in most_common:
        print(f"{ngram}: {count}")
    return n_grams

# 4. 共现分析 (Collocation Analysis)
def collocation_analysis(words, top_n=10):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, top_n)
    print("\n(Collocation Analysis):")
    print(bigrams)
    return bigrams

# 5. 情感分析 (Sentiment Analysis)
def sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    print("\n(Sentiment Analysis):")
    print(f"Polarity : {sentiment.polarity}, Subjectivity : {sentiment.subjectivity}")
    return sentiment

# 6. 高级分析：主题建模 (Topic Modeling)
def topic_modeling(words, num_topics=3, num_words=5):
    dictionary = corpora.Dictionary([words])
    corpus = [dictionary.doc2bow(words)]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    print("\n(Topic Modeling):")
    topics = lda_model.print_topics(num_words=num_words)
    for topic in topics:
        print(f"Topic {topic[0]}: {topic[1]}")
    return topics

# 7. 可视化：生成词云
def generate_wordcloud(words):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(" (Word Cloud)")
    plt.show()

# 主函数
if __name__ == "__main__":
    # 替换为你的 CSV 文件路径
    csv_file = "./processed_data/processed_data.csv"
    
    # 加载清洗后的数据
    words = load_processed_data(csv_file)
    text = " ".join(words)  # 将单词列表组合为文本，供情感分析使用

    # 1. 词频统计
    word_frequency_analysis(words)

    # 2. 关键词提取
    keyword_extraction(words)

    # 3. N-Gram 分析
    ngram_analysis(words, n=2)  # 二元组分析

    # 4. 共现分析
    collocation_analysis(words)

    # 5. 情感分析
    sentiment_analysis(text)

    # 6. 高级分析：主题建模
    topic_modeling(words)

    # 7. 可视化：生成词云
    generate_wordcloud(words)