import pandas as pd
import nltk
from nltk.probability import FreqDist
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import networkx as nx
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

# 1. 词频统计 (Word Frequency) + 可视化
def word_frequency_analysis(words, top_n=10):
    freq_dist = FreqDist(words)
    most_common = freq_dist.most_common(top_n)
    
    print("Word Frequency)")
    for word, count in most_common:
        print(f"{word}: {count}")

    # 可视化
    words, frequencies = zip(*most_common)
    plt.figure(figsize=(10, 6))
    plt.bar(words, frequencies, color='skyblue')
    plt.title('Word Frequency', fontsize=16)
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=45)
    plt.show()

    return freq_dist

# 2. 关键词提取 (Keyword Extraction) + 可视化
def keyword_extraction(words, threshold=5):
    freq_dist = FreqDist(words)
    keywords = [word for word, freq in freq_dist.items() if freq > threshold]
    
    print("\nKeyword Extraction:")
    print(keywords)

    # 词云可视化
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(keywords))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Keyword Extraction Word Cloud', fontsize=16)
    plt.show()

    return keywords

# 3. N-Gram 分析 + 可视化
def ngram_analysis(words, n=2, top_n=10):
    n_grams = list(ngrams(words, n))
    freq_dist = FreqDist(n_grams)
    most_common = freq_dist.most_common(top_n)

    print(f"\n{n}(N-Gram Analysis):")
    for ngram, count in most_common:
        print(f"{ngram}: {count}")

    # 可视化
    ngrams, frequencies = zip(*most_common)
    ngrams = [" ".join(ngram) for ngram in ngrams]
    plt.figure(figsize=(12, 6))
    plt.barh(ngrams, frequencies, color='lightcoral')
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel(f'{n}-Gram', fontsize=12)
    plt.title(f'{n}-Gram Frequency', fontsize=16)
    plt.gca().invert_yaxis()
    plt.show()

    return n_grams

# 4. 共现分析 (Collocation Analysis) + 可视化
def collocation_analysis(words, top_n=10):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, top_n)
    
    print("\n(Collocation Analysis):")
    print(bigrams)

    # 网络图可视化
    G = nx.Graph()
    G.add_edges_from(bigrams)
    plt.figure(figsize=(10, 6))
    nx.draw_networkx(G, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_weight='bold')
    plt.title('Collocation Network', fontsize=16)
    plt.show()

    return bigrams

# 5. 情感分析 (Sentiment Analysis) + 可视化
def sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    
    print("\n(Sentiment Analysis):")
    print(f"Polarity : {sentiment.polarity}, Subjectivity : {sentiment.subjectivity}")

    # 饼图可视化
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity
    neutrality = 1 - abs(polarity) - subjectivity
    labels = ['Polarity', 'Subjectivity', 'Neutrality']
    sizes = [abs(polarity), subjectivity, max(neutrality, 0)]
    colors = ['gold', 'lightcoral', 'lightskyblue']
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Sentiment Analysis', fontsize=16)
    plt.show()

    return sentiment

# 6. 高级分析：主题建模 (Topic Modeling) + 可视化
def topic_modeling(words, num_topics=3, num_words=5):
    dictionary = corpora.Dictionary([words])
    corpus = [dictionary.doc2bow(words)]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    
    print("\n(Topic Modeling):")
    topics = lda_model.print_topics(num_words=num_words)
    for topic in topics:
        print(f"Topic {topic[0]}: {topic[1]}")

    # 堆叠条形图可视化
    topic_data = [[float(value.split('*')[0]) for value in topic[1].split(' + ')] for topic in topics]
    words_labels = [[value.split('"')[1] for value in topic[1].split(' + ')] for topic in topics]
    plt.figure(figsize=(12, 6))
    for i, (data, labels) in enumerate(zip(topic_data, words_labels)):
        plt.bar(labels, data, width=0.5, label=f'Topic {i}')
    plt.title('Topic Modeling Distribution', fontsize=16)
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.legend()
    plt.show()

    return topics

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