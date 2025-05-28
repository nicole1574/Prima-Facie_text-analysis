import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from config import config 
# 下载 NLTK 必需的数据包
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

# 创建存储清洗后数据的文件夹
output_folder = config.PROCESSED_DATA_PATH
os.makedirs(output_folder, exist_ok=True)

# 文本清洗和预处理函数
def preprocess_text(text):
    # 初始化工具
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # 1. 转为小写
    text = text.lower()

    # 2. 分词
    tokens = word_tokenize(text)

    # 3. 去除标点符号和数字
    tokens = [word for word in tokens if word.isalnum()]

    # 4. 去除停用词
    tokens = [word for word in tokens if word not in stop_words]

    # 5. 词形还原（词性还原）
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens

# 主处理函数：读取文本并保存到 CSV 文件
def process_and_save_to_csv(input_file, output_file):
    # 读取文本文件
    with open(input_file, "r", encoding="utf-8") as file:
        text = file.read()

    # 调用清洗函数
    cleaned_tokens = preprocess_text(text)

    # 转换为 DataFrame
    df = pd.DataFrame(cleaned_tokens, columns=["Word"])

    # 保存到 CSV 文件
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"清洗后的数据已保存到: {output_file}")

# 示例调用
if __name__ == "__main__":
    # 输入和输出文件路径
    input_file =config.DATA_PATH   # 替换为你的输入文本文件
    output_file = os.path.join(output_folder, "processed_data.csv")

    # 调用处理函数
    process_and_save_to_csv(input_file, output_file)