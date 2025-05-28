# config.py

class Config:
    # 路径配置
    DATA_PATH = "./raw_data/PrimaFacie_text.txt"            # 数据文件的存储路径
    PROCESSED_DATA_PATH = "./processed_data"                # 处理后的数据存储路径
    OUTPUT_PATH = "./output"                                # 分析结果或导出文件的存储路径
    LOG_PATH = "./logs"                                     # 日志文件的存储路径

    # 文本处理相关配置
    ENCODING = "utf-8"                                      # 默认文件编码格式
    STOPWORDS_PATH = "./data/stopwords.txt"                 # 停用词文件路径

    # 分析参数
    DEFAULT_NGRAM = 2                                       # 默认 N-Gram 分析的 n 值
    KEYWORD_THRESHOLD = 5                                   # 关键词提取的频率阈值
    TOP_WORDS_COUNT = 10                                    # 词频统计中返回的高频词数量

    # 日志配置
    LOG_LEVEL = "INFO"                                      # 日志记录级别 ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

    # 情感分析相关
    SENTIMENT_MODEL = "TextBlob"                            # 情感分析模型名称，可以切换为其他模型

    # 数据库配置（如果需要）
    # DB_HOST = "localhost"
    # DB_PORT = 3306
    # DB_NAME = "example_db"
    # DB_USER = "user"
    # DB_PASSWORD = "password"

    # 其他配置可根据具体需求添加

# 创建配置实例
config = Config()