import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except:
    pass
import pandas as pd

import numpy as np
import re
import string
from collections import Counter
import nltk

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# 加载数据
df = pd.read_csv('//Users//liduo//Desktop//Womens.csv')
# 下载必要的NLTK数据
# ========== 修复 SSL 报错（必须放在最顶部）==========


# ========== 你的正常代码 ==========
import nltk
import pandas as pd
import re
from collections import Counter

# 下载 nltk 资源
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
# 查看评论文本的样本
# 统计非空评论数量
valid_reviews = df['Review Text'].dropna()
print(f"\n\n有效评论数量: {len(valid_reviews)}")
print(f"总评论数量: {len(df)}")

# 查看评论长度分布
review_lengths = valid_reviews.str.len()
print(f"\n评论长度统计:")
print(f"平均长度: {review_lengths.mean():.2f}")
print(f"中位数长度: {review_lengths.median():.2f}")
print(f"最短: {review_lengths.min()}")
print(f"最长: {review_lengths.max()}")


class FashionReviewPreprocessor:
    """时尚评论文本预处理器 - 适用于Python 3.14"""

    def __init__(self):
        # 自定义分词模式
        self.custom_tokenizer = None

        # 基础停用词
        self.base_stopwords = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
            'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above',
            'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
            't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o',
            're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
            'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn',
            'weren', 'won', 'wouldn'
        ])

        # 时尚领域自定义停用词
        self.fashion_stopwords = set([
            # 购物通用词
            'retailer', 'store', 'online', 'order', 'ordered', 'bought', 'buy',
            'purchase', 'purchased', 'shopping', 'shop', 'website', 'site', 'item',
            'product', 'piece', 'clothing', 'clothes', 'apparel', 'garment',

            # 人称代词和主观表达
            'im', 'ive', 'id', 'ill', 'youre', 'youd', 'youll', 'hes', 'shes',
            'its', 'were', 'theyre', 'thats', 'whats', 'wheres', 'hows', 'whens',

            # 无意义填充词
            'really', 'actually', 'definitely', 'probably', 'maybe', 'perhaps',
            'somewhat', 'quite', 'pretty', 'fairly', 'rather', 'kind', 'sort',
            'basically', 'literally', 'honestly', 'personally', 'truly', 'simply',
            'absolutely', 'completely', 'totally', 'extremely', 'very', 'too',

            # 动作词（在评论中过于常见）
            'got', 'get', 'getting', 'gotten', 'went', 'go', 'going', 'gone',
            'came', 'come', 'coming', 'take', 'took', 'taking', 'taken',
            'make', 'made', 'making', 'put', 'puts', 'see', 'saw', 'seen',
            'look', 'looked', 'looking', 'try', 'tried', 'trying', 'wear',
            'wore', 'wearing', 'worn', 'use', 'used', 'using', 'need',
            'needed', 'needing', 'want', 'wanted', 'wanting', 'like',
            'liked', 'liking', 'love', 'loved', 'loving', 'hate', 'hated',

            # 尺寸相关（过于具体，但保留size相关情感词）
            'xs', 's', 'm', 'l', 'xl', 'xxl', 'xxxl', 'petite', 'regular',
            'small', 'medium', 'large', 'extra', 'size', 'sized', 'sizing',

            # 时间词
            'today', 'yesterday', 'tomorrow', 'week', 'month', 'year',
            'day', 'days', 'time', 'times', 'moment', 'minute', 'hour',

            # 其他通用词
            'thing', 'things', 'way', 'ways', 'part', 'parts', 'bit',
            'lot', 'lots', 'ton', 'tons', 'amount', 'number', 'piece',
            'back', 'return', 'returned', 'returning', 'exchange', 'exchanged',
            'keep', 'kept', 'keeping', 'send', 'sent', 'sending'
        ])

        # 合并停用词
        self.all_stopwords = self.base_stopwords | self.fashion_stopwords

        # 词干提取器
        self.stemmer = None
        self.lemmatizer = None

        # 统计信息
        self.stats = {
            'original_tokens': [],
            'after_custom_tokenization': [],
            'after_stopwords': [],
            'after_stemming': [],
            'custom_tokens_added': set()
        }

    def custom_tokenize(self, text):
        """
        自定义分词函数
        针对时尚评论的特殊需求设计
        """
        if pd.isna(text) or text == '':
            return []

        # 转换为小写
        text = str(text).lower()

        # 步骤1: 保护特殊表达（带空格的复合词）
        # 保护 "5'8"" 这样的身高表达
        height_pattern = r"(\d+'\d+\"?)"
        heights = re.findall(height_pattern, text)
        for i, h in enumerate(heights):
            text = text.replace(h, f" HEIGHT_{i} ")

        # 保护 "5'2"" 这样的身高
        short_height = r"(\d+'\d*)"
        short_heights = re.findall(short_height, text)
        for i, h in enumerate(short_heights):
            if h not in heights:
                text = text.replace(h, f" SHORTHEIGHT_{i} ")

        # 保护 "34b", "36d" 等罩杯尺寸
        cup_pattern = r"(\d+[a-d])"
        cups = re.findall(cup_pattern, text)
        for i, c in enumerate(cups):
            text = text.replace(c, f" CUP_{i} ")

        # 步骤2: 处理标点符号
        # 保留撇号在缩写中，但分离其他标点
        text = re.sub(r"([^a-zA-Z0-9'_])", r" \1 ", text)

        # 步骤3: 处理数字（保留但标记）
        # 保留纯数字（可能是评分、价格等）
        text = re.sub(r"(\d+\.\d+)", r" DECIMAL_\1 ", text)
        text = re.sub(r"(\d+)", r" NUM_\1 ", text)

        # 步骤4: 分词
        tokens = text.split()


        # 步骤5: 恢复特殊标记为实际值
        result_tokens = []
        for token in tokens:
            if token.startswith('HEIGHT_'):
                if token.startswith('__COMPOUND__') and len(token.split('_')) >= 3:
                    idx = int(token.split('_')[2])
                if idx < len(heights):
                    result_tokens.append(heights[idx])
            elif token.startswith('SHORTHEIGHT_'):
                if token.startswith('__COMPOUND__') and len(token.split('_')) >= 3:
                    idx = int(token.split('_')[2])
                if idx < len(short_heights):
                    result_tokens.append(short_heights[idx])
            elif token.startswith('CUP_'):
                if token.startswith('__COMPOUND__') and len(token.split('_')) >= 3:
                    idx = int(token.split('_')[2])
                if idx < len(cups):
                    result_tokens.append(cups[idx])
            elif token.startswith('NUM_'):
                result_tokens.append(token.replace('NUM_', ''))
            elif token.startswith('DECIMAL_'):
                result_tokens.append(token.replace('DECIMAL_', ''))
            else:
                # 清理剩余标点
                token = re.sub(r"^[^a-zA-Z0-9']+", "", token)
                token = re.sub(r"[^a-zA-Z0-9']+$", "", token)
                if len(token) > 1 or token in ['i', 'a']:
                    result_tokens.append(token)

        return result_tokens

    def remove_custom_stopwords(self, tokens):
        """移除自定义停用词"""
        return [t for t in tokens if t.lower() not in self.all_stopwords and len(t) > 1]

    def simple_stem(self, word):
        """简单的词干提取（不依赖NLTK）"""
        # 处理常见复数形式
        if word.endswith('ies') and len(word) > 4:
            return word[:-3] + 'y'
        elif word.endswith('es') and len(word) > 3:
            return word[:-2]
        elif word.endswith('s') and not word.endswith('ss') and len(word) > 3:
            return word[:-1]
        elif word.endswith('ing') and len(word) > 5:
            return word[:-3]
        elif word.endswith('ed') and len(word) > 4:
            return word[:-2]
        elif word.endswith('ly') and len(word) > 4:
            return word[:-2]
        elif word.endswith('ment') and len(word) > 6:
            return word[:-4]
        elif word.endswith('ness') and len(word) > 5:
            return word[:-4]
        elif word.endswith('tion') and len(word) > 5:
            return word[:-4]
        return word

    def apply_stemming(self, tokens):
        """应用词干提取"""
        return [self.simple_stem(t) for t in tokens]

    def preprocess(self, text, apply_stem=True):
        """完整的预处理流程"""
        # 步骤1: 自定义分词
        tokens = self.custom_tokenize(text)
        self.stats['original_tokens'].append(len(tokens))

        # 步骤2: 移除停用词
        tokens_no_stop = self.remove_custom_stopwords(tokens)
        self.stats['after_stopwords'].append(len(tokens_no_stop))

        # 步骤3: 词干提取（可选）
        if apply_stem:
            tokens_stemmed = self.apply_stemming(tokens_no_stop)
            self.stats['after_stemming'].append(len(tokens_stemmed))
            return tokens_stemmed

        return tokens_no_stop

    def get_statistics(self):
        """获取处理统计信息"""
        stats = {}
        if self.stats['original_tokens']:
            stats['avg_original_tokens'] = np.mean(self.stats['original_tokens'])
            stats['avg_after_stopwords'] = np.mean(self.stats['after_stopwords'])
            if self.stats['after_stemming']:
                stats['avg_after_stemming'] = np.mean(self.stats['after_stemming'])
            stats['total_docs'] = len(self.stats['original_tokens'])
        return stats


# 初始化预处理器
preprocessor = FashionReviewPreprocessor()
print("预处理器初始化完成")
print(f"基础停用词数量: {len(preprocessor.base_stopwords)}")
print(f"时尚领域停用词数量: {len(preprocessor.fashion_stopwords)}")
print(f"总停用词数量: {len(preprocessor.all_stopwords)}")

# 测试预处理效果 - 展示自定义分词的优势
test_reviews = [
    "I'm 5'8\" and usually wear a 34B. This dress fits perfectly! Love it so much.",
    "The petite small is too tight around my 36D chest. I'm 5'2\" and 125 lbs.",
    "This shirt is super cute but the XS runs small. I'd recommend sizing up.",
    "Love, love, love this jumpsuit! It's fun, flirty, and fabulous!",
    "I bought this at the retailer store in NYC. The quality is amazing for $128."
]

print("=" * 80)
print("测试自定义分词效果")
print("=" * 80)

for i, review in enumerate(test_reviews, 3):
    print(f"\n【测试 {i}】")
    print(f"原文: {review}")

    # 标准分词（简单空格分割）
    simple_tokens = review.lower().split()
    print(f"\n简单分词结果 ({len(simple_tokens)} tokens):")
    print(simple_tokens[:15], "..." if len(simple_tokens) > 15 else "")

    # 自定义分词

    custom_tokens = preprocessor.custom_tokenize(review)
    print(f"\n自定义分词结果 ({len(custom_tokens)} tokens):")
    print(custom_tokens[:15], "..." if len(custom_tokens) > 15 else "")

    # 停用词移除后
    no_stop = preprocessor.remove_custom_stopwords(custom_tokens)
    print(f"\n移除停用词后 ({len(no_stop)} tokens):")
    print(no_stop[:15], "..." if len(no_stop) > 15 else "")

    # 词干提取后
    stemmed = preprocessor.apply_stemming(no_stop)
    print(f"\n词干提取后 ({len(stemmed)} tokens):")
    print(stemmed[:15], "..." if len(stemmed) > 15 else "")

    print("-" * 80)
