# 重新定义所有类并完整执行流程
import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# 读取数据
df = pd.read_csv('//Users//liduo//Desktop//Womens.csv')
print(f"数据加载完成: {len(df)} 行")


# ============================================================
# STEP 1: 自定义分词 (Custom Tokenization)
# ============================================================

class CustomTokenizer:
    def __init__(self):
        self.patterns = {
            'size_patterns': [
                r'\b(?:xxs|xxl|xs|xl|s|m|l)\b',
                r'\b(?:size\s*\d+|size\s*[a-z]+)\b',
                r'\b(?:petite|regular|tall|plus)\b',
                r'\b(?:\d{1,2}\s*(?:inch|in|\"|\')|(?:\d{1,2}\')\s*(?:\d{1,2}\")?)\b',
            ],
            'body_parts': [
                r'\b(?:bust|waist|hip|torso|chest|shoulder|arm|sleeve|length)\b',
            ],
            'emphasis_patterns': [
                r'\b(lo{2,}ve|so{2,}|ve{2,}ry|to{2,})\b',
            ],
            'negation_patterns': [
                r'\b(?:not\s+(?:that|too|very|really|at\s+all))\b',
                r'\b(?:never|nothing|nowhere|no\s+one)\b',
            ],
            'brand_patterns': [
                r'\b(?:retailer|anthro|pilcro|maeve|hd\s+in\s+paris|byron\s+lars)\b',
            ],
        }
        self.compiled_patterns = {}
        for category, patterns in self.patterns.items():
            self.compiled_patterns[category] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def preserve_special_tokens(self, text):
        protected = {}
        counter = 0

        for category in ['size_patterns', 'body_parts', 'emphasis_patterns', 'negation_patterns', 'brand_patterns']:
            for pattern in self.compiled_patterns.get(category, []):
                matches = list(pattern.finditer(text))
                for match in matches:
                    placeholder = f"__{category.upper()}_{counter}__"
                    protected[placeholder] = match.group().lower()
                    text = text.replace(match.group(), placeholder, 1)
                    counter += 1
        return text, protected

    def basic_tokenize(self, text):
        text = text.lower()
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'m", " am", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        tokens = text.split()
        tokens = [t for t in tokens if len(t) >= 2]
        return tokens

    def tokenize(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return []
        protected_text, protected_tokens = self.preserve_special_tokens(text)
        tokens = self.basic_tokenize(protected_text)
        final_tokens = []
        for token in tokens:
            if token in protected_tokens:
                final_tokens.append(protected_tokens[token])
            elif not token.startswith('__'):
                final_tokens.append(token)
        return final_tokens


# 初始化分词器
tokenizer = CustomTokenizer()

# 执行分词
print("\n正在执行 STEP 1: 自定义分词...")
df['tokens_step1'] = df['Review Text'].apply(tokenizer.tokenize)

# 统计
all_tokens_step1 = []
for tokens in df['tokens_step1']:
    all_tokens_step1.extend(tokens)
vocab_step1 = set(all_tokens_step1)

print(f"\n{'=' * 70}")
print("STEP 1 分词统计结果")
print("=" * 70)
print(f"总Token数: {len(all_tokens_step1):,}")
print(f"唯一词汇数 (Vocabulary Size): {len(vocab_step1):,}")
print(f"平均每条评论Token数: {len(all_tokens_step1) / len(df):.2f}")

# 显示最常见的token
print(f"\n最常见的20个Tokens:")
token_counts_step1 = Counter(all_tokens_step1)
for token, count in token_counts_step1.most_common(20):
    print(f"  {token}: {count}")


# ============================================================
# STEP 2: 自定义停用词 (Custom Stop Words)
# ============================================================

class CustomStopWords:
    def __init__(self):
        self.base_stopwords = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before',
            'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should',
            'now', 'would', 'could', 'may', 'might'
        ])

        self.ecommerce_stopwords = set([
            'get', 'got', 'getting', 'buy', 'bought', 'buying', 'purchase', 'purchased',
            'order', 'ordered', 'ordering', 'wear', 'wearing', 'wore', 'try', 'tried',
            'trying', 'look', 'looking', 'looked', 'feel', 'feeling', 'felt',
            'think', 'thinking', 'thought', 'know', 'knew', 'knowing',
            'say', 'said', 'saying', 'go', 'went', 'going', 'gone',
            'come', 'came', 'coming', 'see', 'saw', 'seeing', 'seen',
            'want', 'wanted', 'wanting', 'need', 'needed', 'needing',
            'like', 'liked', 'liking', 'love', 'loved', 'loving',
            'today', 'yesterday', 'tomorrow', 'day', 'week', 'month', 'year',
            'time', 'times', 'moment', 'second', 'minute', 'hour',
            'really', 'actually', 'definitely', 'probably', 'maybe', 'perhaps',
            'certainly', 'absolutely', 'totally', 'completely', 'quite',
            'thing', 'things', 'stuff', 'way', 'ways', 'lot', 'lots',
            'bit', 'piece', 'kind', 'sort', 'type',
            'one', 'ones', 'two', 'three', 'first', 'second', 'last',
            'something', 'anything', 'everything', 'nothing',
            'someone', 'anyone', 'everyone', 'everybody', 'somebody',
            'store', 'online', 'website', 'site', 'item', 'product',
            'clothes', 'clothing', 'dress', 'top', 'shirt', 'pants',
            'skirt', 'sweater', 'jacket', 'coat', 'jeans',
        ])

        self.fashion_stopwords = set([
            'black', 'white', 'blue', 'red', 'green', 'yellow', 'pink',
            'purple', 'orange', 'brown', 'gray', 'grey', 'navy', 'beige',
            'cream', 'ivory', 'tan', 'khaki', 'teal', 'coral',
            'cotton', 'silk', 'wool', 'polyester', 'linen', 'denim',
            'fabric', 'material', 'texture',
            'nice', 'good', 'great', 'pretty', 'cute', 'beautiful',
            'lovely', 'perfect', 'wonderful', 'amazing', 'awesome',
            'bad', 'poor', 'terrible', 'awful', 'horrible',
            'problem', 'problems', 'issue', 'issues', 'wrong',
        ])

        self.all_stopwords = (self.base_stopwords |
                              self.ecommerce_stopwords |
                              self.fashion_stopwords)

    def remove_stopwords(self, tokens):
        filtered = []
        for token in tokens:
            if token.startswith('__') and token.endswith('__'):
                filtered.append(token)
            elif '$' in token:
                filtered.append(token)
            elif any(c.isdigit() for c in token):
                filtered.append(token)
            elif token.lower() not in self.all_stopwords:
                filtered.append(token)
        return filtered


# 初始化停用词管理器
stopword_manager = CustomStopWords()

print("=" * 70)
print("STEP 2: 自定义停用词")
print("=" * 70)
print(f"\n停用词统计:")
print(f"  基础停用词: {len(stopword_manager.base_stopwords)} 个")
print(f"  电商专用停用词: {len(stopword_manager.ecommerce_stopwords)} 个")
print(f"  服装领域停用词: {len(stopword_manager.fashion_stopwords)} 个")
print(f"  总停用词数: {len(stopword_manager.all_stopwords)} 个")

# 应用停用词过滤
print("\n正在应用停用词过滤...")
df['tokens_step2'] = df['tokens_step1'].apply(stopword_manager.remove_stopwords)

# 统计
all_tokens_step2 = []
for tokens in df['tokens_step2']:
    all_tokens_step2.extend(tokens)
vocab_step2 = set(all_tokens_step2)

removed_tokens = len(all_tokens_step1) - len(all_tokens_step2)
removed_vocab = len(vocab_step1) - len(vocab_step2)

print(f"\n{'=' * 70}")
print("STEP 2 停用词过滤统计结果")
print("=" * 70)
print(f"过滤前: 总Token {len(all_tokens_step1):,}, 词汇 {len(vocab_step1):,}")
print(f"过滤后: 总Token {len(all_tokens_step2):,}, 词汇 {len(vocab_step2):,}")
print(
    f"移除: Token {removed_tokens:,} ({removed_tokens / len(all_tokens_step1) * 100:.1f}%), 词汇 {removed_vocab:,} ({removed_vocab / len(vocab_step1) * 100:.1f}%)")
print(f"唯一词汇数 (Vocabulary Size): {len(vocab_step2):,}")
print(f"\n过滤后最常见的20个Tokens:")
token_counts_step2 = Counter(all_tokens_step2)
for token, count in token_counts_step2.most_common(20):
    print(f"  {token}: {count}")


# ============================================================
# STEP 3: 词干提取 (Stemming)
# ============================================================

class CustomStemmer:
    def __init__(self):
        self.suffix_rules = [
            (r'ies$', 'y'),
            (r'([^aeiou])ies$', r'\1y'),
            (r'([^s])s$', r'\1'),
            (r'([^s])es$', r'\1'),
            (r'([^aeiou])es$', r'\1e'),
            (r'ing$', ''),
            (r'([^aeiou])ing$', r'\1'),
            (r'([^aeiou])ed$', r'\1'),
            (r'([^aeiou])er$', r'\1'),
            (r'([^aeiou])est$', r'\1'),
            (r'ly$', ''),
            (r'([^aeiou])ness$', r'\1'),
            (r'([^aeiou])ment$', r'\1'),
        ]

        self.special_mappings = {
            'petites': 'petite', 'regulars': 'regular', 'talls': 'tall',
            'sleeves': 'sleeve', 'shoulders': 'shoulder', 'straps': 'strap',
            'pockets': 'pocket', 'buttons': 'button', 'zippers': 'zipper',
            'seams': 'seam', 'hems': 'hem',
            'fits': 'fit', 'fitted': 'fit', 'fitting': 'fit',
            'fabrics': 'fabric', 'materials': 'material', 'textures': 'texture',
            'reviews': 'review', 'recommended': 'recommend', 'recommending': 'recommend',
            'returns': 'return', 'returned': 'return', 'returning': 'return',
            'exchanges': 'exchange', 'exchanged': 'exchange',
            'wears': 'wear', 'wore': 'wear', 'wearing': 'wear',
            'washes': 'wash', 'washed': 'wash', 'washing': 'wash',
            'shrinks': 'shrink', 'shrank': 'shrink', 'shrunk': 'shrink',
            'colors': 'color', 'colours': 'color', 'patterns': 'pattern',
            'prints': 'print', 'designs': 'design', 'styles': 'style',
            'looks': 'look', 'appearances': 'appearance',
            'busts': 'bust', 'waists': 'waist', 'hips': 'hip',
            'torsos': 'torso', 'chests': 'chest', 'arms': 'arm',
            'legs': 'leg', 'thighs': 'thigh', 'calves': 'calf',
            'gapes': 'gape', 'gaped': 'gape', 'gaping': 'gape',
            'sags': 'sag', 'sagged': 'sag', 'sagging': 'sag',
            'stretches': 'stretch', 'stretched': 'stretch', 'stretching': 'stretch',
            'pills': 'pill', 'pilling': 'pill',
            'looove': 'love', 'loooove': 'love', 'sooo': 'so', 'soooo': 'so',
            'veery': 'very', 'toootally': 'totally',
        }

        self.protected_words = set([
            'xs', 's', 'm', 'l', 'xl', 'xxl', 'xxs',
            'petite', 'regular', 'tall', 'plus',
            'retailer', 'anthro', 'pilcro', 'maeve',
            'midi', 'maxi', 'mini',
        ])

    def stem(self, token):
        if token.startswith('__') or '$' in token or any(c.isdigit() for c in token):
            return token
        if token.lower() in self.protected_words:
            return token
        if token.lower() in self.special_mappings:
            return self.special_mappings[token.lower()]

        word = token.lower()
        for pattern, replacement in self.suffix_rules:
            new_word = re.sub(pattern, replacement, word)
            if new_word != word:
                return new_word
        return word

    def stem_tokens(self, tokens):
        return [self.stem(token) for token in tokens]


# 初始化词干提取器
stemmer = CustomStemmer()

print("=" * 70)
print("STEP 3: 词干提取 (Stemming)")
print("=" * 70)

# 测试
test_words = ['dresses', 'fitted', 'fitting', 'fits', 'sleeves', 'shoulders',
              'wearing', 'wore', 'colors', 'patterns', 'looove', 'recommended',
              'petites', 'gaping', 'stretching', 'quickly', 'bigger', 'biggest']

print("\n词干提取测试:")
print("-" * 40)
for word in test_words:
    stemmed = stemmer.stem(word)
    status = "✓ 变化" if stemmed != word else "= 保持"
    print(f"{word:15} -> {stemmed:15} {status}")

# 应用词干提取
print("\n\n正在应用词干提取...")
df['tokens_step3'] = df['tokens_step2'].apply(stemmer.stem_tokens)

# 统计
all_tokens_step3 = []
for tokens in df['tokens_step3']:
    all_tokens_step3.extend(tokens)
vocab_step3 = set(all_tokens_step3)

# 计算变化
stemming_changes = sum(1 for i, token in enumerate(all_tokens_step2)
                       if i < len(all_tokens_step3) and token != all_tokens_step3[i])

print(f"\n{'=' * 70}")
print("STEP 3 词干提取统计结果")
print("=" * 70)
print(f"词干提取前: 总Token {len(all_tokens_step2):,}, 词汇 {len(vocab_step2):,}")
print(f"词干提取后: 总Token {len(all_tokens_step3):,}, 词汇 {len(vocab_step3):,}")
print(
    f"词汇减少: {len(vocab_step2) - len(vocab_step3)} ({(len(vocab_step2) - len(vocab_step3)) / len(vocab_step2) * 100:.1f}%)")
print(f"唯一词汇数 (Vocabulary Size): {len(vocab_step3):,}")
print(f"\n词干提取后最常见的20个Tokens:")
token_counts_step3 = Counter(all_tokens_step3)
for token, count in token_counts_step3.most_common(20):
    print(f"  {token}: {count}")

# ============================================================
# 生成对比示例和最终报告
# ============================================================

print("=" * 70)
print("完整处理流程对比示例")
print("=" * 70)

for i in range(5):
    original = df.iloc[i]['Review Text']
    step1 = df.iloc[i]['tokens_step1']
    step2 = df.iloc[i]['tokens_step2']
    step3 = df.iloc[i]['tokens_step3']

    print(f"\n{'=' * 70}")
    print(f"示例 {i + 1}")
    print("=" * 70)
    print(f"原文: {original[:200]}...")
    print(f"\nStep 1 - 分词 ({len(step1)} tokens):")
    print(f"  {step1[:20]}{'...' if len(step1) > 20 else ''}")
    print(f"\nStep 2 - 去停用词 ({len(step2)} tokens):")
    print(f"  {step2[:20]}{'...' if len(step2) > 20 else ''}")
    print(f"\nStep 3 - 词干提取 ({len(step3)} tokens):")
    print(f"  {step3[:20]}{'...' if len(step3) > 20 else ''}")

# 生成汇总统计
print(f"\n\n{'=' * 70}")
print("三阶段处理汇总统计")
print("=" * 70)

stats_df = pd.DataFrame({
    '阶段': ['Step 1: 分词', 'Step 2: 去停用词', 'Step 3: 词干提取'],
    '总Token数': [len(all_tokens_step1), len(all_tokens_step2), len(all_tokens_step3)],
    '唯一词汇数': [len(vocab_step1), len(vocab_step2), len(vocab_step3)],
    '平均评论长度': [
        len(all_tokens_step1) / len(df),
        len(all_tokens_step2) / len(df),
        len(all_tokens_step3) / len(df)
    ]
})

print(stats_df.to_string(index=False))

# 计算减少比例
print(f"\n{'=' * 70}")
print("各阶段减少比例")
print("=" * 70)
print(f"Step 1 → Step 2 (停用词过滤):")
print(f"  Token减少: {(len(all_tokens_step1) - len(all_tokens_step2)) / len(all_tokens_step1) * 100:.1f}%")
print(f"  词汇减少: {(len(vocab_step1) - len(vocab_step2)) / len(vocab_step1) * 100:.1f}%")
print(f"\nStep 2 → Step 3 (词干提取):")
print(f"  词汇减少: {(len(vocab_step2) - len(vocab_step3)) / len(vocab_step2) * 100:.1f}%")
print(f"\n总体 (Step 1 → Step 3):")
print(f"  Token减少: {(len(all_tokens_step1) - len(all_tokens_step3)) / len(all_tokens_step1) * 100:.1f}%")
print(f"  词汇减少: {(len(vocab_step1) - len(vocab_step3)) / len(vocab_step1) * 100:.1f}%")

# ============================================================
# 生成最终Excel文件
# ============================================================

# 创建处理后的文本列
df['processed_text_step1'] = df['tokens_step1'].apply(lambda x: ' '.join(x) if x else '')
df['processed_text_step2'] = df['tokens_step2'].apply(lambda x: ' '.join(x) if x else '')
df['processed_text_step3'] = df['tokens_step3'].apply(lambda x: ' '.join(x) if x else '')

# 添加处理统计信息
df['token_count_step1'] = df['tokens_step1'].apply(len)
df['token_count_step2'] = df['tokens_step2'].apply(len)
df['token_count_step3'] = df['tokens_step3'].apply(len)

# 选择输出列
output_columns = [
    'Unnamed: 0', 'Clothing ID', 'Age', 'Title', 'Review Text', 'Rating',
    'Recommended IND', 'Positive Feedback Count', 'Division Name',
    'Department Name', 'Class Name',
    'tokens_step1', 'processed_text_step1', 'token_count_step1',
    'tokens_step2', 'processed_text_step2', 'token_count_step2',
    'tokens_step3', 'processed_text_step3', 'token_count_step3'
]

# 创建输出DataFrame
output_df = df[output_columns].copy()

# 重命名列使其更清晰
output_df.columns = [
    'Index', 'Clothing_ID', 'Age', 'Title', 'Original_Review', 'Rating',
    'Recommended', 'Positive_Feedback_Count', 'Division_Name',
    'Department_Name', 'Class_Name',
    'Tokens_Step1_Raw', 'Processed_Text_Step1', 'Token_Count_Step1',
    'Tokens_Step2_NoStopwords', 'Processed_Text_Step2', 'Token_Count_Step2',
    'Tokens_Step3_Stemmed', 'Processed_Text_Step3', 'Token_Count_Step3'
]

# 保存为Excel
output_path = '//Users//liduo//Desktop//Womens_Clothing_Reviews_Processed.xlsx'
output_df.to_excel(output_path, index=False, engine='openpyxl')

print("=" * 70)
print("Excel文件生成完成!")
print("=" * 70)
print(f"文件路径: {output_path}")
print(f"总行数: {len(output_df):,}")
print(f"总列数: {len(output_df.columns)}")

#print(f"\n列名列表:")
#for i, col in enumerate(output_df.columns, 1):
    #print(f"  {i}. {col}")

#print(f"\n前3行预览 (关键列):")
#preview_cols = ['Index', 'Original_Review', 'Processed_Text_Step1',
                #'Processed_Text_Step2', 'Processed_Text_Step3']
#print(output_df[preview_cols].head(3).to_string())
