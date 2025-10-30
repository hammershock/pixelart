import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ======================
# 1. 读取 CSV 文件
# ======================
csv_path = "pixel_dataset.csv"
df = pd.read_csv(csv_path)

print(f"共有 {len(df)} 条数据")
print(df.head())

# ======================
# 2. 分词函数
# ======================
def tokenize(text):
    """
    简单英文分词：
    - 全部转小写
    - 去掉标点
    - 按空格拆分
    """
    text = text.lower()
    # 仅保留字母和空格
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    return tokens

# ======================
# 3. 分词 + 统计词频
# ======================
all_tokens = []
for caption in df["caption"]:
    tokens = tokenize(caption)
    all_tokens.extend(tokens)

# 构建词表
vocab = sorted(set(all_tokens))
print(f"词表大小: {len(vocab)}")
print(vocab)
# # 统计词频
# word_counts = Counter(all_tokens)
# print("Top 10 最常见词:")
# for w, c in word_counts.most_common(10):
#     print(f"{w}: {c}")

# # ======================
# # 4. 词频可视化（柱状图）
# # ======================
# top_n = 15
# top_words = dict(word_counts.most_common(top_n))

# plt.figure(figsize=(10, 5))
# plt.bar(top_words.keys(), top_words.values())
# plt.title(f"Top {top_n} Frequent Words in Captions")
# plt.xlabel("Words")
# plt.ylabel("Frequency")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # ======================
# # 5. 词云可视化
# # ======================
# wc = WordCloud(
#     width=800,
#     height=400,
#     background_color="white",
#     max_words=100,
#     colormap="viridis"
# ).generate_from_frequencies(word_counts)

# plt.figure(figsize=(10, 6))
# plt.imshow(wc, interpolation="bilinear")
# plt.axis("off")
# plt.title("Word Cloud of Caption Words")
# plt.show()
