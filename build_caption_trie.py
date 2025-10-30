# build_caption_trie.py
# ===============================
# 构建和保存 Pixel Caption 前缀树
# ===============================
import os
import re
import csv
import json
from typing import Dict, List, Optional

class TrieNode:
    """前缀树节点"""
    __slots__ = ["children", "is_end"]

    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.is_end: bool = False


class CaptionTrie:
    """
    基于caption词序列的前缀树：
      - insert(tokens): 插入一个词序列
      - next_tokens(prefix_tokens): 给定前缀返回下一个可能词集合
      - save/load: 序列化到JSON
    """
    def __init__(self):
        self.root = TrieNode()

    # -----------------------------
    # 插入一条caption（分词后）
    # -----------------------------
    def insert(self, tokens: List[str]):
        node = self.root
        for tok in tokens:
            if tok not in node.children:
                node.children[tok] = TrieNode()
            node = node.children[tok]
        node.is_end = True

    # -----------------------------
    # 查询：给定前缀，返回可能的下一个词集合
    # -----------------------------
    def next_tokens(self, prefix_tokens: List[str]) -> List[str]:
        node = self.root
        for tok in prefix_tokens:
            if tok not in node.children:
                return []  # 前缀不存在
            node = node.children[tok]
        return list(node.children.keys())

    # -----------------------------
    # 保存到JSON文件
    # -----------------------------
    def save(self, json_path: str):
        def serialize(node: TrieNode):
            # 为避免JSON过大，这里只存 children的键
            obj = {"end": node.is_end, "children": {}}
            for k, v in node.children.items():
                obj["children"][k] = serialize(v)
            return obj

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serialize(self.root), f, ensure_ascii=False, indent=2)
        print(f"[✔] Trie saved to {json_path}")

    # -----------------------------
    # 从JSON加载
    # -----------------------------
    @classmethod
    def load(cls, json_path: str) -> "CaptionTrie":
        def deserialize(data):
            node = TrieNode()
            node.is_end = data.get("end", False)
            for k, v in data.get("children", {}).items():
                node.children[k] = deserialize(v)
            return node

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        trie = cls()
        trie.root = deserialize(data)
        print(f"[✔] Trie loaded from {json_path}")
        return trie


# -----------------------------
# 构建 Trie 的辅助函数
# -----------------------------
def build_trie_from_csv(csv_path: str) -> CaptionTrie:
    """
    读取CSV文件（image, caption），对caption分词后构建前缀树。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    trie = CaptionTrie()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            caption = row["caption"]
            # 与 Tokenizer 逻辑一致
            text = re.sub(r"[^a-z\s]", "", caption.lower())
            tokens = text.split()
            if tokens:
                trie.insert(tokens)
    print("[✔] Trie built from captions.")
    return trie


# -----------------------------
# 简单演示
# -----------------------------
if __name__ == "__main__":
    csv_path = "./pixel_dataset.csv"
    save_path = "./caption_trie.json"

    if not os.path.exists(save_path):
        print(f"Building trie from {csv_path}...")
        trie = build_trie_from_csv(csv_path)
        trie.save(save_path)
    else:
        print(f"Loading existing trie from {save_path}...")
        trie = CaptionTrie.load(save_path)

    # 测试查询
    while True:
        try:
            prefix = input("\nEnter prefix words (space-separated, or blank to exit): ").strip()
            if not prefix:
                break
            prefix_tokens = prefix.split()
            next_words = trie.next_tokens(prefix_tokens)
            print(f"Next possible words: {next_words if next_words else '(no match)'}")
        except KeyboardInterrupt:
            break
