# pixelart_dataset.py
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
import re
import json

class Tokenizer:
    def __init__(self, json_path):
        # Load vocabulary from JSON file
        with open(json_path, 'r') as f:
            self.vocab = json.load(f)

    def tokenize(self, text):
        text = re.sub(r"[^a-z\s]", "", text.lower())
        tokens = text.split()
        # Bag of words encoding
        vec = np.zeros(len(self.vocab))
        for token in tokens:
            if token in self.vocab:
                idx = self.vocab[token]
                vec[idx] = 1
        return vec

class PixelArtDataset(Dataset):
    def __init__(self, csv_path, image_folder, tokenizer):
        # Load CSV file
        self.data = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Get image and caption
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_folder, row['image'])
        caption = row['caption']

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Tokenize caption
        label = torch.tensor(self.tokenizer.tokenize(caption), dtype=torch.float32)

        return label, image

    def __len__(self):
        return len(self.data)
    
if __name__ == "__main__":
    tokenizer = Tokenizer("./vocab.json")
    dataset = PixelArtDataset("./pixel_dataset.csv", "./images", tokenizer=tokenizer)
    print(dataset[0])