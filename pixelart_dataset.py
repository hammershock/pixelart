# pixelart_dataset.py
from torch.utils.data import Dataset
import numpy as np

class Tokenizer:
    def __init__(self, json_path):
        # load json from path
        # {"word1": 0, "word2": 1, ...}
        self.vocab = ...
        # len(self.vocab) = 270
        
    def tokenize(self, text):
        text = re.sub(r"[^a-z\s]", "", text)
        tokens = text.split()
        # bag of words encode
        vec = np.zeros(len(self.vocab))
        for token in tokens:
            if token in self.vocab:
                idx = self.vocab[token]
                vec[idx] = 1
        return vec


class PixelArtDataset(Dataset):
    def __init__(self, csv_path, image_folder, tokenizer):
        # load csv
        # image,caption
        # image_0.JPEG,"A pixel art of a green alien with a white face and a black background, 16-bit style"
        # image_1.JPEG,"A pixel art of a green alien with a black background, 16-bit style"
        # ... 
        pass
    
        # load images
        # image_folder/image_0.JPEG...
        pass
    
    def __getitem__(self):
        label = ...  # bag of words vector
        image = ...
        return label, image  # (torch.tensor)
    
    def __len__(self):
        return ...