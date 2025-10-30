# train.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from pixelart_dataset import Tokenizer, PixelArtDataset
import os

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = encoded.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

def loss_function(reconstructed, original, mu, logvar):
    recon_loss = nn.functional.mse_loss(reconstructed, original, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

if __name__ == "__main__":
    tokenizer = Tokenizer("./vocab.json")
    dataset = PixelArtDataset("./pixel_dataset.csv", "/cache/hanmo/pixelart/images/images", tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = 270
    latent_dim = 64
    model = VAE(input_dim, latent_dim)

    checkpoint_path = "./models/last.pt"
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Checkpoint loaded.")

    # Training loop
    for epoch in range(10):  # Number of epochs
        model.train()
        total_loss = 0
        for labels, images in dataloader:
            if torch.cuda.is_available():
                labels, images = labels.cuda(), images.cuda()

            optimizer.zero_grad()
            reconstructed, mu, logvar = model(labels)
            loss = loss_function(reconstructed, labels, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

        # Save checkpoint
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        print("Checkpoint saved.")