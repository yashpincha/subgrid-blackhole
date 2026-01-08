import torch
from torch.utils.data import DataLoader
from data.dataset import SphericalDataset
import torch.nn as nn
import argparse
from sfno import SphericalFourierNeuralOperator as SFNO

class Trainer:
    def __init__(self, batch_size=2, loss_fn = nn.MSELoss(), shuffle=True):
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.train_dataset = SphericalDataset('/pscratch/sd/y/ypincha/', 'train', train_ratio = 0.8)
        self.val_dataset = SphericalDataset('/pscratch/sd/y/ypincha/', 'val', train_ratio = 0.8)
        self.train_data = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle, pin_memory=True)
        self.val_data = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        self.means = torch.load('/global/homes/y/ypincha/blackholes/subgrid-blackhole/data/stats.pt')['mean'].reshape(8, 1, 1, 1)
        self.vars = torch.load('/global/homes/y/ypincha/blackholes/subgrid-blackhole/data/stats.pt')['var'].reshape(8, 1, 1, 1)

    def normalize_data(self, batch):
        u_n, u_np1 = batch # each [B, C, R, H, W]
        
        u_n = (u_n - self.means.to(u_n.device)) / torch.sqrt(self.vars.to(u_n.device))
        u_np1 = (u_np1 - self.means.to(u_np1.device)) / torch.sqrt(self.vars.to(u_np1.device))
        
        print('post normalization', u_n.shape, u_n.mean())
        channel_means = u_n.mean(dim=(0, 2, 3, 4))# mean over B, R, H, W
        print(channel_means)

        return u_n, u_np1


    def train_epoch(self, model, optimizer):
        model.train()
        total_loss = 0.0

        for batch in self.train_data:
            u_n, u_np1 = self.normalize_data(batch)

            B, C, R, H, W = u_n.shape

            # flatten radius into channels: [B, C, R, H, W] -> [B, C*R, H, W]
            u_n = u_n.reshape(B, C*R, H, W)
            u_np1 = u_np1.reshape(B, C*R, H, W)

            device = next(model.parameters()).device
            u_n = u_n.to(device)
            u_np1 = u_np1.to(device)

            optimizer.zero_grad()
            output = model(u_n) # [B, C*R, H, W]
            loss = self.compute_loss(output, u_np1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss/len(self.train_data)
        return avg_loss
    
    def validate_epoch(self, model):
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_data:
                u_n, u_np1 = self.normalize_data(batch)
                B, C, R, H, W = u_n.shape

                # flatten radius into channels: [B, C, R, H, W] -> [B, C*R, H, W]
                u_n = u_n.reshape(B, C*R, H, W)
                u_np1 = u_np1.reshape(B, C*R, H, W)

                device = next(model.parameters()).device
                u_n = u_n.to(device)
                u_np1 = u_np1.to(device)

                output = model(u_n)
                loss = self.compute_loss(output, u_np1)
                total_loss += loss.item()
        avg_loss = total_loss/len(self.val_data)
        return avg_loss

    def compute_loss(self, output, target):
        loss_val = self.loss_fn(output, target)
        return loss_val
    
def train(args):
    # C=8 channels, R=64 radial shells --> C*R=512 channels
    model = SFNO(img_size=(64, 64), pos_embed='spectral', scale_factor = 4, in_chans=512, out_chans=512, embed_dim=64).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, min_lr=1e-6)
    trainer = Trainer(batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(model, optimizer)
        val_loss = trainer.validate_epoch(model)
        scheduler.step(val_loss)
        print(f"epoch {epoch+1}/{args.epochs}, train Loss: {train_loss:.6f}, val Loss: {val_loss:.6f}")

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument(
    '--device',
    type=str,
    default='cuda' if torch.cuda.is_available() else 'cpu'
)

args = parser.parse_args()

train(args)
