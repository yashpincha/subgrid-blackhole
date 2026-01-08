import torch
from torch.utils.data import DataLoader
from data.dataset import SphericalDataset
import torch.nn as nn
import argparse
from sfno import SphericalFourierNeuralOperator as SFNO
from visualize import plot_predictions, FIELDS

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

        # Get coordinates for plotting
        coords = self.val_dataset.get_coords()
        self.r_array = coords[0]
        self.theta = coords[1]
        self.phi = coords[2]

    def normalize_data(self, batch):
        u_n, u_np1 = batch # each [B, C, R, H, W]
        u_n = (u_n - self.means) / torch.sqrt(self.vars)
        u_np1 = (u_np1 - self.means) / torch.sqrt(self.vars)
        
        # print('post normalization', u_n.shape, u_n.mean())
        channel_means = u_n.mean(dim=(0, 2, 3, 4))# mean over B, R, H, W
        # print(channel_means)

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
        print('avg_loss is', avg_loss)
        return avg_loss
    
    def validate_epoch(self, model, epoch=0, plot=False, plot_dir='plots'):
        model.eval()
        total_loss = 0.0

        first_batch_pred = None
        first_batch_gt = None
        original_shape = None

        with torch.no_grad():
            for i, batch in enumerate(self.val_data):
                u_n, u_np1 = self.normalize_data(batch)
                B, C, R, H, W = u_n.shape

                if i == 0 and plot:
                    original_shape = (C, R, H, W)
                # flatten radius into channels: [B, C, R, H, W] -> [B, C*R, H, W]
                u_n = u_n.reshape(B, C*R, H, W)
                u_np1 = u_np1.reshape(B, C*R, H, W)

                device = next(model.parameters()).device
                u_n = u_n.to(device)
                u_np1 = u_np1.to(device)

                output = model(u_n)
                loss = self.compute_loss(output, u_np1)
                total_loss += loss.item()

                if i == 0 and plot:
                    first_batch_pred = output[0].cpu()
                    first_batch_gt = u_np1[0].cpu()

        avg_loss = total_loss/len(self.val_data)

        if plot and first_batch_pred is not None:
            C, R, H, W = original_shape
            pred_reshaped = first_batch_pred.reshape(C, R, H, W)
            gt_reshaped = first_batch_gt.reshape(C, R, H, W)

            plot_predictions(
                self.theta, self.phi, pred_reshaped, gt_reshaped,
                self.r_array, FIELDS, epoch, output_dir=plot_dir
            )

        return avg_loss

    def compute_loss(self, output, target):
        loss_val = self.loss_fn(output, target)
        return loss_val
    
def train(args):
    # C=8 channels, R=64 radial shells --> C*R=512 channels
    model = SFNO(img_size=(64, 64), pos_embed='spectral', scale_factor = 10, in_chans=512, out_chans=512, embed_dim=32).to(args.device)
    print('param count is', sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, min_lr=1e-6)
    trainer = Trainer(batch_size=args.batch_size, shuffle=True)
    print('starting training')
    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(model, optimizer)

        should_plot = (epoch % args.plot_freq == 0) or (epoch == 0) or (epoch == args.epochs - 1)
        val_loss = trainer.validate_epoch(model, epoch=epoch, plot=should_plot, plot_dir=args.plot_dir)

        scheduler.step(val_loss)
        print(f"epoch {epoch+1}/{args.epochs}, train Loss: {train_loss:.6f}, val Loss: {val_loss:.6f}")

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--plot_freq', type=int, default=10)
parser.add_argument('--plot_dir', type=str, default='plots')
parser.add_argument(
    '--device',
    type=str,
    default='cuda' if torch.cuda.is_available() else 'cpu'
)

args = parser.parse_args()

train(args)
