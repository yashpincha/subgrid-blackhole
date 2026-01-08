from torch.utils.data import Dataset
import glob
import torch
import h5py

class SphericalDataset(Dataset):
    FIELDS = ['bcc1', 'bcc2', 'bcc3', 'dens', 'eint', 'velx', 'vely', 'velz']
    coords = ['r', 'theta', 'phi']
    def __init__(self, pscratch_path, mode='train', train_ratio=0.8, normalize=True):

        self.pscratch_path = pscratch_path
        self.mode = mode
        self.normalize = normalize
        self.train_ratio = train_ratio
        pattern = f"{self.pscratch_path}/mhd/3d-h5-sphere/*.h5"
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no .h5 files found under {pattern}")

        split_idx = int(len(files) * train_ratio)
        if mode == 'train':
            self.files = files[:split_idx]
        else:
            self.files = files[split_idx:]

    def sorted_files(self):
        return self.files

    def compute_statistics(self):
        num_files = len(self.files)
        num_channels = len(self.FIELDS)
        means = torch.zeros(num_files, num_channels)
        variances = torch.zeros(num_files, num_channels)

        for i, fname in enumerate(self.files):
            with h5py.File(fname, 'r') as f:
                arrays = torch.stack([torch.tensor(f[field][:], dtype=torch.float32)
                                      for field in self.FIELDS]).view(num_channels, -1)
                means[i] = arrays.mean(dim=1)
                variances[i] = arrays.var(dim=1, unbiased=False)

        return means, variances

    def global_stats(self):
        means, variances = self.compute_statistics()
        global_mean = means.mean(dim=0)
        global_var = (variances + (means - global_mean)**2).mean(dim=0)
        # print(global_mean.shape, global_var.shape)
        return global_mean, global_var

    def __len__(self):
        return len(self.files)-1

    def stack_tensor(self, fname):
        with h5py.File(fname, 'r') as f:
            arrays = [
                torch.tensor(f[field][:], dtype=torch.float32)
                for field in self.FIELDS
            ]
        return torch.stack(arrays)

    def __getitem__(self, idx):
        return self.stack_tensor(self.files[idx]), self.stack_tensor(self.files[idx+1])
    
    def get_coords(self):
        with h5py.File(self.files[0], 'r') as f:
            coords_array = torch.stack([torch.tensor(f[coord][:], dtype=torch.float32)
                                      for coord in self.coords])

        return coords_array

if __name__ == '__main__':
    dataset = SphericalDataset('/pscratch/sd/y/ypincha/', 'train', train_ratio = 1.0)
    # global_mean, global_var = dataset.global_stats()
    # stats_dict = {'mean': global_mean, 'var': global_var}
    # torch.save(stats_dict, f'stats.pt')
