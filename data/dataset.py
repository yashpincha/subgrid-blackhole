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
    
    def transform_to_spherical(self, data):

        # TO_CONVERT = ['bcc1', 'bcc2', 'bcc3', 'velx', 'vely', 'velz'] # data has size [C, R, H, W] 
        # # we want to ingest a list [bcc1, bcc2, bcc3] --> [bcc_r, bcc_theta, bcc_phi] 
        # # similarly, [velx, vely, velz] --> [vel_r, vel_theta, vel_phi] 
        # # __getitem__ returns arrays of size [C, R, H, W] for u_n
        # each field is now size of [R, H, W]

        theta, phi = self.get_coords()[1:] # each of size [H, W]
        BCC_IDXS = (0, 1, 2)
        VEL_IDXS = (5, 6, 7)

        def convert_block(field_x, field_y, field_z):
            field_r = (field_x*torch.sin(theta)*torch.cos(phi) +
                       field_y*torch.sin(theta)*torch.sin(phi) +
                       field_z*torch.cos(theta))

            field_theta = (field_x*torch.cos(theta)*torch.cos(phi) +
                           field_y*torch.cos(theta)*torch.sin(phi) -
                           field_z*torch.sin(theta))

            field_phi = (-field_x*torch.sin(phi) +
                         field_y*torch.cos(phi))
            
            return field_r, field_theta, field_phi

        spherical_fields = []

        bcc_r, bcc_theta, bcc_phi = convert_block(data[BCC_IDXS[0]], data[BCC_IDXS[1]], data[BCC_IDXS[2]])
        spherical_fields.append(torch.stack([bcc_r, bcc_theta, bcc_phi], dim=0))
        spherical_fields.append(data[3:5])  # dens, eint remain unchanged
        vel_r, vel_theta, vel_phi = convert_block(data[VEL_IDXS[0]], data[VEL_IDXS[1]], data[VEL_IDXS[2]])
        spherical_fields.append(torch.stack([vel_r, vel_theta, vel_phi], dim=0))

        return spherical_fields

    def stack_tensor(self, fname):
        with h5py.File(fname, 'r') as f:
            arrays = [
                torch.tensor(f[field][:], dtype=torch.float32)
                for field in self.FIELDS
            ]
        return torch.stack(arrays)

    def __getitem__(self, idx):
        u_n, u_np1 = self.stack_tensor(self.files[idx]), self.stack_tensor(self.files[idx+1])
        return self.transform_to_spherical(u_n), self.transform_to_spherical(u_np1)

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
