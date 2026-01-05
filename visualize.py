import matplotlib.pyplot as plt
import numpy as np
import math
import cartopy.crs as ccrs
from data.dataset import SphericalDataset
import matplotlib.pyplot as plt

FIELDS = [
    'bcc1', 'bcc2', 'bcc3', 'dens',
    'eint', 'velx', 'vely', 'velz'
]

def plot_sphere(theta, phi, data_3d, channel_names,
                r0=32, cmap='twilight_shifted',
                vmin=None, vmax=None):

    lon = phi.numpy()*180/math.pi
    lat = theta.numpy() * 180/math.pi - 90.0
    Lon, Lat = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(20, 10))
    proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=20.0)

    for c in range(data_3d.shape[0]):
        ax = fig.add_subplot(2, 4, c+1, projection=proj)

        field = data_3d[c, r0].numpy()

        im = ax.pcolormesh(Lon, Lat, field,
            transform=ccrs.PlateCarree(), cmap=cmap,
            vmin=vmin,vmax=vmax, shading='auto'
        )

        ax.set_global()
        ax.set_title(channel_names[c], fontsize=12)

        cbar = fig.colorbar(im, ax=ax,orientation='horizontal',
                    fraction=0.05, pad=0.05)

        
        cbar.set_label(f'{channel_names[c]} at r={r0}')

    plt.savefig('all_fields.png', dpi=300)

if __name__ == '__main__':
    dataset = SphericalDataset('/pscratch/sd/y/ypincha/', 'train', train_ratio=1.0)
    r, theta, phi = dataset.get_coords()
    sample, sample_p1 = dataset.__getitem__(0)
    # print(type(sample))
    # print(sample, sample.shape)

    plot_sphere(theta,phi,sample,FIELDS, r0=0)
