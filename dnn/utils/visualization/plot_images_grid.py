import torch
import matplotlib
matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'

import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def plot_images_grid(x: torch.tensor, export_img, title: str = '', nrow=8, padding=2, normalize=False, pad_value=0):
    """Plot 4D Tensor of images of shape (B x C x H x W) as a grid."""

    grid = make_grid(x, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value)
    npgrid = grid.cpu().numpy()

    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')

    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if not (title == ''):
        plt.title(title)

    plt.savefig(export_img, bbox_inches='tight', pad_inches=0.1)
    plt.clf()

def plot_signal_list(x: list, export_img, title: str = '', nrow=8, ncol=4, padding=2, normalize=False, pad_value=0):
    """Plot a list of 2D signal (H x W) as a grid."""

    assert len(x) == nrow * ncol
    # print(len(x))
    fig, axes = plt.subplots(nrow, ncol)
    # print(x[0])
    for i in range(nrow):
        for j in range(ncol):
            # print(f'i j {i}{j}')
            # print(f'[i*nrow+j] {i*ncol+j}')
            axes[i, j].plot(x[i*ncol+j][0])

    if not (title == ''):
        fig.title(title)

    fig.savefig(export_img, bbox_inches='tight', pad_inches=0.1)
    fig.clf()