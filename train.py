from __future__ import print_function

import os
import shutil
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader

from utils.load_data import load_data
from models.VQVAEModel import VQVAEModel
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from six.moves import xrange

import umap

import torch.nn.functional as F
import torch.optim as optim

from torchvision.utils import make_grid

from utils.utils import show

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import logging
from accelerate.logging import get_logger
from tqdm import tqdm

logger = get_logger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

batch_size = 256
num_training_updates = 15000

eval_step = 200

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3

@record
def train():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    device = accelerator.device

    training_loader, validation_loader, data_variance = load_data(batch_size=batch_size)

    model = VQVAEModel(num_hiddens, num_residual_layers, num_residual_hiddens,
                       num_embeddings, embedding_dim,
                       commitment_cost, decay).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    # load stats
    best_val_loss = float('inf')
    if os.path.exists("./ckp/"):
        file_name = os.listdir("./ckp/")[0]
        best_val_loss = float(file_name.split("_")[-1])
        accelerator.load_state(f"./ckp/check_point_{best_val_loss:.2f}")
    logger.info(f"current best_val_loss: {best_val_loss}")

    model, optimizer, training_loader, validation_loader \
        = accelerator.prepare(model, optimizer, training_loader, validation_loader)
    training_loader: DataLoader
    model.train()
    train_res_recon_error = []
    train_res_perplexity = []
    progress_bar = tqdm(xrange(num_training_updates), total=num_training_updates)
    for i in progress_bar:
        (data, _) = next(iter(training_loader))
        data = data.to(device)
        optimizer.zero_grad()
        logger.debug(f"train data type: {type(data)}")  # torch.Tensor
        logger.debug(f"data:{data.shape}")  # [256,3,32,32]
        vq_loss, data_recon, perplexity = model(data)
        logger.debug(f"vq_loss:{vq_loss.shape}, data_recon:{data_recon.shape}, per:{perplexity.shape}")
        logger.debug(f"device: data_recon:{data_recon.device}, "
                     f"data:{data.device}")

        recon_error = F.mse_loss(data_recon, data) / data_variance
        loss = recon_error + vq_loss
        accelerator.backward(loss)
        optimizer.step()

        progress_bar.set_postfix_str(f"loss:{loss:.2f}")

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if (i + 1) % eval_step == 0:
            logger.info(f'{i + 1} iterations')
            logger.info(f'recon_error: {np.mean(train_res_recon_error[-100:]):.3f}')
            logger.info(f'perplexity: {np.mean(train_res_perplexity[-100:]):.3f}\n')
            # val and save_model
            val_loss = valid(validation_loader, data_variance, model)
            logger.info(f"valid loss: {val_loss}")
            if val_loss < best_val_loss and accelerator.is_main_process:
                if not os.path.exists("./ckp/"):
                    os.makedirs("./ckp/")
                else:
                    shutil.rmtree(f"./ckp/check_point_{best_val_loss:.2f}")
                best_val_loss = val_loss
                accelerator.save_state(f"./ckp/check_point_{best_val_loss:.2f}")

    # Plot Loss
    train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
    train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)

    f = plt.figure(figsize=(16, 8))
    ax = f.add_subplot(1, 2, 1)
    ax.plot(train_res_recon_error_smooth)
    ax.set_yscale('log')
    ax.set_title('Smoothed NMSE.')
    ax.set_xlabel('iteration')

    ax = f.add_subplot(1, 2, 2)
    ax.plot(train_res_perplexity_smooth)
    ax.set_title('Smoothed Average codebook usage (perplexity).')
    ax.set_xlabel('iteration')
    f.savefig("./img/loss.png")

    """validation"""
    model.eval()

    (valid_originals, _) = next(iter(validation_loader))
    valid_originals = valid_originals  # .to(device)

    vq_output_eval = model.module.pre_vq_conv(model.module.encoder(valid_originals))
    _, valid_quantize, _, _ = model.module.vq_vae(vq_output_eval)
    valid_reconstructions = model.module.decoder(valid_quantize)

    (train_originals, _) = next(iter(training_loader))
    train_originals = train_originals  # .to(device)
    _, train_reconstructions, _, _ = model.module.vq_vae(train_originals)

    show(make_grid(valid_reconstructions.cpu().data) + 0.5, )

    show(make_grid(valid_originals.cpu() + 0.5))

    # View Embedding
    proj = umap.UMAP(n_neighbors=3,
                     min_dist=0.1,
                     metric='cosine').fit_transform(model.module.vq_vae.embedding.weight.data.cpu())

    plt.scatter(proj[:, 0], proj[:, 1], alpha=0.3)
    plt.savefig("./img/embedding.png")

def valid(validation_loader, data_variance, model):
    val_loss = 0
    for idx, batch in enumerate(validation_loader):
        # batch: [32,3,32,32], [32]
        vq_loss, data_recon, _ = model(batch[0])
        recon_error = F.mse_loss(data_recon, batch[0]) / data_variance
        val_loss += vq_loss + recon_error
    val_loss /= len(validation_loader)
    return val_loss

if __name__ == '__main__':
    train()
    """
    accelerate launch --config_file ./vqvae_config.yaml --main_process_port 25910 train.py
    """