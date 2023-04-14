from torch import nn

from models.VectorQuantizer import VectorQuantizerEMA, VectorQuantizer
from models.Decoder import Decoder
from models.Encoder import Encoder
from accelerate.logging import get_logger
logger = get_logger(__name__)

class VQVAEModel(nn.Module):
    def __init__(self, num_hiddens=128, num_residual_layers=2,
                 num_residual_hiddens=32,num_embeddings=512,
                 embedding_dim=64, commitment_cost=0.25, decay=0):
        super(VQVAEModel, self).__init__()

        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

    def forward(self, x):  # input: [256,3,32,32]
        logger.debug(f"input:{x.shape}")
        z = self._encoder(x)  # encoder: [256,128,8,8]
        logger.debug(f"encoder:{z.shape}")
        # pre_vq_conv: used to
        z = self._pre_vq_conv(z)   # pre_vq_conv: [256,64,8,8]
        logger.debug(f"pre_vq_conv:{z.shape}")
        loss, quantized, perplexity, _ = self._vq_vae(z)  # quantized: [256,64,8,8]
        logger.debug(f"vq_vae:{quantized.shape}")
        x_recon = self._decoder(quantized)  # [256,3,32,32]
        logger.debug(f"decoder:{x_recon.shape}")
        return loss, x_recon, perplexity

    @property
    def pre_vq_conv(self):
        return self._pre_vq_conv

    @property
    def decoder(self):
        return self._decoder

    @property
    def vq_vae(self):
        return self._vq_vae

    @property
    def encoder(self):
        return self._encoder
