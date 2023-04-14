import torch.nn as nn
import torch.nn.functional as F
import torch
from accelerate.logging import get_logger
logger = get_logger(__name__)

"""Vector Quantizer Layer"""
class VectorQuantizer(nn.Module):
    """
    This layer takes a tensor to be quantized. The channel dimension will be used as the space
    in which to quantize. All other dimensions will be flattened and will be seen as different
    examples to quantize.

    The output tensor will have the same shape as the input.
    As an example for a `BCHW` tensor of shape `[16, 64, 32, 32]`,
    we will first convert it to an `BHWC` tensor of shape `[16, 32, 32, 64]`
    and then reshape it into `[16384, 64]` and all `16384` vectors of size `64`  will be quantized independently.
    In other words, the channels are used as the space in which to quantize.
    All other dimensions will be flattened and be seen as different examples to quantize, `16384` in this case.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        # code book: [num_emb, emb_dim]
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        # uniform init weight
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from [B,C,H,W] -> [B,H,W,C]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input => [BHW, C]
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances between flatten input: [BHW, C] and embedding: [num_emb, emb_dim]
        # quantize for each vector:[1, C] and emb:[1, emb_dim], quantize BHW times in total
        # dist: [BHW, num_emb]=[16384, 64]
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        logger.debug(f"dist:{distances.shape}")
        """Encoding"""
        # encoding_indices: [16384, 1]
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        logger.debug(f"encoding_dist:{encoding_indices.shape}")
        # encodings: [16384,512]
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        logger.debug(f"encoding:{encodings.shape}")
        # scatter: [16384, 512]
        encodings.scatter_(1, encoding_indices, 1)
        logger.debug(f"encoding_scatter:{encodings.shape}")
        # Quantize and unflatten
        # quantized: [256,8,8,64]
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        logger.debug(f"quantized:{quantized.shape}")
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        # [512]
        avg_probs = torch.mean(encodings, dim=0)
        logger.debug(f"avg_probs:{avg_probs.shape}")
        # Tensor.float
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        logger.debug(f"perplexity:{perplexity}")
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

    @property
    def embedding(self):
        return self._embedding

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from [B,C,H,W] -> [B,H,W,C]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        # [BHW, C]
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        perplexity.requires_grad = False
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

    @property
    def embedding(self):
        return self._embedding




