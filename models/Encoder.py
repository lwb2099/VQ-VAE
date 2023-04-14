from torch import nn
import torch.nn.functional as F

from models.Residual import ResidualStack
from accelerate.logging import get_logger
logger = get_logger(__name__)

class Encoder(nn.Module):
    def __init__(self, in_channels=3, num_hiddens=128,
                 num_residual_layers=2, num_residual_hiddens=64):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):  # [256,3,32,32]
        logger.debug(f"input:{inputs.shape}")
        x = self._conv_1(inputs)  # [256,64,16,16]
        logger.debug(f"conv_1:{x.shape}")
        x = F.relu(x)

        x = self._conv_2(x)  # [256,128,8,8]
        logger.debug(f"conv_2:{x.shape}")
        x = F.relu(x)

        x = self._conv_3(x)  # [256,128,8,8]
        logger.debug(f"conv_3:{x.shape}")
        x = self._residual_stack(x)  # [256,128,8,8]
        logger.debug(f"residual:{x.shape}")
        return x
