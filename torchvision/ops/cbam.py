from torch import nn, Tensor, cat
import torch

class ConvolutionalBlockAttentionModule(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) from 'CBAM: Convolutional Block Attention Module,'
    Woo et al. (https://arxiv.org/abs/1807.06521).

    Args:
        in_channels (int): Number of channels of the input tensor
        reduction_ratio (int): Reduction ratio for the squeeze step. Default: 16
        act_layer (nn.Module): Activation layer. Default: nn.ReLU
        gate_layer (nn.Module): Gate layer. Default: nn.Sigmoid

    Examples::
        >>> m = torchvision.ops.ConvolutionalBlockAttentionModule(10, 1)
        >>> input = torch.rand(1, 10, 16, 16)
        >>> output = m(input)
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        act_layer: nn.Module = nn.ReLU,
        gate_layer: nn.Module = nn.Sigmoid
    ) -> None:
        super(ConvolutionalBlockAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # channel average pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1) # channel max pooling
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False), # W0
            act_layer(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False), # W1
        )
        self.gate_layer = gate_layer()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x: Tensor) -> Tensor:

        b, c, _, _ = x.size()

        max = self.max_pool(x).view(b, c) # max pooling
        avg = self.avg_pool(x).view(b, c) # average pooling

        # channel attention
        max = self.fc(max).view(b, c, 1, 1) # apply fully connected layer (W0, W1)
        avg = self.fc(avg).view(b, c, 1, 1) # apply fully connected layer (W0, W1)
        mc = self.gate_layer(max + avg)

        # apply channel attention to the input tensor
        x1 = x * mc.expand_as(x)

        # spatial attention
        max_s, _ = torch.max(x1, dim=1, keepdim=True)
        avg_s = torch.mean(x1, dim=1, keepdim=True)

        pool_s = cat([max_s, avg_s], dim=1) # concatenate the two tensors
        ms = self.gate_layer(self.conv(pool_s)) # apply convolutional layer

        # apply spatial attention
        return x1 * ms