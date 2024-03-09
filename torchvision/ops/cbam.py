from torch import nn, Tensor, cat

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
        gate_layer: nn.Module = nn.Sigmoid,
    ) -> None:
        super(ConvolutionalBlockAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=True),
            act_layer(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=True),
        )
        self.gate_layer = gate_layer(),
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()

        max = self.max_pool(x).view(b, c) # global max pooling
        avg = self.avg_pool(x).view(b, c) # global average pooling

        # channel attention
        max = self.fc1(max).view(b, c, 1, 1)
        avg = self.fc1(avg).view(b, c, 1, 1)
        mc = self.gate_layer(max.expand_as(x) + avg.expand_as(x))

        # apply channel attention to the input tensor
        x = x * mc.expand_as(x)

        # spatial attention
        max_s = self.max_pool(x)
        avg_s = self.avg_pool(x)
        pool_s = cat([max_s, avg_s], dim=1) # concatenate the two tensors
        ms = self.gate_layer(self.conv1(pool_s)) # apply convolutional layer

        # apply spatial attention
        return x * ms.expand_as(x)