from torch import nn, Tensor

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) block from 'Squeeze-and-Excitation Networks,' Hu et al.
    (https://arxiv.org/abs/1709.01507).

    Args:
        in_channels (int): Number of channels of the input tensor
        reduction_ratio (int): Reduction ratio for the squeeze step. Default: 16
        act_layer (nn.Module): Activation layer. Default: nn.ReLU
        gate_layer (nn.Module): Gate layer. Default: nn.Sigmoid

    Examples::
        >>> m = torchvision.ops.SqueezeExcitation(10, 1)
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
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=True),
            act_layer(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=True),
            gate_layer(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)