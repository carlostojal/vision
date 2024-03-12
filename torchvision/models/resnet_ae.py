from typing import Any
import torch
from torch import nn
from torchvision.models import resnet50, se_resnet50, cbam_resnet50
from ._api import register_model
from ._utils import handle_legacy_interface

__all__ = [
    "resnet50_unet",
    "se_resnet50_unet",
    "cbam_resnet50_unet",
    "resnet50_ae",
    "se_resnet50_ae",
    "cbam_resnet50_ae"
]

class ResNetAutoEncoder(nn.Module):
    def __init__(
        self,
        resnet: nn.Module, # resnet encoder to use,
        with_residuals: bool = False, # whether to use residual connections from the encoder to the decoder
        attention: nn.Module = None, # attention module to use on the decoder
    ) -> None:
        super().__init__()

        self.resnet = resnet

        self.with_residuals = with_residuals

        #  define four decoder blocks, one for each feature map
        self.decoder1 = self.make_decoder_block(2048, 1024, attention)
        self.decoder2 = self.make_decoder_block(1024, 512, attention)
        self.decoder3 = self.make_decoder_block(512, 256, attention)
        self.decoder4 = self.make_decoder_block(256, 64, attention)
        self.decoder5 = self.make_decoder_block(64, 64, attention)

        # 1x1 convolution to obtain the final mask
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.with_residuals:
            return self.forward_unet(x)
        else:
            return self.forward_ae(x)
    
    def forward_ae(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, l4 = self.resnet(x)

        # without residual connections from the encoder to the decoder
        x = self.decoder1(l4)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)
        x = self.decoder5(x)

        # obtain the final mask
        x = self.final_conv(x)

        return x

    def forward_unet(self, x: torch.Tensor) -> torch.Tensor:

        # use the encoder, obtaining four feature maps of different sizes
        l1, l2, l3, l4 = self.resnet(x)

        # use the decoder blocks to obtain the final mask
        # the sums are the skip connections from the encoder
        x = self.decoder1(l4)
        x = self.decoder2(x + l3)
        x = self.decoder3(x + l2)
        x = self.decoder4(x + l1)
        x = self.decoder5(x)

        # obtain the final mask
        x = self.final_conv(x)

        return x

    # make a decoder block
    def make_decoder_block(self, in_channels: int, out_channels: int, attention: nn.Module = None) -> nn.Module:
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2), # 2x2 upsampling
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), # 3x3 conv
            nn.BatchNorm2d(out_channels), # batch normalization
            nn.ReLU()
        )

        # if an attention module was provided
        if attention is not None:
            block.add_module('attention', attention(out_channels))

        return block

# register the models
@register_model()
@handle_legacy_interface()
def resnet50_ae(*, in_channels: int = 3, weights = None, progress: bool = True, **kwargs: Any) -> ResNetAutoEncoder:
    resnet = resnet50(in_channels=in_channels)
    return ResNetAutoEncoder(resnet)

@register_model()
@handle_legacy_interface()
def se_resnet50_ae(*, in_channels: int = 3, weights = None, progress: bool = True, **kwargs: Any) -> ResNetAutoEncoder:
    resnet = se_resnet50(in_channels=in_channels)
    return ResNetAutoEncoder(resnet)

@register_model()
@handle_legacy_interface()
def cbam_resnet50_ae(*, in_channels: int = 3, weights = None, progress: bool = True, **kwargs: Any) -> ResNetAutoEncoder:
    resnet = cbam_resnet50(in_channels=in_channels)
    return ResNetAutoEncoder(resnet)

@register_model()
@handle_legacy_interface()
def resnet50_unet(*, in_channels: int = 3, weights = None, progress: bool = True, **kwargs: Any) -> ResNetAutoEncoder:
    resnet = resnet50(in_channels=in_channels)
    return ResNetAutoEncoder(resnet, with_residuals=True)

@register_model()
@handle_legacy_interface()
def se_resnet50_unet(*, in_channels: int = 3, weights = None, progress: bool = True, **kwargs: Any) -> ResNetAutoEncoder:
    resnet = se_resnet50(in_channels=in_channels)
    return ResNetAutoEncoder(resnet, with_residuals=True)

@register_model()
@handle_legacy_interface()
def cbam_resnet50_unet(*, in_channels: int = 3, weights = None, progress: bool = True, **kwargs: Any) -> ResNetAutoEncoder:
    resnet = cbam_resnet50(in_channels=in_channels)
    return ResNetAutoEncoder(resnet, with_residuals=True)