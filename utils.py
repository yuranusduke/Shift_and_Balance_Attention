"""
Define some utilities

Created by Kunhong Yu
Date: 2021/03/28
"""
import torch as t
from torch.nn import functional as F
from SBAttention import SBAttention

####################################
#         Define utilities         #
####################################
def _conv_layer(in_channels,
                out_channels):
    """Define conv layer
    Args :
        --in_channels: input channels
        --out_channels: output channels
    return :
        --conv layer
    """
    conv_layer = t.nn.Sequential(
        t.nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                    kernel_size = 3, stride = 1, padding = 1),
        t.nn.BatchNorm2d(out_channels),
        t.nn.ReLU(inplace = True)
    )

    return conv_layer


def vgg_block(in_channels,
              out_channels,
              repeat):
    """Define VGG block
    Args :
        --in_channels: input channels
        --out_channels: output channels
        --repeat
    return :
        --block
    """
    block = [
        _conv_layer(in_channels = in_channels if i == 0 else out_channels,
                    out_channels = out_channels)
        for i in range(repeat)
    ]

    return block

####################################
#            Define SE             #
####################################
class SEAttention(t.nn.Module):
    """Define SE operation"""

    def __init__(self, num_channels, attn_ratio):
        """
        Args :
             --num_channels: # of input channels
             --attn_ratio: hidden size ratio
        """
        super(SEAttention, self).__init__()

        self.num_channels = num_channels
        self.hidden_size = int(attn_ratio * self.num_channels)

        # 1. Trunk, we use T(x) = x
        # 2. SE attention
        self.SE = t.nn.Sequential(
            t.nn.Linear(self.num_channels, self.hidden_size),
            t.nn.BatchNorm1d(self.hidden_size),
            t.nn.ReLU(inplace = True),

            t.nn.Linear(self.hidden_size, self.num_channels),
            t.nn.BatchNorm1d(self.num_channels),
            t.nn.Sigmoid()
        )

    def forward(self, x):
        # 1. T(x)
        Tx = x
        # 2. SE attention
        x = F.adaptive_avg_pool2d(x, (1, 1)) # global average pooling
        x = x.squeeze()
        Ax = self.SE(x)

        # 3. output
        x = Tx * t.unsqueeze(t.unsqueeze(Ax, dim = -1), dim = -1) # broadcasting

        return x

####################################
#          Define VGG16            #
####################################
class VGG16(t.nn.Module):
    """Define VGG16-style model"""

    def __init__(self, use_sb = True, use_se = False, device = 'cuda', attn_ratio = 0.5, activation = 'tanh'):
        """
        Args :
            --use_sb: default is True
            --device: learning device, default is 'cuda'
            --attn_ratio: hidden size ratio, default is 0.5
        """
        super(VGG16, self).__init__()

        assert use_se + use_sb != 2 # can't make se and sb happen at the same time

        self.use_sb = use_sb
        self.use_se = use_se

        self.layer1 = t.nn.Sequential(*vgg_block(in_channels = 3,
                                                 out_channels = 64,
                                                 repeat = 2))

        if self.use_sb:
            self.sb1 = SBAttention(num_channels = 64,
                                   device = device, attn_ratio = attn_ratio, activation = activation)
        elif self.use_se:
            self.se1 = SEAttention(num_channels = 64,
                                   attn_ratio = attn_ratio)

        self.layer2 = t.nn.Sequential(*vgg_block(in_channels = 64,
                                                 out_channels = 128,
                                                 repeat = 2))

        if self.use_sb:
            self.sb2 = SBAttention(num_channels = 128,
                                   device = device, attn_ratio = attn_ratio, activation = activation)
        elif self.use_se:
            self.se2 = SEAttention(num_channels = 128,
                                   attn_ratio = attn_ratio)

        self.layer3 = t.nn.Sequential(*vgg_block(in_channels = 128,
                                                 out_channels = 256,
                                                 repeat = 3))

        if self.use_sb:
            self.sb3 = SBAttention(num_channels = 256,
                                   device = device, attn_ratio = attn_ratio, activation = activation)
        elif self.use_se:
            self.se3 = SEAttention(num_channels = 256,
                                   attn_ratio = attn_ratio)

        self.layer4 = t.nn.Sequential(*vgg_block(in_channels = 256,
                                                 out_channels = 512,
                                                 repeat = 3))

        if self.use_sb:
            self.sb4 = SBAttention(num_channels = 512,
                                   device = device, attn_ratio = attn_ratio, activation = activation)
        elif self.use_se:
            self.se4 = SEAttention(num_channels = 512,
                                   attn_ratio = attn_ratio)

        self.fc = t.nn.Sequential(  # unlike original VGG16, I reduce some fc
            # parameters to fit my 2070 device
            t.nn.Linear(512, 256),
            t.nn.ReLU(inplace = True),
            t.nn.Linear(256, 10)
        )

        self.max_pool = t.nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        x1 = self.layer1(x)
        if self.use_sb:
            x1 = self.sb1(x1)
        elif self.use_se:
            x1 = self.se1(x1)
        x1 = self.max_pool(x1)

        x2 = self.layer2(x1)
        if self.use_sb:
            x2 = self.sb2(x2)
        elif self.use_se:
            x2 = self.se2(x2)
        x2 = self.max_pool(x2)

        x3 = self.layer3(x2)
        if self.use_sb:
            x3 = self.sb3(x3)
        elif self.use_se:
            x3 = self.se3(x3)
        x3 = self.max_pool(x3)

        x4 = self.layer4(x3)
        if self.use_sb:
            x4 = self.sb4(x4)
        elif self.use_se:
            x4 = self.se4(x4)
        x4 = self.max_pool(x4)

        x = F.adaptive_avg_pool2d(x4, (1, 1))
        x = x.squeeze()
        x = self.fc(x)

        return x