import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from msd_pytorch.msd_model import (MSDModel)

# This code is copied and adapted from:
# https://github.com/milesial/Pytorch-UNet

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down_avg(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_avg, self).__init__()
        self.mpconv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 2, padding=0, stride=2),
            double_conv(in_ch, out_ch)
        )
        self.mpconv[0].weight.data.fill_(0)
        for c in range(self.mpconv[0].weight.size()[0]):
            self.mpconv[0].weight.data[c,c,:,:].fill_(0.25)
        self.mpconv[0].bias.data.fill_(0)
        for param in self.mpconv[0].parameters():
             param.requires_grad = False

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet4Avg(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet4Avg, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down_avg(64, 128)
        self.down2 = down_avg(128, 256)
        self.down3 = down_avg(256, 512)
        self.down4 = down_avg(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        H, W = x.shape[2:]
        Hp, Wp = ((-H % 16), (-W % 16))
        padding = (Wp // 2, Wp - Wp // 2, Hp // 2, Hp - Hp // 2)
        reflect = nn.ReflectionPad2d(padding)
        x = reflect(x)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        H2 = H + padding[2] + padding[3]
        W2 = W + padding[0] + padding[1]
        return x[:, :, padding[2]:H2-padding[3], padding[0]:W2-padding[1]]

    def clear_buffers(self):
        pass

class UNet4AvgSegmModel(MSDModel):
    def __init__(self, c_in, num_labels):
        #Allow supplying a list of labels instead of just the number
        # of labels.
        if isinstance(num_labels, list):
            c_out = len(num_labels)
            self.labels = num_labels
        else:
            self.labels = range(num_labels)
            c_out = num_labels

        # Initialize msd network. The dilation, width, depth do nothing, its a wrapper.
        super().__init__(c_in, c_out, 1, 1, [1,2,3,4,5,6,7,8,9,10])

        self.criterion = nn.NLLLoss()

        # Make Unet
        self.msd = UNet4Avg(c_in, c_out)

        # Initialize network
        net_trained = nn.Sequential(self.msd, nn.LogSoftmax(dim=1))
        self.net = nn.Sequential(self.scale_in, net_trained)
        self.net.cuda()

        # Train all parameters apart from self.scale_in.
        self.init_optimizer(net_trained)

    def set_normalization(self, dataloader):
        """Normalize input data.

        This function goes through all the training data to compute
        the mean and std of the training data. It modifies the
        network so that all future invocations of the network first
        normalize input data. The normalization parameters are saved.

        :param dataloader: The dataloader associated to the training data.
        :returns:
        :rtype:

        """
        mean = 0
        square = 0
        for (data_in, _) in dataloader:
            mean += data_in.mean()
            square += data_in.pow(2).mean()

        mean /= len(dataloader)
        square /= len(dataloader)
        std = np.sqrt(square - mean ** 2)

        # The input data should be roughly normally distributed after
        # passing through net_fixed.
        self.scale_in.bias.data.fill_(- mean / std)
        self.scale_in.weight.data.fill_(1 / std)

    def set_target(self, data):
        # relabel if necessary:
        target = data.clone()
        if self.labels:
            for i, label in enumerate(self.labels):
                target[data == label] = i

        # The class labels must be of long data type
        target = target.long()
        # The NLLLoss does not accept a channel dimension. So we
        # squeeze the target.
        target = target.squeeze(1)
        # The class labels must reside on the GPU
        target = target.cuda()
        self.target = Variable(target)
