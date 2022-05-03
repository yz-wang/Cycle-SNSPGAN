import torch
import torch.nn as nn
from FCA import FcaLayer


# n_residual_blocks=9
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Discriminator(nn.Module):
    """ The Discriminator used in the PatchGan
    """

    def __init__(self, input_nc):

        super(Discriminator, self).__init__()

        # The first 2 dimensional convolutional layer where batch normalization is applied
        # on the convolutional output before leaky reLu is used as an activation function.
        self.conv_1 = nn.Sequential(
            # in channels, out channels, filter kernel size, stride, padding, bias
            nn.Conv2d(input_nc, 64, 6, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, 6, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        # At the last layer batch normalization is not used and
        # sigmoid is applied to the convolutional output.
        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 1, 3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, three_channel_image):
        """ The discriminators forward pass.
            Input: three_channel_image: 3 x 256 x 256
            Output: 1 x 64 x 64
        """

        # h1: 64 x 128 x 128
        h1 = self.conv_1(three_channel_image)

        # h2: 128 x 64 x 64
        h2 = self.conv_2(h1)

        # h3: 256 x 64 x 64
        h3 = self.conv_3(h2)

        # h4: 512 x 64 x 64
        h4 = self.conv_4(h3)

        # h5: 1 x 64 x 64
        h5 = self.conv_5(h4)

        return h5


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Generator, self).__init__()

        # Encoding layers
        # Initial convolution block
        # inputsize:3*256*256, outputsize:64*256*256
        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7, stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            FcaLayer(64, 8, 32, 32)
        )

        # Downsampling
        # inputsize:64*256*256, outputsize:128*128*128
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            FcaLayer(128, 8, 32, 32)
        )
        # inputsize:128*128*128, outputsize:256*64*64
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            FcaLayer(256, 8, 32, 32)
        )

        # Residual blocks
        # inputsize:256*64*64, outputsize:256*64*64
        in_features = 256
        self.conv_41 = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(in_features, in_features, 3),
                                     nn.InstanceNorm2d(in_features),
                                     nn.ReLU(inplace=True))

        self.conv_42 = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(in_features, in_features, 3),
                                     nn.InstanceNorm2d(in_features),
                                     FcaLayer(in_features, 8, 32, 32))

        # Upsampling
        # inputsize:256*64*64, outputsize:128*128*128
        self.conv_5 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            FcaLayer(128, 8, 32, 32)
        )

        # inputsize:128*128*128, outputsize:64*256*256
        self.conv_6 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            FcaLayer(64, 8, 32, 32)
        )

        # Output layer
        self.conv_7 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(128, output_nc, 7),
            nn.Tanh()
        )


    def forward(self, x):
        """ The forward pass of the generator with skip connections
                """
        # Encoding
        # batch size x 64 x 256 x 256
        c1 = self.conv_1(x)
        # c1_1 = self.cbam1(c1)

        # batch size x 128 x 128 x 128
        c2 = self.conv_2(c1)

        # batch size x 256 x 64 x 64
        c3 = self.conv_3(c2)

        # Residual blocks * 9
        # batch size x 256 x 64 x 64
        # c4_1 = self.conv_41(c3)
        # c4_2 = c3 + self.conv_42(c4_1)
        # c4_3 = self.conv_41(c4_2)
        # c4_4 = c4_2 + self.conv_42(c4_3)
        # c4_5 = self.conv_41(c4_4)
        # c4_6 = c4_4 + self.conv_42(c4_5)
        # c4_7 = self.conv_41(c4_6)
        # c4_8 = c4_6 + self.conv_42(c4_7)
        # c4_9 = self.conv_41(c4_8)
        # c4_10 = c4_8 + self.conv_42(c4_9)
        # c4_11 = self.conv_41(c4_10)
        # c4_12 = c4_10 + self.conv_42(c4_11)
        # c4_13 = self.conv_41(c4_12)
        # c4_14 = c4_12 + self.conv_42(c4_13)
        # c4_15 = self.conv_41(c4_14)
        # c4_16 = c4_14 + self.conv_42(c4_15)
        # c4_17 = self.conv_41(c4_16)
        # c4 = c4_16 + self.conv_42(c4_17)

        # double res
        c4_1 = c3 + self.conv_41(c3)
        c4_2 = c3 + self.conv_42(c4_1)
        c4_3 = c4_2 + self.conv_41(c4_2)
        c4_4 = c4_2 + self.conv_42(c4_3)
        c4_5 = c4_4 + self.conv_41(c4_4)
        c4_6 = c4_4 + self.conv_42(c4_5)
        c4_7 = c4_6 + self.conv_41(c4_6)
        c4_8 = c4_6 + self.conv_42(c4_7)
        c4_9 = c4_8 + self.conv_41(c4_8)
        c4_10 = c4_8 + self.conv_42(c4_9)
        c4_11 = c4_10 + self.conv_41(c4_10)
        c4_12 = c4_10 + self.conv_42(c4_11)
        c4_13 = c4_12 + self.conv_41(c4_12)
        c4_14 = c4_12 + self.conv_42(c4_13)
        c4_15 = c4_14 + self.conv_41(c4_14)
        c4_16 = c4_14 + self.conv_42(c4_15)
        c4_17 = c4_16 + self.conv_41(c4_16)
        c4 = c4_16 + self.conv_42(c4_17)

        # Decoding
        # batch size x 512 x 64 x 64
        skip1_de = torch.cat((c3, c4), 1)

        # batch size x 128 x 128 x 128
        c1_de = self.conv_5(skip1_de)

        # batch size x 256 x 128 x 128
        skip2_de = torch.cat((c2, c1_de), 1)

        # batch size x 64 x 256 x 256
        c3_de = self.conv_6(skip2_de)

        # batch size x 128 x 256 x 256
        skip3_de = torch.cat((c1, c3_de), 1)

        # batch size x 3 x 256 x 256
        c4_de = self.conv_7(skip3_de)

        return c4_de

