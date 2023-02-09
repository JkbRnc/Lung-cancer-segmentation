import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)

        self.conv2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.conv_block = ConvolutionBlock(input_channels, output_channels)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv_block(inputs)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            input_channels, output_channels, kernel_size=2, stride=2, padding=0)
        self.conv = ConvolutionBlock(
            output_channels + output_channels, output_channels)

    def forward(self, inputs, skip_connection):
        x = self.up(inputs)
        x = torch.cat([x, skip_connection], axis=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = EncoderBlock(3, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        """ Bottleneck/Bridge connection """
        self.b = ConvolutionBlock(512, 1024)

        """" Decoder """
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        """ Segmentation output """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoding """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck/Bridge connection """
        b = self.b(p4)  # 32 -> 64

        """ Decoding """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        return self.outputs(d4)
