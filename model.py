import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):

    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes

        self.contracting_11 = self.conv_block(in_channels=1, out_channels=64)
        self.contracting_12 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.contracting_41 = self.conv_block(in_channels=256, out_channels=512)
        self.contracting_42 = nn.MaxPool3d(kernel_size=2, stride=2)
        # central hidden layer
        self.middle = self.conv_block(in_channels=512, out_channels=1024)
        
        self.expansive_11 = nn.ConvTranspose3d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=1024, out_channels=512)
        self.expansive_21 = nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_31 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_41 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv3d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
    
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm3d(num_features=out_channels),
                              nn.ReLU(),
                              nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm3d(num_features=out_channels),
                              nn.ReLU(),
                             )
        return block

    def forward(self, x):
        
        # encoding
        x1 = self.contracting_11(x)
        x2 = self.contracting_12(x1) 
        x3 = self.contracting_21(x2)
        x4 = self.contracting_22(x3)
        x5 = self.contracting_31(x4)
        x6 = self.contracting_32(x5)
        x7 = self.contracting_41(x6)
        x8 = self.contracting_42(x7) 
        middle_out = self.middle(x8)

        # decoding
        x = self.expansive_11(middle_out)
        x = self.expansive_12(torch.cat((x, x7), dim=1))
        x = self.expansive_21(x)
        x = self.expansive_22(torch.cat((x, x5), dim=1))
        x = self.expansive_31(x)
        x = self.expansive_32(torch.cat((x, x3), dim=1))
        x = self.expansive_41(x)
        x = self.expansive_42(torch.cat((x, x1), dim=1))
        output = self.output(x)
        output = self.sigmoid(output)

        return output
