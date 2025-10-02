import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class ChannelReconstructionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(5,64); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64,128); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128,256); self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256,512); self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512,1024)
        self.upconv4 = nn.ConvTranspose2d(1024,512,2,2); self.dec4 = DoubleConv(1024,512)
        self.upconv3 = nn.ConvTranspose2d(512,256,2,2); self.dec3 = DoubleConv(512,256)
        self.upconv2 = nn.ConvTranspose2d(256,128,2,2); self.dec2 = DoubleConv(256,128)
        self.upconv1 = nn.ConvTranspose2d(128,64,2,2); self.dec1 = DoubleConv(128,64)
        self.final_conv = nn.Conv2d(64,5,1)
    
    def forward(self, image, mask_vector):
        # Masquer les canaux à reconstruire
        image = image.clone()
        for i in range(image.shape[0]):
            for c in range(5):
                if mask_vector[i,c] == 1: image[i,c] = -1.0
        x = image
        e1 = self.enc1(x); p1=self.pool1(e1)
        e2 = self.enc2(p1); p2=self.pool2(e2)
        e3 = self.enc3(p2); p3=self.pool3(e3)
        e4 = self.enc4(p3); p4=self.pool4(e4)
        b = self.bottleneck(p4)
        up4 = self.upconv4(b); merge4 = torch.cat([up4,e4],1); d4=self.dec4(merge4)
        up3 = self.upconv3(d4); merge3 = torch.cat([up3,e3],1); d3=self.dec3(merge3)
        up2 = self.upconv2(d3); merge2 = torch.cat([up2,e2],1); d2=self.dec2(merge2)
        up1 = self.upconv1(d2); merge1 = torch.cat([up1,e1],1); d1=self.dec1(merge1)
        output = self.final_conv(d1)
        return torch.sigmoid(output)
