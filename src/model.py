import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelSpatialAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelSpatialAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_att
        
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_s, max_s], dim=1)))
        return x * spatial_att

class TransformerBottleneck(nn.Module):
    def __init__(self, in_channels, embed_dim=256, num_heads=4):
        super(TransformerBottleneck, self).__init__()
        self.project_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.project_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_proj = self.project_in(x)
        x_flat = x_proj.flatten(2).permute(0, 2, 1)
        x_trans = self.transformer(x_flat)
        x_reshaped = x_trans.permute(0, 2, 1).view(B, -1, H, W)
        return self.project_out(x_reshaped)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class MultiClassSiameseUNet(nn.Module):
    """Module 12: 5-Class Semantic Change Detection Architecture"""
    def __init__(self, in_channels=3, num_classes=5):
        super(MultiClassSiameseUNet, self).__init__()
        
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.bottleneck = DoubleConv(1024, 512)
        self.transformer = TransformerBottleneck(512, embed_dim=256, num_heads=4)
        
        self.att3 = ChannelSpatialAttention(512)
        self.att2 = ChannelSpatialAttention(256)
        self.att1 = ChannelSpatialAttention(128)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(768, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(384, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(192, 64)
        
        # 5 Output channels for multi-class logits
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward_single(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x1, x2, x3, x4

    def forward(self, t1, t2):
        t1_x1, t1_x2, t1_x3, t1_x4 = self.forward_single(t1)
        t2_x1, t2_x2, t2_x3, t2_x4 = self.forward_single(t2)

        f4 = torch.cat([t1_x4, t2_x4], dim=1)
        f3 = torch.cat([t1_x3, t2_x3], dim=1)
        f2 = torch.cat([t1_x2, t2_x2], dim=1)
        f1 = torch.cat([t1_x1, t2_x1], dim=1)

        b = self.bottleneck(f4)
        b = self.transformer(b)

        d3 = self.up3(b)
        f3_att = self.att3(f3)
        d3 = self.dec3(torch.cat([d3, f3_att], dim=1))

        d2 = self.up2(d3)
        f2_att = self.att2(f2)
        d2 = self.dec2(torch.cat([d2, f2_att], dim=1))

        d1 = self.up1(d2)
        f1_att = self.att1(f1)
        d1 = self.dec1(torch.cat([d1, f1_att], dim=1))

        return self.outc(d1) # (B, 5, H, W)

# Aliases for backward compatibility
HybridTransformerSiameseUNet = MultiClassSiameseUNet
AttentionSiameseUNet = MultiClassSiameseUNet
SiameseUNet = MultiClassSiameseUNet
