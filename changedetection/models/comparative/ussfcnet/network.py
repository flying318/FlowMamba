import sys
sys.path.append('changedetection/models/comparative')
import torch.nn as nn
import torch.nn.functional as F
import torch
from thop import profile, clever_format
from torchsummary import summary
from ussfcnet.util import First_DoubleConv, DoubleConv


class USSFCNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=2, ratio=0.5):
        super(USSFCNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_1 = First_DoubleConv(in_ch, int(64 * ratio))
        self.Conv1_2 = First_DoubleConv(in_ch, int(64 * ratio))
        self.Conv2_1 = DoubleConv(int(64 * ratio), int(128 * ratio))
        self.Conv2_2 = DoubleConv(int(64 * ratio), int(128 * ratio))
        self.Conv3_1 = DoubleConv(int(128 * ratio), int(256 * ratio))
        self.Conv3_2 = DoubleConv(int(128 * ratio), int(256 * ratio))
        self.Conv4_1 = DoubleConv(int(256 * ratio), int(512 * ratio))
        self.Conv4_2 = DoubleConv(int(256 * ratio), int(512 * ratio))
        self.Conv5_1 = DoubleConv(int(512 * ratio), int(1024 * ratio))
        self.Conv5_2 = DoubleConv(int(512 * ratio), int(1024 * ratio))

        self.Up5 = nn.ConvTranspose2d(int(1024 * ratio), int(512 * ratio), 2, stride=2)
        self.Up_conv5 = DoubleConv(int(1024 * ratio), int(512 * ratio))

        self.Up4 = nn.ConvTranspose2d(int(512 * ratio), int(256 * ratio), 2, stride=2)
        self.Up_conv4 = DoubleConv(int(512 * ratio), int(256 * ratio))

        self.Up3 = nn.ConvTranspose2d(int(256 * ratio), int(128 * ratio), 2, stride=2)
        self.Up_conv3 = DoubleConv(int(256 * ratio), int(128 * ratio))

        self.Up2 = nn.ConvTranspose2d(int(128 * ratio), int(64 * ratio), 2, stride=2)
        self.Up_conv2 = DoubleConv(int(128 * ratio), int(64 * ratio))

        self.Conv_1x1 = nn.Conv2d(int(64 * ratio), out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):
        # encoding
        # x1, x2 = torch.unsqueeze(x1[0], dim=0), torch.unsqueeze(x1[1], dim=0)
        c1_1 = self.Conv1_1(x1)
        c1_2 = self.Conv1_2(x2)
        x1 = torch.abs(torch.sub(c1_1, c1_2))

        c2_1 = self.Maxpool(c1_1)
        c2_1 = self.Conv2_1(c2_1)
        c2_2 = self.Maxpool(c1_2)
        c2_2 = self.Conv2_2(c2_2)
        x2 = torch.abs(torch.sub(c2_1, c2_2))

        c3_1 = self.Maxpool(c2_1)
        c3_1 = self.Conv3_1(c3_1)
        c3_2 = self.Maxpool(c2_2)
        c3_2 = self.Conv3_2(c3_2)
        x3 = torch.abs(torch.sub(c3_1, c3_2))

        c4_1 = self.Maxpool(c3_1)
        c4_1 = self.Conv4_1(c4_1)
        c4_2 = self.Maxpool(c3_2)
        c4_2 = self.Conv4_2(c4_2)
        x4 = torch.abs(torch.sub(c4_1, c4_2))

        c5_1 = self.Maxpool(c4_1)
        c5_1 = self.Conv5_1(c5_1)
        c5_2 = self.Maxpool(c4_2)
        c5_2 = self.Conv5_2(c5_2)
        x5 = torch.abs(torch.sub(c5_1, c5_2))

        # decoding
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class USSFCNet_BDA(USSFCNet):
    def __init__(self, out_loc, out_dmg, ratio=0.5):
        super().__init__(ratio=ratio, out_ch=out_loc)
        self.dmg_up5 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dmg_up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dmg_up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dmg_up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dmg_up_conv5 = DoubleConv(1024, 512)
        self.dmg_up_conv4 = DoubleConv(512, 256)
        self.dmg_up_conv3 = DoubleConv(256, 128)
        self.dmg_up_conv2 = DoubleConv(128, 64)
        self.seg_up_conv5, self.seg_up_conv4, self.seg_up_conv3, self.seg_up_conv2 = self.Up_conv5, \
            self.Up_conv4, self.Up_conv3, self.Up_conv2
        self.seg_up5, self.seg_up4, self.seg_up3, self.seg_up2 = self.Up5, self.Up4, self.Up3, self.Up2
        self.cls_seg = nn.Conv2d(32, out_loc, kernel_size=1, stride=1, padding=0)
        self.cls_dmg = nn.Conv2d(64, out_dmg, kernel_size=1, stride=1, padding=0)
        del self.Conv_1x1, self.Up5, self.Up4, self.Up3, self.Up2, \
            self.Up_conv5, self.Up_conv4, self.Up_conv3, self.Up_conv2
    def forward(self, x1, x2):
        size = x1.size()[2:]
        # encoding
        # x1, x2 = torch.unsqueeze(x1[0], dim=0), torch.unsqueeze(x1[1], dim=0)
        c1_1 = self.Conv1_1(x1)
        c1_2 = self.Conv1_2(x2)
        seg_1 = c1_1
        # x1 = torch.abs(torch.sub(c1_1, c1_2))
        x1 = torch.cat((c1_1, c1_2), dim=1)

        c2_1 = self.Maxpool(c1_1)
        c2_1 = self.Conv2_1(c2_1)
        c2_2 = self.Maxpool(c1_2)
        c2_2 = self.Conv2_2(c2_2)
        seg_2 = c2_1
        # x2 = torch.abs(torch.sub(c2_1, c2_2))
        x2 = torch.cat((c2_1, c2_2), dim=1)

        c3_1 = self.Maxpool(c2_1)
        c3_1 = self.Conv3_1(c3_1)
        c3_2 = self.Maxpool(c2_2)
        c3_2 = self.Conv3_2(c3_2)
        seg_3 = c3_1
        # x3 = torch.abs(torch.sub(c3_1, c3_2))
        x3 = torch.cat((c3_1, c3_2), dim=1)

        c4_1 = self.Maxpool(c3_1)
        c4_1 = self.Conv4_1(c4_1)
        c4_2 = self.Maxpool(c3_2)
        c4_2 = self.Conv4_2(c4_2)
        seg_4 = c4_1
        # x4 = torch.abs(torch.sub(c4_1, c4_2))
        x4 = torch.cat((c4_1, c4_2), dim=1)

        c5_1 = self.Maxpool(c4_1)
        c5_1 = self.Conv5_1(c5_1)
        c5_2 = self.Maxpool(c4_2)
        c5_2 = self.Conv5_2(c5_2)
        seg_5 = c5_1
        # x5 = torch.abs(torch.sub(c5_1, c5_2))
        x5 = torch.cat((c5_1, c5_2), dim=1)

        # decoding
        dmg_5 = self.dmg_up_conv5(torch.cat((x4, self.dmg_up5(x5)), dim=1))
        seg_5 = self.seg_up_conv5(torch.cat((seg_4, self.seg_up5(seg_5)), dim=1))

        dmg_4 = self.dmg_up_conv4(torch.cat((x3, self.dmg_up4(dmg_5)), dim=1))
        seg_4 = self.seg_up_conv4(torch.cat((seg_3, self.seg_up4(seg_5)), dim=1))

        dmg_3 = self.dmg_up_conv3(torch.cat((x2, self.dmg_up3(dmg_4)), dim=1))
        seg_3 = self.seg_up_conv3(torch.cat((seg_2, self.seg_up3(seg_4)), dim=1))

        dmg_2 = self.dmg_up_conv2(torch.cat((x1, self.dmg_up2(dmg_3)), dim=1))
        seg_2 = self.seg_up_conv2(torch.cat((seg_1, self.seg_up2(seg_3)), dim=1))

        dmg = self.cls_dmg(dmg_2)
        seg = self.cls_seg(seg_2)
        dmg = F.interpolate(dmg, size, mode='bilinear', align_corners=True)
        seg = F.interpolate(seg, size, mode='bilinear', align_corners=True)

        return seg, dmg


if __name__ == "__main__":
    A2016 = torch.randn(8, 3, 256, 256).to('cuda:0')
    A2019 = torch.randn(8, 3, 256, 256).to('cuda:0')
    # model = USSFCNet(3, 1, ratio=0.5).to('cuda:0')
    model = USSFCNet_BDA(2, 5, ratio=0.5).to('cuda:0')
    out_result = model(A2016, A2019)

    # summary(model, [(3, 256, 256), (3, 256, 256)])
    flops, params = profile(model, inputs=(A2016, A2019))
    flops, params = clever_format([flops, params], "%.3f")
    total = sum(p.numel() for p in model.parameters())
    print(f'flops: {flops}, params: {params}, Total params: {total/1e6:.4f} M')
    print(out_result[0].shape, out_result[1].shape)
    from thop import profile
    flops, params = profile(model, inputs=(A2016, A2019), verbose=False)
    print(f"FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M")