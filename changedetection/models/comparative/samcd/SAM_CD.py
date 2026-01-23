import sys
sys.path.append('/mnt/home/guoxin/proj/MambaCD/changedetection/models/comparative')
import torch
from torch import nn
from samcd.FastSAM.fastsam import FastSAM
from torch.nn import functional as F
from utils.util import initialize_weights
from samcd.util import LatentSimilarity, _DecoderBlock, ResBlock, Space_Attention, conv1x1, conv3x3

class SAM_CD(nn.Module):
    def __init__(
        self,
        num_embed=8,
        model_name: str='FastSAM-x.pt',
        device: str='cuda',
        conf: float=0.4,
        iou: float=0.9,
        imgsz: int=1024,
        retina_masks: bool=True,
        done_warmup: bool=True,
        ):
        super(SAM_CD, self).__init__()
        self.model = FastSAM(model_name)
        self.device = device
        self.retina_masks = retina_masks
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.image = None
        self.image_feats = None
        self.num_bed = num_embed
         
        self.Adapter32 = nn.Sequential(nn.Conv2d(640, 160, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(160), nn.ReLU())
        self.Adapter16 = nn.Sequential(nn.Conv2d(640, 160, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(160), nn.ReLU())
        self.Adapter8 = nn.Sequential(nn.Conv2d(320, 80, kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(80), nn.ReLU())
        self.Adapter4 = nn.Sequential(nn.Conv2d(160, 40, kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(40), nn.ReLU())
                                       
        self.Dec2 = _DecoderBlock(160, 160, 80)
        self.Dec1 = _DecoderBlock(80, 80, 40)  
        self.Dec0 = _DecoderBlock(40, 40, 64)
        
        self.SA = Space_Attention(16, 16, 4)
        self.segmenter = nn.Conv2d(64, num_embed, kernel_size=1)        
        self.resCD = self._make_layer(ResBlock, 128, 128, 6, stride=1)
        self.headC = nn.Sequential(nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.segmenterC = nn.Conv2d(16, 1, kernel_size=1)
        self.criterion = LatentSimilarity(T=3.0).to(self.device)
                                        
        for param in self.model.model.parameters():
            param.requires_grad = False
        initialize_weights(self.Adapter32, self.Adapter16, self.Adapter8, self.Adapter4, self.Dec2, self.Dec1, self.Dec0,\
                           self.segmenter, self.resCD, self.headC, self.segmenterC)

    def run_encoder(self, image):
        self.image = image
        feats = self.model(
            self.image,
            device=self.device,
            retina_masks=self.retina_masks,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou
            )
        return feats

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, label: torch.Tensor):
    
        input_shape = x1.shape[-2:]
        featsA = self.run_encoder(x1)
        featsB = self.run_encoder(x2)
        
        featA_s4 = self.Adapter4(featsA[3].clone())
        featA_s8 = self.Adapter8(featsA[0].clone())
        featA_s16 = self.Adapter16(featsA[1].clone())
        featA_s32 = self.Adapter32(featsA[2].clone())
        
        decA_2 = self.Dec2(featA_s32, featA_s16)
        decA_1 = self.Dec1(decA_2, featA_s8)
        decA_0 = self.Dec0(decA_1, featA_s4)
        outA = self.segmenter(decA_0)
        
        featB_s4 = self.Adapter4(featsB[3].clone())
        featB_s8 = self.Adapter8(featsB[0].clone())
        featB_s16 = self.Adapter16(featsB[1].clone())
        featB_s32 = self.Adapter32(featsB[2].clone())              
        
        decB_2 = self.Dec2(featB_s32, featB_s16)
        decB_1 = self.Dec1(decB_2, featB_s8)
        decB_0 = self.Dec0(decB_1, featB_s4)
        outB = self.segmenter(decB_0)
             
        A = self.SA(torch.cat([outA, outB], dim=1))  
        featC = torch.cat([decA_0, decB_0], 1)
        featC = self.resCD(featC)
        featC = self.headC(featC) * A
        outC = self.segmenterC(featC)
        out = F.interpolate(outC, input_shape, mode="bilinear", align_corners=True)
        out1 = F.interpolate(outA, input_shape, mode="bilinear", align_corners=True)
        out2 = F.interpolate(outB, input_shape, mode="bilinear", align_corners=True)
        label = label.unsqueeze(1) if label.dim() != 4 else label
        loss = self.criterion(out1, out2, label)
        return out, loss

class SAM_BDA(SAM_CD):
    def __init__(self, out_loc, out_dmg, num_bed=8, 
                 backbone='FastSAM-x.pt',
                 device='cuda', imgsz=1024):
        super(SAM_BDA, self).__init__(num_embed=num_bed, model_name=backbone,
                                      device=device, imgsz=imgsz)
        self.cls_seg = nn.Conv2d(num_bed, out_loc, kernel_size=1)
        self.cls_dmg = nn.Conv2d(16, out_dmg, kernel_size=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, label=None, mode='train'):
        size = x1.shape[-2:]
        feats_1 = self.run_encoder(x1)
        feats_2 = self.run_encoder(x2)
        feat_1_s4 = self.Adapter4(feats_1[3].clone())
        feat_1_s8 = self.Adapter8(feats_1[0].clone())
        feat_1_s16 = self.Adapter16(feats_1[1].clone())
        feat_1_s32 = self.Adapter32(feats_1[2].clone())
        
        dec_1_2 = self.Dec2(feat_1_s32, feat_1_s16)
        dec_1_1 = self.Dec1(dec_1_2, feat_1_s8)
        dec_1_0 = self.Dec0(dec_1_1, feat_1_s4)
        out_1 = self.segmenter(dec_1_0)
        
        feat_2_s4 = self.Adapter4(feats_2[3].clone())
        feat_2_s8 = self.Adapter8(feats_2[0].clone())
        feat_2_s16 = self.Adapter16(feats_2[1].clone())
        feat_2_s32 = self.Adapter32(feats_2[2].clone())

        dec_2_2 = self.Dec2(feat_2_s32, feat_2_s16)
        dec_2_1 = self.Dec1(dec_2_2, feat_2_s8)
        dec_2_0 = self.Dec0(dec_2_1, feat_2_s4)
        out_2 = self.segmenter(dec_2_0)
        
        A = self.SA(torch.cat([out_1, out_2], dim=1))
        feat_C = torch.cat([dec_1_0, dec_2_0], 1)
        feat_C = self.resCD(feat_C)
        feat_C = self.headC(feat_C) * A

        out_loc = self.cls_seg(out_1)
        out_dmg = self.cls_dmg(feat_C)
        out_loc = F.interpolate(out_loc, size, mode="bilinear", align_corners=True)
        out_dmg = F.interpolate(out_dmg, size, mode="bilinear", align_corners=True)
        if mode=='train' and label is not None:
            out_1 = F.interpolate(out_1, size, mode="bilinear", align_corners=True)
            out_2 = F.interpolate(out_2, size, mode="bilinear", align_corners=True)
            label = label.unsqueeze(1) if label.dim() != 4 else label
            loss = self.criterion(out_1, out_2, label)
            return out_loc, out_dmg, loss
        return out_loc, out_dmg
        

if __name__ == '__main__':
    import cv2
    # model = SAM_CD(device='cpu', imgsz=256)
    model = SAM_BDA(2, 5, device='cpu', imgsz=256, backbone='pretrained/FastSAM-x.pt')
    # x1 = cv2.imread('dataset/second/raw/test/T1/00019.png')
    # x2 = cv2.imread('dataset/second/raw/test/T2/00019.png')
    # x1 = torch.tensor(x1).permute(2, 0, 1).unsqueeze(0).float()
    # x2 = torch.tensor(x2).permute(2, 0, 1).unsqueeze(0).float()
    # label = torch.zeros(8, 1, 256, 256) 
    label = torch.randint(0, 2, (8, 1, 256, 256))
    x1 = torch.randn(8, 3, 256, 256)
    x2 = torch.randn(8, 3, 256, 256)
    out1, out2, loss = model(x1, x2, label)
    print(out1.shape, out2.shape, loss.item())
    from thop import profile
    flops, params = profile(model, inputs=(x1, x2), verbose=False)
    print(f"FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M")