import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DoubleConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True))
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(ic, oc))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.up = nn.ConvTranspose2d(ic, ic // 2, 2, stride=2)
        self.conv = DoubleConv(ic, oc)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2); dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([x2, x1], 1))

class LightweightUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(1, 16)
        self.down1, self.down2, self.down3, self.down4 = Down(16,32), Down(32,64), Down(64,128), Down(128,256)
        self.up1, self.up2, self.up3, self.up4 = Up(256,128), Up(128,64), Up(64,32), Up(32,16)
        self.outc = nn.Conv2d(16, 1, 1)
    def forward(self, x):
        x1=self.inc(x); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); x5=self.down4(x4)
        x=self.up1(x5,x4); x=self.up2(x,x3); x=self.up3(x,x2); x=self.up4(x,x1)
        return torch.sigmoid(self.outc(x))

class RobustVascularMetricsExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('neighbor_kernel', torch.tensor([[[[1.,1.,1.],[1.,0.,1.],[1.,1.,1.]]]]))
        self.register_buffer('sobel_x', torch.tensor([[[[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]]], dtype=torch.float32))
        self.register_buffer('sobel_y', torch.tensor([[[[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]]], dtype=torch.float32))

    def gpu_skeletonize(self, mask, iterations=8):
        thin = mask.clone()
        for _ in range(iterations):
            nb = F.conv2d(thin, self.neighbor_kernel, padding=1) * thin
            ero = 1.0 - F.max_pool2d(1.0 - thin, 3, stride=1, padding=1)
            bnd = ((thin - ero) > 0.5).float()
            rem = bnd * (nb >= 2).float() * (nb <= 6).float()
            thin = (thin * (1.0 - rem) > 0.5).float()
            if rem.sum() == 0: break
        return thin

    def forward(self, bm):
        B, eps = bm.shape[0], 1e-6

        # ----- Adaptive binarization: keep brightest 40% pixels per image -----
        flat = bm.view(B, -1)
        k = int(0.3 * flat.shape[1])          # discard 30% (keep top 70%)
        thresh_vals, _ = torch.kthvalue(flat, k, dim=1)
        thresh_vals = thresh_vals.view(B,1,1,1)
        binary = (bm > thresh_vals).float()
        sk = self.gpu_skeletonize(binary)

        # ---- DEBUG: check skeleton stats ----
        sk_sum = sk.view(B,-1).sum(1)

        # ---- vessel density ----
        dn = bm.view(B,-1).mean(1)

        # ---- fractal dimension & lacunarity ----
        scales = [1,2,4,8,16]
        bcs, lacs = [], []
        for s in scales:
            Ns = bm.view(B,-1).sum(1) if s==1 else F.max_pool2d(bm,s,s).view(B,-1).sum(1)
            ap = bm.view(B,-1) if s==1 else F.avg_pool2d(bm,s,s).view(B,-1)
            bcs.append(Ns)
            mm = ap.mean(1)
            vm = ap.var(1, unbiased=False)
            lacs.append(vm/(mm**2+eps))
        bct = torch.stack(bcs,1)
        lis = torch.tensor([-math.log(s+eps) for s in scales], device=bm.device).unsqueeze(0).expand(B,-1)
        lN = torch.log(bct+eps)
        xm, ym = lis.mean(1,True), lN.mean(1,True)
        fd = torch.clamp(((lis-xm)*(lN-ym)).sum(1)/(((lis-xm)**2).sum(1)+eps), 0.5, 2.0)
        lac = torch.stack(lacs,1).mean(1)

        # ---- branching index & endpoints ----
        sn = F.conv2d(sk, self.neighbor_kernel, padding=1) * sk
        ep = (sn == 1.).float().view(B,-1).sum(1)
        br = (sn > 2.).float().view(B,-1).sum(1)
        bl = sk.view(B,-1).sum(1)

        # ---- tortuosity (unchanged) ----
        grad_x = F.conv2d(bm, self.sobel_x, padding=1)
        grad_y = F.conv2d(bm, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + eps)
        vessel_mask = (bm > 0.3).float()
        grad_mag_masked = grad_mag * vessel_mask
        at = (grad_mag_masked.view(B,-1).sum(1)) / (vessel_mask.view(B,-1).sum(1) + eps)
        k = max(1, int(0.1 * vessel_mask.view(B,-1).sum(1).max().item()))
        grad_vals = grad_mag_masked.view(B,-1)
        sorted_vals, _ = grad_vals.sort(dim=1, descending=True)
        mt = sorted_vals[:, :k].mean(dim=1)

        out = torch.stack([dn, fd, lac, at, mt, br, ep, bl], 1)
        return torch.nan_to_num(out, nan=0., posinf=0., neginf=0.)

class RobustGeometricEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = RobustVascularMetricsExtractor()
        self.thresholds = [0.3, 0.5, 0.7]
        self.correction_mlp = nn.Sequential(nn.Linear(8,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64,64))
    def forward(self, soft_map):
        sm = F.interpolate(soft_map, size=(128,128), mode='bilinear', align_corners=False)
        all_m = [self.extractor(torch.sigmoid(20.*(sm-t))).unsqueeze(1) for t in self.thresholds]
        raw, _ = torch.median(torch.cat(all_m, 1), dim=1)
        return self.correction_mlp(raw), raw

class BidirectionalCAFM(nn.Module):
    def __init__(self, vis_dim=1280, embed_dim=64, num_heads=4):
        super().__init__()
        self.vis_proj = nn.Linear(vis_dim, embed_dim)
        self.attn_v2g = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_g2v = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_v, self.norm_g = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.fusion_mlp = nn.Sequential(nn.Linear(embed_dim*2, 512), nn.ReLU())
    def forward(self, vf, ge):
        B,C,H,W = vf.shape
        vs = self.vis_proj(vf.view(B,C,-1).transpose(1,2)); gs = ge.unsqueeze(1)
        vr,_ = self.attn_v2g(vs,gs,gs); vr = self.norm_v(vs+vr)
        gr,attn = self.attn_g2v(gs,vr,vr); gr = self.norm_g(gs+gr)
        return self.fusion_mlp(torch.cat([vr.mean(1), gr.squeeze(1)], 1)), attn

class FinalPatentArchitecture(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.vis_backbone = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', pretrained=False)
        self.vis_pool = nn.AdaptiveAvgPool2d((7,7))
        self.unet = LightweightUNet()
        for p in self.unet.parameters(): p.requires_grad = False
        self.unet.eval()
        self.geo_encoder = RobustGeometricEncoder()
        self.cafm = BidirectionalCAFM()
        self.classifier = nn.Sequential(nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256,4))
    def forward(self, iv, ig, use_unet=True):
        vf = self.vis_pool(self.vis_backbone.forward_features(iv))
        sm = self.unet(ig) if use_unet else ig
        ge, rm = self.geo_encoder(sm)
        fv, attn = self.cafm(vf, ge)
        return self.classifier(fv), rm, attn
