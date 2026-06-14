class FinalPatentArchitecture(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.vis_backbone = timm.create_model(
            'tf_efficientnetv2_s.in21k_ft_in1k', pretrained=False)
        self.vis_pool     = nn.AdaptiveAvgPool2d((7, 7))
        self.unet         = LightweightUNet(n_channels=1, n_classes=1)
        for p in self.unet.parameters():
            p.requires_grad = False
        self.unet.eval()
        self.geo_encoder  = RobustGeometricEncoder()
        self.cafm         = BidirectionalCAFM(vis_dim=1280, embed_dim=64, num_heads=4)
        self.classifier   = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def freeze_backbone(self):
        for p in self.vis_backbone.parameters(): p.requires_grad = False
    def unfreeze_backbone(self):
        for p in self.vis_backbone.parameters(): p.requires_grad = True

    def forward(self, img_vis, img_geo, use_unet=True):
        vis_feats = self.vis_backbone.forward_features(img_vis)
        vis_feats = self.vis_pool(vis_feats)

        if use_unet:
            with torch.no_grad():
                soft_vessel_map = self.unet(img_geo)
        else:
            soft_vessel_map = img_geo

        geo_embed, raw_metrics = self.geo_encoder(soft_vessel_map)
        fused_vector, attn_maps = self.cafm(vis_feats, geo_embed)
        logits = self.classifier(fused_vector)
        return logits, raw_metrics, attn_maps

