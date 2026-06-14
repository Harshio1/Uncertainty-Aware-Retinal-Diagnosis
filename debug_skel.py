import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from inference import ModelEngine
from preprocessing import preprocess_image

engine = ModelEngine('weights/final_patent_architecture.pth')

try:
    with open('skeleton_test.png', 'rb') as f:
        img_bytes = f.read()
except Exception:
    import urllib.request
    urllib.request.urlretrieve("https://huggingface.co/spaces/Harshio/adaptive-oct-classifier/resolve/main/test_images/CNV-1016042-1.jpeg", "skel.jpg")
    with open('skel.jpg', 'rb') as f:
        img_bytes = f.read()

img_vis, img_geo = preprocess_image(img_bytes)

# U-Net map
engine.model.unet.eval()
with torch.no_grad():
    vessel_map = engine.model.unet(img_geo.to(engine.device))

# Multi-threshold median
B = vessel_map.shape[0]
sm = F.interpolate(vessel_map, size=(128, 128), mode='bilinear', align_corners=False)

# Median aggregation is directly inside the DiagnosticMetricsExtractor,
# so let's simulate the loop up to skeletonization:
mask = torch.sigmoid(20 * (sm - 0.5))  # Middle threshold
binary = (mask > 0.5).float()
skeleton = engine.diag_extractor._skeletonize(binary, iters=25)

def save_t(t_tensor, name):
    t_np = t_tensor[0, 0].cpu().detach().numpy()
    if t_np.max() <= 1.0: t_np = t_np * 255.0
    t_img = Image.fromarray(t_np.astype(np.uint8)).convert('L')
    t_img.save(name)
    print('Saved:', name)

save_t(sm, 'debug_vessel_map.png')
save_t(binary, 'debug_mask.png')
save_t(skeleton, 'debug_skeleton.png')

print('\nMetrics Output Check:')
with torch.no_grad():
    res = engine.predict_single(img_vis, img_geo)
m = res['vascular_metrics']
print(f"branch_points: {m['branching_index'] * (m['branch_length'] + 1e-6):.4f}")
print(f"skeleton_length: {m['branch_length']:.4f}")
print(f"branching_index: {m['branching_index']:.6f}")
