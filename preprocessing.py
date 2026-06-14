import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def preprocess_image(image_bytes: bytes):
    """
    Direct replication of FastDualStreamDataset.__getitem__ from the notebook.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    vis_tf = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((512, 512), dtype=np.uint8)
        
    c = clahe.apply(img)
    vis_img = cv2.cvtColor(cv2.resize(c, (224, 224), interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2RGB)
    vis = vis_tf(image=vis_img)['image']
    
    geo = torch.from_numpy(cv2.resize(c, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0).unsqueeze(0)
    
    # Add batch dimensions
    return vis.unsqueeze(0), geo.unsqueeze(0)
