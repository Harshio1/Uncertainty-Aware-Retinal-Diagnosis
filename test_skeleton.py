import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model import RobustVascularMetricsExtractor

def create_dummy_vessel(size=64):
    # Create a Y-shaped vessel
    mask = torch.zeros((1, 1, size, size))
    # Vertical trunk
    mask[0, 0, 10:40, 31:34] = 1.0
    # Left branch
    for i in range(15):
        mask[0, 0, 40+i, 31-i:34-i] = 1.0
    # Right branch
    for i in range(15):
        mask[0, 0, 40+i, 31+i:34+i] = 1.0
        
    # Add another intersection
    mask[0, 0, 25:28, 10:50] = 1.0
    return mask

def test():
    extractor = RobustVascularMetricsExtractor()
    mask = create_dummy_vessel()
    
    def gpu_skeletonize_fixed(mask, iterations=20):
        thin = mask.clone()
        neighbor_kernel = extractor.neighbor_kernel
        cyclic_idx = [0, 1, 2, 5, 8, 7, 6, 3] # Row-major 3x3 cyclic ordering
        
        for _ in range(iterations):
            neighbors = F.conv2d(thin, neighbor_kernel, padding=1) * thin
            eroded = 1.0 - F.max_pool2d(1.0 - thin, kernel_size=3, stride=1, padding=1)
            boundary = ((thin - eroded) > 0.5).float()
            
            # Crossing number computation
            patches = F.unfold(thin, kernel_size=3, padding=1) # [B, 9, H*W]
            p = patches[:, cyclic_idx, :]
            p_shift = torch.roll(p, shifts=-1, dims=1)
            transitions = torch.abs(p - p_shift).sum(dim=1) / 2.0
            transitions = transitions.view(thin.shape)
            
            # A pixel is removable if it's on boundary, has 2 to 6 neighbors, AND has exactly 1 transition
            removable = boundary * (neighbors >= 2).float() * (neighbors <= 6).float() * (transitions == 1.0).float()
            
            thin = thin * (1.0 - removable)
            thin = (thin > 0.5).float()
            if removable.sum() == 0:
                break
        return thin
        
    skel = gpu_skeletonize_fixed(mask, iterations=8)
    
    # 2. Get neighbor counts
    sn = F.conv2d(skel, extractor.neighbor_kernel, padding=1) * skel
    
    # Print stats
    print("Mask sum:", mask.sum().item())
    print("Skeleton sum:", skel.sum().item())
    print("Max neighbors in skeleton:", sn.max().item())
    unique, counts = torch.unique(sn, return_counts=True)
    print("Neighbor distribution:")
    for u, c in zip(unique.tolist(), counts.tolist()):
        print(f"  {u}: {c}")
        
    branching = (sn > 2.0).float().view(1, -1).sum(dim=1).item()
    branching_eq_3 = (sn == 3.0).float().view(1, -1).sum(dim=1).item()
    print(f"Branching index (>2): {branching}")
    print(f"Branching index (==3): {branching_eq_3}")
    
    # 3. Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mask[0,0].numpy(), cmap='gray')
    axes[0].set_title("Original Mask")
    
    axes[1].imshow(skel[0,0].numpy(), cmap='gray')
    axes[1].set_title("Skeleton")
    
    branch_nodes = (sn > 2.0).float()
    
    axes[2].imshow(skel[0,0].numpy(), cmap='gray', alpha=0.5)
    axes[2].imshow(branch_nodes[0,0].numpy(), cmap='Reds', alpha=0.5)
    axes[2].set_title("Branch Nodes (>2 neighbors)")
    
    plt.savefig('skeleton_test.png')
    print("Saved skeleton_test.png visualization.")

if __name__ == "__main__":
    test()
