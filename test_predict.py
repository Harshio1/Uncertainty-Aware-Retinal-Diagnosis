import torch
from inference import ModelEngine

engine = ModelEngine('weights/final_patent_architecture.pth')
vis = torch.randn(1, 3, 224, 224)
geo = torch.randn(1, 1, 512, 512)

print("\n--- INFERENCE PREDICTION TEST ---")
res = engine.predict_single(vis, geo)
metrics = res['vascular_metrics']

bl = metrics['branch_length']
bi = metrics['branching_index']
bp = bi * (bl + 1e-6)

print(f"branch_points: {bp:.4f}")
print(f"skeleton_length: {bl:.4f}")
print(f"branching_index: {bi:.6f}")
