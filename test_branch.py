import torch
from model import DiagnosticMetricsExtractor

extractor = DiagnosticMetricsExtractor()
extractor.eval()

# Fake probability map
vessel_map = torch.zeros(1, 1, 512, 512)
vessel_map[0, 0, 256, 100:400] = 1.0 # Horizontal
vessel_map[0, 0, 256:400, 250] = 1.0 # Vertical

metrics_np = extractor(vessel_map)
branching_index = metrics_np[0, 5]
branch_len = metrics_np[0, 7]
branch_points = branching_index * (branch_len + 1e-6)

print("branch_points:", branch_points)
print("skeleton_length:", branch_len)
print("branching_index:", branching_index)
