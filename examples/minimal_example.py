import torch
import rectiou

# Rectangles: [x, y, w, h, radians]
rect_a = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])
rect_b = torch.tensor([0.2, 0.1, 1.1, 0.8, torch.pi / 4])

# Intersection over union
iou = rectiou.compute_iou(rect_a, rect_b)
