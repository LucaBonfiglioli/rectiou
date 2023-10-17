# rectiou

Rectiou is a pytorch implementation of the intersection over union (IoU) between two rotated rectangles.

Supports:
- ✅ Works with rotated rectangles: can handle rectangles with any angle.
- ✅ Batched computation: No for loops, if statements, or other control flow. Can have as many batch dimensions as you want.
- ✅ Fast: faster than OpenCV's implementation.
- ✅ Differentiable: Can be used in a pytorch model and backpropagated through.
- ✅ GPU: can be run on the GPU.
- ✅ No weird stuff: everything is implemented in plain pytorch, no weird custom CUDA kernels or anything like that.

## Example

Usage is simple:
```python

import torch
import rectiou

# Rectangles: [x, y, w, h, radians]
rect_a = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])
rect_b = torch.tensor([0.2, 0.1, 1.1, 0.8, torch.pi / 4])

# Intersection over union
iou = rectiou.compute_iou(rect_a, rect_b)
```

## Installation

```bash
pip install rectiou
```

To test that everything is working correctly, run:
```bash
pip install rectiou[tests]
pytest
```