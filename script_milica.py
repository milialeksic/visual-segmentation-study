import torch
from dlvc.metrics import SegMetrics  # or wherever your class is saved

pred = torch.tensor([[
    [[0.9, 0.1], [0.0, 0.2]],  # class 0
    [[0.1, 0.8], [0.6, 0.1]],  # class 1
    [[0.0, 0.1], [0.4, 0.7]]   # class 2
]])  # shape: (1, 3, 2, 2)

# Ground truth: shape (b, h, w)
# True labels for each pixel
target = torch.tensor([
    [[0, 1],
     [2, 255]]  # last pixel is ignored
])

metric = SegMetrics(classes=3)
metric.reset()
metric.update(pred, target)
print(metric.conf_matrix)

print(metric)           # calls __str__ → mIoU: 0.xx
print(metric.mIoU()) 