"""A3b — which ResNet freeze depth? Parameter count drives DP noise, so the
eps=inf winner may not be the DP winner. This sets up that comparison."""
import numpy as np
from scripts.a3_centralized import cross_validate

CONFIGS = [
    ("r18 all    (11.2M)", "resnet18", 224, False, dict(pretrained=True, trainable="all")),
    ("r18 layer4  (8.4M)", "resnet18", 224, False, dict(pretrained=True, trainable="layer4")),
    ("r18 layer4b (4.7M)", "resnet18", 224, False, dict(pretrained=True, trainable="layer4b")),
    ("r18 fc      (1.5K)", "resnet18", 224, False, dict(pretrained=True, trainable="fc")),
]

if __name__ == "__main__":
    print("A3b | freeze-depth sweep | 5-fold subject-grouped CV\n")
    res = {}
    for label, name, size, aug, kw in CONFIGS:
        print(label)
        res[label] = cross_validate(label, name, size, aug, **kw)
    print("=" * 60)
    for k, (a, s, f) in sorted(res.items(), key=lambda x: -x[1][0]):
        print(f"  {k:<22} {a:.4f} +- {s:.4f}  F1 {f:.4f}")
