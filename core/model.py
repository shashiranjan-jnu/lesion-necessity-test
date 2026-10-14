"""Models. Batch normalization is replaced with group normalization so that the
same networks remain usable with per-example gradients (differentially private
training, Opacus). Every model exposes a named final conv layer for Grad-CAM.

in_ch added for the HAM10000 arm: dermoscopy is RGB and colour is diagnostic,
so in_ch=3 keeps the pretrained ResNet stem untouched instead of averaging it
to one channel. Defaults are unchanged (in_ch=1, N_CLASSES=3), so every
brain-collection script keeps its current behaviour.
"""
import torch
import torch.nn as nn

def _patch_resnet_inplace():
    """torchvision BasicBlock does `out += identity`, which is in-place.
    Opacus backward hooks return views, and in-place modification of a view
    from a custom Function is forbidden. Rebind forward to the out-of-place
    equivalent. Mathematically identical; only autograd bookkeeping differs."""
    from torchvision.models.resnet import BasicBlock

    if getattr(BasicBlock, "_opacus_patched", False):
        return

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity          # was: out += identity
        return self.relu(out)

    BasicBlock.forward = forward
    BasicBlock._opacus_patched = True


N_CLASSES = 3


class SmallCNN(nn.Module):
    """From-scratch CNN. GroupNorm throughout."""

    def __init__(self, n_classes=N_CLASSES, width=32, in_size=128, in_ch=1):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, padding=1), nn.GroupNorm(8, w),     nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(w, w*2, 3, padding=1),   nn.GroupNorm(8, w*2),   nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(w*2, w*4, 3, padding=1), nn.GroupNorm(8, w*4),   nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.last_conv = nn.Conv2d(w*4, w*4, 3, padding=1)   # named for Grad-CAM
        self.post = nn.Sequential(nn.GroupNorm(8, w*4), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(w*4, n_classes))

    def forward(self, x):
        return self.head(self.post(self.last_conv(self.features(x))))


def resnet18_gn(n_classes=N_CLASSES, pretrained=True, trainable="layer4", in_ch=1):
    """ImageNet ResNet-18, BatchNorm -> GroupNorm.

    trainable: which blocks stay trainable. Everything earlier is frozen.
      "all"     -> full fine-tune            (~11.2M)
      "layer4"  -> layer4 + fc               (~8.4M)
      "layer4b" -> second half of layer4 +fc (~4.7M)
      "fc"      -> classifier only           (~1.5K)
    NOTE: the BN->GN swap discards pretrained BN statistics, so some transfer
    benefit is lost. How much is empirical.

    in_ch=1 replaces the stem with the channel mean of the pretrained filters
    (grayscale MRI). in_ch=3 leaves the pretrained RGB stem in place. Any other
    value gets a fresh stem initialised by repeating the channel mean, which
    loses the pretrained first layer -- avoid unless you mean it.
    """
    from torchvision.models import resnet18, ResNet18_Weights
    from opacus.validators import ModuleValidator

    _patch_resnet_inplace()

    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    if in_ch == 1:
        w = m.conv1.weight.data.mean(dim=1, keepdim=True)  # RGB -> grayscale
        m.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        m.conv1.weight.data = w
    elif in_ch != 3:
        w = m.conv1.weight.data.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1)
        m.conv1 = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
        m.conv1.weight.data = w
    m.fc = nn.Linear(512, n_classes)
    m = ModuleValidator.fix(m)

    # Opacus hooks return views; in-place ReLU corrupts them
    for mod in m.modules():
        if isinstance(mod, nn.ReLU):
            mod.inplace = False

    if trainable != "all":
        keep = {"layer4": ("layer4", "fc"),
                "layer4b": ("layer4.1", "fc"),
                "fc": ("fc",)}[trainable]
        for name, p in m.named_parameters():
            p.requires_grad = any(name.startswith(k) for k in keep)
    return m


def build(name: str, **kw) -> nn.Module:
    if name == "smallcnn":
        return SmallCNN(**kw)
    if name == "resnet18":
        return resnet18_gn(**kw)
    raise ValueError(name)


def assert_dp_safe(model: nn.Module):
    from opacus.validators import ModuleValidator
    errs = ModuleValidator.validate(model, strict=False)
    assert not errs, f"model not DP-compatible: {errs[:3]}"
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return {"trainable": tr, "total": tot}
