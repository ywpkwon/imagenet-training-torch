import sys
sys.path.append('/nfs/fast-data/user/paul/projects/torchphantom')
sys.path.append('/nfs/fast-data/user/paul/projects/torchphantom/workspaces/yeonhwa/torchdev/tidl_phantomfront')

import torch
from torch import nn
import hydra
from omegaconf import DictConfig, OmegaConf
import fpnliteefficientnetv2
from fpnliteefficientnetv2 import get_fpnliteefficientnetv2


# @hydra.main(version_base=None, config_path='/nfs/fast-data/user/paul/projects/torchphantom/workspaces/yeonhwa/torchdev/tidl_phantomfront/config', config_name='config_r11')
# def get_model(config: DictConfig) -> fpnliteefficientnetv2.FpnLiteEfficientNetv2:
#     model = get_fpnliteefficientnetv2(pretrained=None, **config)
#     return model

class Encoder(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 dropout=0.5):
        super().__init__()

        config = OmegaConf.load('/nfs/fast-data/user/paul/projects/torchphantom/workspaces/yeonhwa/torchdev/tidl_phantomfront/config/config_r11.yaml')
        model = get_fpnliteefficientnetv2(pretrained=None, **config)

        self.backbone = model.backbone
        self.fpn = model.fpn
        self.avgpool = nn.AdaptiveAvgPool2d(8)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(20448, 1024),
            nn.SiLU(),
            nn.LayerNorm(1024),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)

        feats = []
        for k in sorted(x.keys()):
            y = x[k]
            _, _, w, h = y.shape
            assert w == h
            if w > 8:
                y = self.avgpool(y)
            feats.append(torch.flatten(y, 1))

        x = torch.concat(feats, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    config = OmegaConf.load('/nfs/fast-data/user/paul/projects/torchphantom/workspaces/yeonhwa/torchdev/tidl_phantomfront/config/config_r11.yaml')

    model = Encoder()
    x = torch.zeros((4, 3, 256, 256)).float()
    print(x.shape, ' -> ', model(x).shape)

    torch.save({'model_state_dict': model.state_dict()}, 'xx.pth')
