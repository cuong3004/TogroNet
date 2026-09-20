import torch.nn as nn
from ultralytics import YOLO

# Backbone cua ho yolo26 luon ket thuc bang C2PSA, nhung SO LAYER TRUOC DO
# khac nhau giua cac bien the: yaml/checkpoint "detect" co SPPF ngay truoc
# C2PSA (Conv,Conv,C3k2,Conv,C3k2,Conv,C3k2,Conv,C3k2,SPPF,C2PSA -- 11 layer,
# C2PSA o index 10), con yaml/checkpoint "classify" (vd yolo26n-cls.yaml)
# KHONG co SPPF (... C3k2,C2PSA -- 10 layer, C2PSA o index 9). Vi vay khong
# hardcode so layer co dinh (se sai lech 1 layer tuy loai checkpoint), ma tu
# do tim index cua C2PSA cuoi cung de lam ranh gioi backbone/neck-head.


def find_backbone_end(model_seq):
    for i, m in enumerate(model_seq):
        if "C2PSA" in m.__class__.__name__:
            return i

    raise RuntimeError(
        "Khong tim thay C2PSA de xac dinh ranh gioi backbone -- "
        "kien truc nay co the khac ho yolo26, can kiem tra lai."
    )


class YOLOBackboneEncoder(nn.Module):
    def __init__(self, weight="last.pt"):
        super().__init__()

        yolo = YOLO(weight)

        # Giu nguyen yolo.model (khong slice/wrap lai) de state_dict giu dung
        # tien to "encoder.model.model.X..." nhu quy uoc cu, tuong thich voi
        # convert_ssl_ckpt_to_yolo.py. Viec chi chay qua backbone nam o
        # forward() ben duoi, khong anh huong cach luu tham so.
        self.model = yolo.model
        self.backbone_end = find_backbone_end(self.model.model)

        # Weight da qua strip_optimizer() (moi lan train ket thuc) bi set
        # requires_grad=False vi muc dich goc la inference/export -- bat lai
        # o day de SSL train duoc, khong chi la doi mode BN/Dropout.
        for p in self.model.parameters():
            p.requires_grad = True

        self.model.train()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        feat = x
        for layer in self.model.model[: self.backbone_end + 1]:
            feat = layer(feat)

        out = self.pool(feat)
        out = self.flatten(out)

        return out


def build_encoder(weight="last.pt"):
    return YOLOBackboneEncoder(weight)
