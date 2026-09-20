import torch
from ultralytics import YOLO
from pathlib import Path


WEIGHT_PATH = "yolo26n.pt"


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def inspect_raw_pt(weight_path):
    print_section("RAW .pt FILE")

    ckpt = torch.load(
        weight_path,
        map_location="cpu",
        weights_only=False,
    )

    print("Type:", type(ckpt))

    if isinstance(ckpt, dict):
        print("Top-level keys:")
        for k in ckpt.keys():
            print("  -", k)

    return ckpt


def inspect_ultralytics_model(weight_path):
    print_section("ULTRALYTICS MODEL")

    yolo = YOLO(weight_path)
    model = yolo.model

    print("Model type:", type(model))
    print("Model class:", model.__class__.__name__)

    print("\nModel yaml keys:")
    if hasattr(model, "yaml"):
        for k in model.yaml.keys():
            print("  -", k)

    print("\nNumber of layers:", len(model.model))

    print("\nLayers:")
    for i, layer in enumerate(model.model):
        print(f"{i:03d} | {layer.__class__.__name__}")

    return model


def inspect_state_dict(model):
    print_section("MODEL STATE_DICT")

    sd = model.state_dict()

    print("Number of tensors:", len(sd))

    for name, tensor in sd.items():
        print(f"{name:80s} {tuple(tensor.shape)}")

    return sd


def inspect_backbone(model):
    print_section("BACKBONE LAYERS")

    n_backbone = len(model.yaml["backbone"])

    print("n_backbone =", n_backbone)

    for i in range(n_backbone):
        layer = model.model[i]
        print(f"{i:03d} | {layer.__class__.__name__}")

    print("\nBackbone state_dict keys:")

    for name, tensor in model.state_dict().items():
        layer_id = name.split(".")[1] if name.startswith("model.") else None

        if layer_id is not None and layer_id.isdigit():
            if int(layer_id) < n_backbone:
                print(f"{name:80s} {tuple(tensor.shape)}")


def inspect_detect_input_shapes(model):
    print_section("DETECT INPUT FEATURE SHAPES")

    detect = model.model[-1]
    feats = {}

    def hook(module, inputs):
        feats["x"] = inputs[0]

    handle = detect.register_forward_pre_hook(hook)

    x = torch.randn(1, 3, 640, 640)

    model.eval()
    with torch.no_grad():
        _ = model(x)

    handle.remove()

    xs = feats["x"]

    for i, f in enumerate(xs):
        print(f"P{i+3}: {tuple(f.shape)}")


def main():
    weight_path = Path(WEIGHT_PATH)

    if not weight_path.exists():
        raise FileNotFoundError(weight_path)

    inspect_raw_pt(weight_path)

    model = inspect_ultralytics_model(weight_path)

    inspect_state_dict(model)

    inspect_backbone(model)

    inspect_detect_input_shapes(model)


if __name__ == "__main__":
    main()