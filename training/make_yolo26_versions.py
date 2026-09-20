from pathlib import Path
import copy
import yaml


BASE_YAML = Path("/home/agi/thesis_code/togrow2.yaml")
OUT_DIR = Path("yolo26_versions")


VERSIONS = {
    "m2_width_0225": {
        "depth": 0.50,
        "width": 0.225,
        "max_channels": 1024,
        "note": "Width reduction I",
    },

    "m3_width_0200": {
        "depth": 0.50,
        "width": 0.200,
        "max_channels": 1024,
        "note": "Width reduction II",
    },

    "m4_width_0175": {
        "depth": 0.50,
        "width": 0.175,
        "max_channels": 1024,
        "note": "Width reduction III",
    },

    "m5_depth_045_width_0200": {
        "depth": 0.45,
        "width": 0.200,
        "max_channels": 1024,
        "note": "Depth reduction I",
    },

    "m6_depth_040_width_0200": {
        "depth": 0.40,
        "width": 0.200,
        "max_channels": 1024,
        "note": "Depth reduction II",
    },
}


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def make_version(base_cfg, version_name, cfg):
    new_cfg = copy.deepcopy(base_cfg)

    depth = cfg["depth"]
    width = cfg["width"]
    max_channels = cfg["max_channels"]

    # Chỉ sửa scale n
    new_cfg["scales"]["n"] = [
        depth,
        width,
        max_channels,
    ]

    # Ghi thêm metadata để sau này nhìn file biết nó là gì
    new_cfg["experiment"] = {
        "name": version_name,
        "note": cfg["note"],
        "depth": depth,
        "width": width,
        "max_channels": max_channels,
    }

    return new_cfg


def main():
    if not BASE_YAML.exists():
        raise FileNotFoundError(f"Cannot find {BASE_YAML}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(BASE_YAML)

    for name, cfg in VERSIONS.items():
        new_cfg = make_version(base_cfg, name, cfg)

        out_path = OUT_DIR / f"togroth_{name}.yaml"
        save_yaml(new_cfg, out_path)

        print(f"Created: {out_path}")


if __name__ == "__main__":
    main()