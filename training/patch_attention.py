from pathlib import Path
import argparse
import re
import shutil
import ultralytics


FILE_PATH = Path(ultralytics.__file__).parent / "nn/modules/block.py"
BACKUP_PATH = FILE_PATH.with_suffix(FILE_PATH.suffix + ".backup")


NEW_CLASS_CODE = r"""
class Attention(nn.Module):
    # PSA-compatible linear attention (MobileViTv2 style)

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()

        self.W_i = nn.Conv2d(dim, 1, 1)
        self.W_k = nn.Conv2d(dim, dim, 1)
        self.W_v = nn.Conv2d(dim, dim, 1)

        self.proj = Conv(dim, dim, 1, act=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape

        cs = self.W_i(x)
        cs = cs.view(B, 1, -1)
        cs = torch.softmax(cs, dim=-1)

        k = self.W_k(x).view(B, C, -1)
        v = self.act(self.W_v(x)).view(B, C, -1)

        cv = (k * cs).sum(dim=-1, keepdim=True)

        out = v * cv
        out = out.view(B, C, H, W)

        return self.proj(out)
"""


def check_file():
    if not FILE_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {FILE_PATH}")


def backup_file():
    check_file()

    if BACKUP_PATH.exists():
        print(f"Backup đã tồn tại, bỏ qua:\n{BACKUP_PATH}")
        return

    shutil.copy2(FILE_PATH, BACKUP_PATH)
    print(f"Đã backup file gốc:\n{BACKUP_PATH}")


def replace_attention_class():
    check_file()

    content = FILE_PATH.read_text(encoding="utf-8")

    pattern = r"^class Attention\(.*?\):[\s\S]*?(?=^class |\Z)"

    if not re.search(pattern, content, flags=re.MULTILINE):
        raise RuntimeError("Không tìm thấy class Attention để thay thế.")

    new_content = re.sub(
        pattern,
        NEW_CLASS_CODE.strip() + "\n\n",
        content,
        count=1,
        flags=re.MULTILINE,
    )

    FILE_PATH.write_text(new_content, encoding="utf-8")
    print(f"Đã thay thế class Attention trong:\n{FILE_PATH}")


def restore_file():
    if not BACKUP_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file backup: {BACKUP_PATH}")

    shutil.copy2(BACKUP_PATH, FILE_PATH)
    print(f"Đã khôi phục file gốc:\n{FILE_PATH}")


def patch():
    backup_file()
    replace_attention_class()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["backup", "replace", "restore", "patch"],
        help="backup | replace | restore | patch",
    )

    args = parser.parse_args()

    if args.action == "backup":
        backup_file()
    elif args.action == "replace":
        replace_attention_class()
    elif args.action == "restore":
        restore_file()
    elif args.action == "patch":
        patch()


if __name__ == "__main__":
    main()