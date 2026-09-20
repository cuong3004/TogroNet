#!/usr/bin/env bash
# Tai (co the RESUME) cac file annotation (bounding box) cua Open Images V7.
# KHONG tai anh that - chi tai file CSV annotation (~2.3GB tong cong cho ca
# 3 split train/validation/test), sau do loc lai chi giu cac dong thuoc 10
# lop can quan tam trong count_openimages_classes.py, roi XOA file tho de
# tiet kiem dia.
#
# An toan khi tat may / mat mang giua chung: chay lai script nay, curl se
# tu dong resume tu vi tri da tai (nho "-C -"), khong tai lai tu dau.
#
# Cach dung:
#   bash download_openimages_annotations.sh
#   python3 count_openimages_classes.py

set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR="openimages_stats_data"
mkdir -p "$DATA_DIR"

MID_FILE="$DATA_DIR/target_mids.txt"
cat > "$MID_FILE" <<'EOF'
/m/07j87
/m/05s2s
/m/03fp41
/m/0c9ph5
/m/0fm3zh
/m/07j7r
/m/0f4s2w
/m/02xwb
/m/0jg57
/m/015x4r
EOF

declare -A URLS=(
  [train]="https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
  [validation]="https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
  [test]="https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv"
)

for split in train validation test; do
  filtered="$DATA_DIR/bbox_filtered_${split}.csv"
  if [ -f "$filtered" ]; then
    echo "[$split] Da co file loc san ($filtered), bo qua."
    continue
  fi

  raw="$DATA_DIR/raw_${split}_bbox.csv"
  url="${URLS[$split]}"
  echo "[$split] Tai (resume neu da tai do dang) tu $url ..."
  curl -f -C - -o "$raw" "$url"

  echo "[$split] Tai xong, dang loc theo 10 nhan can thiet ..."
  awk -F',' -v midfile="$MID_FILE" -v sp="$split" '
    BEGIN{ while((getline line < midfile) > 0) mids[line] = 1 }
    NR == 1 { print "Split," $0; next }
    ($3 in mids) { print sp "," $0 }
  ' "$raw" > "$filtered"

  rm -f "$raw"
  echo "[$split] Xong -> $filtered ($(wc -l < "$filtered") dong), da xoa file tho de tiet kiem dia."
done

echo ""
echo "Hoan tat tai + loc. Chay tiep: python3 count_openimages_classes.py"
