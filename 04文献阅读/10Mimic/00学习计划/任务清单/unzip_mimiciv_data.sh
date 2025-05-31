#!/bin/bash

# 解压 hosp 数据
mkdir -p /workspace/mimiciv/3.1/hosp
for f in /workspace/physionet.org/files/mimiciv/3.1/hosp/*.csv.gz; do
  gzip -dkc "$f" > /workspace/mimiciv/3.1/hosp/$(basename "${f%.gz}")
done

# 解压 icu 数据
mkdir -p /workspace/mimiciv/3.1/icu
for f in /workspace/physionet.org/files/mimiciv/3.1/icu/*.csv.gz; do
  gzip -dkc "$f" > /workspace/mimiciv/3.1/icu/$(basename "${f%.gz}")
done

echo "✅ 解压完成，原始文件已保留。"
