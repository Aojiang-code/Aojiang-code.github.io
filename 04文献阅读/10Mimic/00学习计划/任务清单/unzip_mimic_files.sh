#!/bin/bash

# 解压 hosp 文件夹下所有 .csv.gz 文件
echo "解压 hosp 文件夹中的文件..."
gzip -dk /workspace/physionet.org/files/mimiciv/3.1/hosp/*.csv.gz

# 解压 icu 文件夹下所有 .csv.gz 文件
echo "解压 icu 文件夹中的文件..."
gzip -dk /workspace/physionet.org/files/mimiciv/3.1/icu/*.csv.gz

echo "所有文件解压完成。"
