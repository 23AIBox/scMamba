#!/bin/bash

# 定义包含数据集名字的数组
datasets=$(cat goldendatasets.txt)


# 遍历数组，执行Python脚本并传入每个数据集名作为参数
for dataset in $datasets
do
    echo $dataset
    python3 main.py --data_dir "$dataset"
done
