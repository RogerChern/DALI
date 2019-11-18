import argparse
import os
import pyarrow as pa
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, required=True)
args = parser.parse_args()

split = args.split
data_dir = "/mnt/lustre/chenyuntao1/datasets/imagenet/{}".format(split)
file_list = []

with open("/mnt/lustre/chenyuntao1/datasets/imagenet/rec/{}.lst.col3".format(split)) as fin:
    for line in fin:
        file_list.append(line.strip())

bytes_dict = dict()
for i, line in enumerate(file_list):
    filename = os.path.join(data_dir, line)
    with open(filename, "rb") as fin:
        bytes_dict[line] = np.frombuffer(fin.read(), dtype=np.uint8)
    
    if i % 1000 == 999 or i == len(file_list) - 1:
        with open("/mnt/lustre/chenyuntao1/datasets/imagenet/{}_splits/{}.pa".format(split, (i // 1000)), "wb") as fout:
            pa.serialize_to(bytes_dict, fout)
        bytes_dict.clear()