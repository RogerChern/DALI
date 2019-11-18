import glob
import time
import random


filelist = glob.glob('/mnt/lustre/chenyuntao1/datasets/imagenet/train/*/*')
random.shuffle(filelist)

begin = time.time()
for i, f in enumerate(filelist):
    if i == 10000:
        break
    with open(f, "rb") as fin:
        result = fin.read()
end = time.time()

print("%.1f images/s" % (10000 / (end - begin)))