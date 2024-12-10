#/bin/bash

python python/pth_conv3d_bench.py -bs=1 -oc=512 -ic=512 -f=4 -hi=74 -wi=54 -kf=3 -kh=3 -kw=3
python python/pth_conv3d_bench.py -bs=1 -oc=32 -ic=32 -f=3 -hi=576 -wi=416 -kf=3 -kh=3 -kw=3



