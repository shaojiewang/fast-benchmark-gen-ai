#/bin/bash

echo "post quant conv"
python python/pth_conv3d_bench.py -bs=1 -oc=8 -ic=8 -f=2 -hi=74 -wi=54 -kf=1 -kh=1 -kw=1

echo "decoder conv"
python python/pth_conv3d_bench.py -bs=1 -oc=512 -ic=8 -f=4 -hi=74 -wi=54 -kf=3 -kh=3 -kw=3
for((i=1;i<=16;i++));
do
python python/pth_conv3d_bench.py -bs=1 -oc=512 -ic=512 -f=4 -hi=74 -wi=54 -kf=3 -kh=3 -kw=3
done


