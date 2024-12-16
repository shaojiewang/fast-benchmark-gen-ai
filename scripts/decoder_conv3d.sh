#/bin/bash

echo "post quant conv"
python python/pth_conv3d_bench.py -bs=1 -oc=8 -ic=8 -f=2 -hi=74 -wi=54 -kf=1 -kh=1 -kw=1

echo "vae decoder"
echo "res block 1 and 2"
python python/pth_conv3d_bench.py -bs=1 -oc=512 -ic=8 -f=4 -hi=74 -wi=54 -kf=3 -kh=3 -kw=3
for((i=1;i<=16;i++));
do
python python/pth_conv3d_bench.py -bs=1 -oc=512 -ic=512 -f=4 -hi=74 -wi=54 -kf=3 -kh=3 -kw=3
done
echo "3d transpose conv1"
python python/pth_conv3d_bench.py -bs=1 -oc=4096 -ic=512 -f=4 -hi=74 -wi=54 -kf=3 -kh=3 -kw=3
echo "res block 3"
python python/pth_conv3d_bench.py -bs=1 -oc=256 -ic=512 -f=5 -hi=146 -wi=106 -kf=3 -kh=3 -kw=3
python python/pth_conv3d_bench.py -bs=1 -oc=256 -ic=256 -f=5 -hi=146 -wi=106 -kf=3 -kh=3 -kw=3
python python/pth_conv3d_bench.py -bs=1 -oc=256 -ic=512 -f=5 -hi=146 -wi=106 -kf=3 -kh=3 -kw=3
echo "res block 4"
for((i=1;i<=6;i++));
do
python python/pth_conv3d_bench.py -bs=1 -oc=256 -ic=256 -f=5 -hi=146 -wi=106 -kf=3 -kh=3 -kw=3
done
echo "3d transpose conv2"
python python/pth_conv3d_bench.py -bs=1 -oc=2048 -ic=256 -f=5 -hi=146 -wi=106 -kf=3 -kh=3 -kw=3
echo "res block 5"
for((i=1;i<=8;i++));
do
python python/pth_conv3d_bench.py -bs=1 -oc=256 -ic=256 -f=7 -hi=290 -wi=210 -kf=3 -kh=3 -kw=3
done
echo "2d transpose conv1"
python python/pth_conv3d_bench.py -bs=1 -oc=1024 -ic=256 -f=7 -hi=290 -wi=210 -kf=3 -kh=3 -kw=3
echo "res block 6"
python python/pth_conv3d_bench.py -bs=1 -oc=128 -ic=256 -f=7 -hi=578 -wi=418 -kf=3 -kh=3 -kw=3
python python/pth_conv3d_bench.py -bs=1 -oc=128 -ic=128 -f=7 -hi=578 -wi=418 -kf=3 -kh=3 -kw=3
python python/pth_conv3d_bench.py -bs=1 -oc=128 -ic=256 -f=7 -hi=578 -wi=418 -kf=3 -kh=3 -kw=3
echo "res block 7"
for((i=1;i<=6;i++));
do
python python/pth_conv3d_bench.py -bs=1 -oc=128 -ic=128 -f=7 -hi=578 -wi=418 -kf=3 -kh=3 -kw=3
done


