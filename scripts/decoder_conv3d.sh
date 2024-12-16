#/bin/bash
num_latent_frames=19

for((i=0;i<19;i++));
do
echo "post quant conv"
python python/pth_conv3d_bench.py -bs=1 -oc=8 -ic=8 -f=2 -hi=72 -wi=52 -kf=1 -kh=1 -kw=1

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
echo "conv last"
python python/pth_conv3d_bench.py -bs=1 -oc=3 -ic=128 -f=7 -hi=578 -wi=418 -kf=3 -kh=3 -kw=3

echo "vae encoder"
python python/pth_conv3d_bench.py -bs=1 -oc=32 -ic=3 -f=3 -hi=578 -wi=418 -kf=3 -kh=3 -kw=3
echo "res block 1"
for((i=1;i<=8;i++));
do
python python/pth_conv3d_bench.py -bs=1 -oc=32 -ic=32 -f=3 -hi=578 -wi=418 -kf=3 -kh=3 -kw=3
done
echo "space down block 1"
python python/pth_conv3d_bench.py -bs=1 -oc=32 -ic=32 -f=3 -hi=578 -wi=418 -kf=3 -kh=3 -kw=3 -ts=1 -ss=2
echo "res block 2"
python python/pth_conv3d_bench.py -bs=1 -oc=64 -ic=32 -f=3 -hi=290 -wi=210 -kf=3 -kh=3 -kw=3
python python/pth_conv3d_bench.py -bs=1 -oc=64 -ic=64 -f=3 -hi=290 -wi=210 -kf=3 -kh=3 -kw=3
python python/pth_conv3d_bench.py -bs=1 -oc=64 -ic=32 -f=3 -hi=290 -wi=210 -kf=3 -kh=3 -kw=3
echo "res block 3"
for((i=1;i<=6;i++));
do
python python/pth_conv3d_bench.py -bs=1 -oc=64 -ic=64 -f=3 -hi=290 -wi=210 -kf=3 -kh=3 -kw=3
done
echo "space down block 2"
python python/pth_conv3d_bench.py -bs=1 -oc=64 -ic=64 -f=3 -hi=290 -wi=210 -kf=3 -kh=3 -kw=3 -ts=1 -ss=2
echo "res block 4"
for((i=1;i<=8;i++));
do
python python/pth_conv3d_bench.py -bs=1 -oc=64 -ic=64 -f=3 -hi=146 -wi=104 -kf=3 -kh=3 -kw=3
done
echo "space down block 3"
python python/pth_conv3d_bench.py -bs=1 -oc=64 -ic=64 -f=3 -hi=146 -wi=104 -kf=3 -kh=3 -kw=3 -ts=1 -ss=2
echo "res block 5"
python python/pth_conv3d_bench.py -bs=1 -oc=128 -ic=64 -f=3 -hi=74 -wi=54 -kf=3 -kh=3 -kw=3
python python/pth_conv3d_bench.py -bs=1 -oc=128 -ic=128 -f=3 -hi=74 -wi=54 -kf=3 -kh=3 -kw=3
python python/pth_conv3d_bench.py -bs=1 -oc=128 -ic=64 -f=3 -hi=74 -wi=54 -kf=3 -kh=3 -kw=3
echo "res block 6 and 7"
for((i=1;i<=14;i++))
do
python python/pth_conv3d_bench.py -bs=1 -oc=128 -ic=128 -f=3 -hi=74 -wi=54 -kf=3 -kh=3 -kw=3
done
echo "encoder last conv"
python python/pth_conv3d_bench.py -bs=1 -oc=8 -ic=128 -f=1 -hi=72 -wi=52 -kf=1 -kh=1 -kw=1
echo "encoder latent dist"
python python/pth_conv3d_bench.py -bs=1 -oc=16 -ic=8 -f=1 -hi=72 -wi=52 -kf=1 -kh=1 -kw=1
done


