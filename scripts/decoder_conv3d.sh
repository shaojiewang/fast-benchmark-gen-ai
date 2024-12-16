#/bin/bash
CUDA_VISIBLE_DEVICES=2
apt install bc
num_latent_frames=1
time_total=0.0
for((i=0;i<${num_latent_frames};i++));
do
echo "post quant conv"
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=8 -ic=8 -f=2 -hi=72 -wi=52 -kf=1 -kh=1 -kw=1 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)

echo "vae decoder"
echo "res block 1 and 2"
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=512 -ic=8 -f=4 -hi=74 -wi=54 -kf=3 -kh=3 -kw=3  | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
for((i=1;i<=16;i++));
do
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=512 -ic=512 -f=4 -hi=74 -wi=54 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
done
echo $time_total
echo "3d transpose conv1"
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=4096 -ic=512 -f=4 -hi=74 -wi=54 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
echo "res block 3"
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=256 -ic=512 -f=5 -hi=146 -wi=106 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=256 -ic=256 -f=5 -hi=146 -wi=106 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=256 -ic=512 -f=5 -hi=146 -wi=106 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
echo "res block 4"
for((i=1;i<=6;i++));
do
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=256 -ic=256 -f=5 -hi=146 -wi=106 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
done
echo $time_total
echo "3d transpose conv2"
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=2048 -ic=256 -f=5 -hi=146 -wi=106 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
echo "res block 5"
for((i=1;i<=8;i++));
do
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=256 -ic=256 -f=7 -hi=290 -wi=210 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
done
echo $time_total
echo "2d transpose conv1"
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=1024 -ic=256 -f=7 -hi=290 -wi=210 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
echo "res block 6"
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=128 -ic=256 -f=7 -hi=578 -wi=418 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=128 -ic=128 -f=7 -hi=578 -wi=418 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=128 -ic=256 -f=7 -hi=578 -wi=418 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
echo $time_total
echo "res block 7"
for((i=1;i<=6;i++));
do
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=128 -ic=128 -f=7 -hi=578 -wi=418 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
done
echo $time_total
echo "conv last"
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=3 -ic=128 -f=7 -hi=578 -wi=418 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
echo $time_total

echo "vae encoder"
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=32 -ic=3 -f=3 -hi=578 -wi=418 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
echo "res block 1"
for((i=1;i<=8;i++));
do
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=32 -ic=32 -f=3 -hi=578 -wi=418 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
done
echo "space down block 1"
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=32 -ic=32 -f=3 -hi=578 -wi=418 -kf=3 -kh=3 -kw=3 -ts=1 -ss=2 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
echo "res block 2"
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=64 -ic=32 -f=3 -hi=290 -wi=210 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=64 -ic=64 -f=3 -hi=290 -wi=210 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=64 -ic=32 -f=3 -hi=290 -wi=210 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
echo "res block 3"
for((i=1;i<=6;i++));
do
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=64 -ic=64 -f=3 -hi=290 -wi=210 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
done
echo "space down block 2"
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=64 -ic=64 -f=3 -hi=290 -wi=210 -kf=3 -kh=3 -kw=3 -ts=1 -ss=2 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
echo "res block 4"
for((i=1;i<=8;i++));
do
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=64 -ic=64 -f=3 -hi=146 -wi=104 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
done
echo "space down block 3"
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=64 -ic=64 -f=3 -hi=146 -wi=104 -kf=3 -kh=3 -kw=3 -ts=1 -ss=2 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
echo "res block 5"
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=128 -ic=64 -f=3 -hi=74 -wi=54 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=128 -ic=128 -f=3 -hi=74 -wi=54 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=128 -ic=64 -f=3 -hi=74 -wi=54 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
echo "res block 6 and 7"
for((i=1;i<=14;i++))
do
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=128 -ic=128 -f=3 -hi=74 -wi=54 -kf=3 -kh=3 -kw=3 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
done
echo "encoder last conv"
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=8 -ic=128 -f=1 -hi=72 -wi=52 -kf=1 -kh=1 -kw=1 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
echo "encoder latent dist"
time_ms=$(python python/pth_conv3d_bench.py -bs=1 -oc=16 -ic=8 -f=1 -hi=72 -wi=52 -kf=1 -kh=1 -kw=1 | awk -F'=|ms' '{print $5}')
time_total=$(echo "$time_total + $time_ms" | bc -l)
echo $time_total
done


