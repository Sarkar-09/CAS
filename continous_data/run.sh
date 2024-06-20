# sample script for running the certification code
python3 certify.py --dataset cifar10 --checkpoint ../models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar --n_sample 10000 --batch_size 128 --smoothing_sigma 0.25 --wandb --wandb_project Robust_CP;
python3 certify.py --dataset imagenet --checkpoint ../models/imagenet/resnet50/noise_0.50/checkpoint.pth.tar --n_sample 10000 --batch_size 64 --smoothing_sigma 0.5 --skip 100 --wandb --wandb_project Robust_CP;

