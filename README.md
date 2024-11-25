# Long-Tailed Out-of-Distribution Detection via Normalized Outlier Distribution Adaptation (AdaptOD)

This is the official implementation of the [Long-Tailed Out-of-Distribution Detection via Normalized Outlier Distribution Adaptation]() NeurIPS2024

## Dataset Preparation

### In-distribution dataset

Please download [CIFAR10](), [CIFAR100](), and [ImageNet-LT](https://liuziwei7.github.io/projects/LongTail.html) , place them  in`./datasets` 

### Auxiliary/Out-of-distribution dataset

For [CIFAR10-LT]() and [CIFAR100-LT](), please download [TinyImages 300K Random Images]() for auxiliary in `./datasets` 

For [CIFAR10-LT]() and [CIFAR100-LT](), please download [SC-OOD](https://jingkang50.github.io/projects/scood) benchmark  for out-of-distribution in `./datasets` 

For [ImageNet-LT](https://liuziwei7.github.io/projects/LongTail.html), please download [ImageNet10k_eccv2010](https://image-net.org/data/imagenet10k_eccv2010.tar) benchmark for auxiliary and out-of-distribution in `./datasets` 

All datasets follow [PASCL](https://github.com/amazon-science/long-tailed-ood-detection) and [COCL](https://github.com/mala-lab/COCL)

### Pretrained model

please save in `./pretrain`

## Training

### CIFAR10-LT: 

```
python train.py --gpu 0 --ds cifar10 --drp <where_you_store_all_your_datasets> --srp <where_to_save_the_ckpt>
```

### CIFAR100-LT:

```
python train.py --gpu 0 --ds cifar100 --drp <where_you_store_all_your_datasets> --srp <where_to_save_the_ckpt>
```

### ImageNet-LT:

```
python stage1.py --gpu 0,1,2,3 --ds imagenet --md ResNet50 --lr 0.01 --wd 5e-3  --drp <where_you_store_all_your_datasets> --srp <where_to_save_the_ckpt>
```

## Testing

### CIFAR10-LT:

```
python test.py --gpu 0 --ds cifar10 --drp <where_you_store_all_your_datasets> --ckpt_path <where_you_save_the_ckpt>
```

### CIFAR100-LT:

```
python test.py --gpu 0 --ds cifar100 --drp <where_you_store_all_your_datasets> --ckpt_path <where_you_save_the_ckpt>
```

### ImageNet-LT:

```
python test_imagenet.py --gpu 0  --drp <where_you_store_all_your_datasets> --ckpt_path <where_you_save_the_ckpt>
```


## Acknowledgment

Part of our codes are adapted from these repos:

Outlier-Exposure - https://github.com/hendrycks/outlier-exposure - Apache-2.0 license

PASCL - https://github.com/amazon-science/long-tailed-ood-detection - Apache-2.0 license

COCL - https://github.com/mala-lab/COCL - Apache-2.0 license

BERL - https://github.com/hyunjunChhoi/Balanced_Energy - Apache-2.0 license

Long-Tailed-Recognition.pytorch - https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch - GPL-3.0 license

## License

This project is licensed under the Apache-2.0 License.
