## Long-Tailed Out-of-Distribution Detection via Normalized Outlier Distribution Adaptation

This is the official implementation of the [Long-Tailed Out-of-Distribution Detection via Normalized Outlier Distribution Adaptation](https://arxiv.org/abs/2410.20807) (NeurIPS2024)

## Dataset Preparation

### In-distribution dataset

Please download CIFAR10, CIFAR100, and [ImageNet-LT](https://liuziwei7.github.io/projects/LongTail.html) , place them  in`./dataset` 

### Auxiliary/Out-of-distribution dataset

For CIFAR10 and CIFAR100, please download [TinyImages 300K Random Images]() for auxiliary in `./dataset` 

For CIFAR10 and CIFAR100, please download [SC-OOD](https://jingkang50.github.io/projects/scood) benchmark  for out-of-distribution in `./dataset` 

For [ImageNet-LT](https://liuziwei7.github.io/projects/LongTail.html), please download [ImageNet10k_eccv2010](https://image-net.org/data/imagenet10k_eccv2010.tar) benchmark for auxiliary and out-of-distribution in `./dataset` 

All datasets follow [PASCL](https://github.com/amazon-science/long-tailed-ood-detection) and [COCL](https://github.com/mala-lab/COCL)

## Training

### CIFAR10-LT: 

```
python train.py --gpu 0 --ds cifar10 --drp <where_you_store_all_your_datasets> --srp <where_to_save_the_ckpt>
```

### CIFAR100-LT:

```
python train.py --gpu 0 --ds cifar100 --drp <where_you_store_all_your_datasets> --srp <where_to_save_the_ckpt>
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


## Acknowledgment

Part of our codes are adapted from these repos:

Outlier-Exposure - https://github.com/hendrycks/outlier-exposure - Apache-2.0 license

PASCL - https://github.com/amazon-science/long-tailed-ood-detection - Apache-2.0 license

COCL - https://github.com/mala-lab/COCL - Apache-2.0 license

BERL - https://github.com/hyunjunChhoi/Balanced_Energy - Apache-2.0 license

Long-Tailed-Recognition.pytorch - https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch - GPL-3.0 license

## License

This project is licensed under the Apache-2.0 License.
