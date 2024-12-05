'''
Codes adapted from https://github.com/hendrycks/outlier-exposure/blob/master/CIFAR/test.py
which uses Apache-2.0 license.
'''
import os, argparse, time
from contextlib import ExitStack
from torch.utils.data import DataLoader, Subset
from functools import partial
import numpy as np 
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100
from dataset.SCOODBenchmarkDataset import SCOODDataset
from dataset.tinyimages_300k import TinyImages
from models.network_arch_resnet import ResnetEncoder
from skimage.filters import gaussian as gblur

from utils.utils import *
from utils.ltr_metrics import *

from scipy import stats
import random
import math

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1
    auroc = roc_auc_score(labels, examples)
    aupr_in = average_precision_score(labels, examples)
    labels_rev = np.zeros(len(examples), dtype=np.int32)
    labels_rev[len(pos):] += 1
    aupr_out = average_precision_score(labels_rev, -examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr_in, aupr_out, fpr


def create_ood_noise(noise_type, ood_num_examples, num_to_avg):
    if noise_type == "Gaussian":
        dummy_targets = torch.ones(ood_num_examples * num_to_avg)
        ood_data = torch.from_numpy(np.float32(np.clip(
            np.random.normal(size=(ood_num_examples * num_to_avg, 3, 32, 32), scale=0.5), -1, 1)))
        ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    elif noise_type == "Rademacher":
        dummy_targets = torch.ones(ood_num_examples * num_to_avg)
        ood_data = torch.from_numpy(np.random.binomial(
            n=1, p=0.5, size=(ood_num_examples * num_to_avg, 3, 32, 32)).astype(np.float32)) * 2 - 1
        ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    elif noise_type == "Blob":
        ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(ood_num_examples * num_to_avg, 32, 32, 3)))
        for i in range(ood_num_examples * num_to_avg):
            ood_data[i] = gblur(ood_data[i], sigma=1.5, multichannel=False)
            ood_data[i][ood_data[i] < 0.75] = 0.0

        dummy_targets = torch.ones(ood_num_examples * num_to_avg)
        ood_data = torch.from_numpy(ood_data.transpose((0, 3, 1, 2))) * 2 - 1
        ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    return ood_data


def val_cifar():
    '''
    Evaluate ID acc and OOD detection on CIFAR10/100
    '''
    model.eval()
    ts = time.time()
    test_acc_meter = AverageMeter()
    labels_list = []
    pred_list = []
    logits_list = []

    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)

        pred = logits.data.max(1)[1]            
        acc = pred.eq(targets.data).float().mean()

        # append loss:
        labels_list.append(targets.detach().cpu().numpy())
        pred_list.append(pred.detach().cpu().numpy())
        logits_list.append(logits.detach().cpu().numpy())
        test_acc_meter.append(acc.item())
    print('clean test time: %.2fs' % (time.time()-ts))
    # test loss and acc of this epoch:
    test_acc = test_acc_meter.avg
    in_labels = np.concatenate(labels_list, axis=0)
    in_preds = np.concatenate(pred_list, axis=0)
    in_logits = np.concatenate(logits_list, axis=0)

    np.save(os.path.join(save_dir, 'in_logits.npy'), in_logits)
    np.save(os.path.join(save_dir, 'in_labels.npy'), in_labels)
    many_acc, median_acc, low_acc, _ = shot_acc(in_preds, in_labels, img_num_per_cls, acc_per_cls=True)

    clean_str = 'ACC: %.4f (%.4f, %.4f, %.4f)' % (test_acc, many_acc, median_acc, low_acc)
    print(clean_str)
    fp.write(clean_str + '\n')
    fp.flush()

    avg_auroc = 0
    avg_aupr_in = 0
    avg_aupr_out = 0
    avg_fpr95 = 0

    if args.noise_type == None:
        OOD_datasets = ['texture', 'svhn', 'cifar', 'tin', 'lsun', 'places365']       
    else:
        OOD_datasets = ['Gaussian', 'Rademacher', 'Blob']

    for douts in OOD_datasets:
        if args.noise_type == None:
            args.dout = douts
            if args.dout == 'cifar':
                if args.dataset == 'cifar10':
                    args.dout = 'cifar100'
                elif args.dataset == 'cifar100':
                    args.dout = 'cifar10'
            ood_set = SCOODDataset(os.path.join(args.data_root_path, 'SCOOD'), id_name=args.dataset, ood_name=args.dout, transform=test_transform)
        else:
            ood_set = create_ood_noise(douts, 10000, 1)       
        ood_loader = DataLoader(ood_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
                                    drop_last=False, pin_memory=True)
        print('Dout is %s with %d images' % (args.dout, len(ood_set)))
        
        # confidence distribution of correct samples:
        sc_labels_list = []
        ood_logit_list = []

        for images, sc_labels in ood_loader:
            images = images.to(device)
            logits = model(images)

            # append loss:
            ood_logit_list.append(logits.detach().cpu().numpy())
            sc_labels_list.append(sc_labels)
        ood_logit = np.concatenate(ood_logit_list, axis=0)
        sc_labels = np.concatenate(sc_labels_list, axis=0)
        np.save(os.path.join(save_dir, douts), ood_logit)

        

        # DODA  
        # The core idea of outlier distribution adaptation is that 
        # using the distribution of test true OOD data to calibrate the test data (both ID and OOD) to further decouple ID and OOD

        # There are two part in DODA, creating true outlier distribution and calibrating output logit based on this outlier distribution
        # In our paper we propose a distribution update manner depending on both training data and accurate predicted OOD data, 
        # which limits the generalization and is sensitive with hyperparameters
        # Therefore, here we provide a simple but effective way, without using training data, without predicted OOD data
        
        # randomize the test data
        all_logits = np.concatenate((in_logits, ood_logit), axis=0)       
        random_num = [i for i in range(all_logits.shape[0])]
        random.shuffle(random_num)
        random_num = np.vstack(random_num)
        random_logits = all_logits[random_num].squeeze()

        all_probs = []
        batch_size = 100 
        test_batch = math.ceil(all_logits.shape[0] / batch_size)
        for i in range(test_batch):
            start = i*batch_size
            end = (i+1)*batch_size
            if i == test_batch - 1:
                end = all_logits.shape[0]
            test_logits = random_logits[start:end]

            # w/ calibration
            # softmax on each class for all batch sampels, not on individual sample 
            # This formalization is mathematically equivalent with our paper Eq.5 
            # We can decouple 'test_logits' into ID part and OOD part (ID test data and OOD test data in this batch), 
                # the ID part similar to 1 in Eq.5, which is a constant
                # the OOD part similar to outlier distribution 'P^out', which is determined by OOD data in this batch
            # We use OOD data in each test batch to calibate the same batch ID data, eliminating the need of a cumulative outlier distribution
            # This way can be applied to other methods based on any training data directly
            test_prob = np.exp(test_logits) / np.sum(np.exp(test_logits), axis=0)   
            test_prob = np.sum(test_prob, axis=1)
            # The limitation is the need for batch inference

            # # w/o calibration
            # test_prob = np.exp(test_logits).sum(axis=1) # Energy for each sample

            # # Oracle upperbound
            # The same calibration way with paper
            # test_prob = np.exp(test_logits) / (1 + np.exp(ood_logit).mean(axis=0))
            # test_prob = np.sum(test_prob, axis=1)


            all_probs.append(test_prob)
        
        all_probs = np.hstack(all_probs)

        random_sort = np.argsort(random_num.squeeze())
        all_probs = all_probs[random_sort].squeeze()


        in_metric = all_probs[:10000]
        # in_metric = in_probs[in_labels<33]
        # in_metric = in_probs[in_labels>66]

        ood_metric = all_probs[10000:]


        fake_ood_scores = ood_metric[sc_labels>=0]
        real_ood_scores = ood_metric[sc_labels<0]
        real_in_scores = np.concatenate([in_metric, fake_ood_scores], axis=0)


        print('fake_ood_scores:', fake_ood_scores.shape)
        print('real_in_scores:', real_in_scores.shape)
        print('real_ood_scores:', real_ood_scores.shape)

        auroc, aupr_in, aupr_out, fpr95 = get_measures(real_in_scores, real_ood_scores)
        avg_auroc += auroc
        avg_aupr_in += aupr_in
        avg_aupr_out += aupr_out
        avg_fpr95 += fpr95

        # print:     
        ood_detectoin_str = 'auroc: %.4f, aupr_in: %.4f, aupr_out: %.4f, fpr95: %.4f' % (auroc, aupr_in, aupr_out, fpr95)
        print(ood_detectoin_str)
        fp.write('\n===%s===\n' % (args.dout))
        fp.write(ood_detectoin_str + '\n')
        fp.flush()


    ood_detectoin_str = 'avg_auroc: %.4f, avg_aupr_in: %.4f, avg_aupr_out: %.4f, avg_fpr95: %.4f' % (avg_auroc/6, avg_aupr_in/6, avg_aupr_out/6, avg_fpr95/6)
    print(ood_detectoin_str)
    fp.write('\n===average===\n')
    fp.write(ood_detectoin_str + '\n')
    fp.write('\n')
    fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a CIFAR Classifier')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_workers', type=int, default=4)
    # dataset:
    parser.add_argument('--dataset', '--ds', default='cifar100', choices=['cifar10', 'cifar100'], help='which dataset to use')
    parser.add_argument('--data_root_path', '--drp', default='./dataset', help='Where you save all your datasets.')
    parser.add_argument('--dout', default='cifar', choices=['svhn', 'places365', 'cifar', 'texture', 'tin', 'lsun'], help='which dout to use')
    parser.add_argument('--model', '--md', default='ResNet18', choices=['ResNet18', 'ResNet34'], help='which model to use')
    parser.add_argument('--noise_type', default=None, choices=['Gaussian', 'Rademacher', 'Blob'], help='data root path')
    # 
    parser.add_argument('--imbalance_ratio', '--rho', default=0.01, type=float)
    parser.add_argument('--test_batch_size', '--tb', type=int, default=1000)
    parser.add_argument('--ckpt_path', default='./result')
    args = parser.parse_args()
    print(args)


    device = 'cuda:'+str(args.gpu)

    save_dir = os.path.join(args.ckpt_path, args.dataset, args.model)
    create_dir(save_dir)
    

    # data:
    train_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #LTR
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #normal
    ])
    if args.dataset == 'cifar10':
        num_classes = 10
        train_set = IMBALANCECIFAR10(train=True, transform=train_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR10(train=False, transform=test_transform, imbalance_ratio=1, root=args.data_root_path)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_set = IMBALANCECIFAR100(train=True, transform=train_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR100(train=False, transform=test_transform, imbalance_ratio=1, root=args.data_root_path)

    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, 
                                drop_last=False, pin_memory=True)

    img_num_per_cls = np.array(train_set.img_num_per_cls)

    # model:
    if args.model == 'ResNet18':
        encoder_num_layers = 18 # network architecture is ResNet34
        isPretrained = False
        model = ResnetEncoder(encoder_num_layers, isPretrained, embDimension=num_classes, poolSize=4).to(device)
    elif args.model == 'ResNet34':
        encoder_num_layers = 34 # network architecture is ResNet34
        isPretrained = False
        model = ResnetEncoder(encoder_num_layers, isPretrained, embDimension=num_classes, poolSize=4).to(device)

    # load model:
    if args.dataset == 'cifar10':
        model.load_state_dict(torch.load('./pretrain/CIFAR10.param'))   
    if args.dataset == 'cifar100':
        model.load_state_dict(torch.load('./pretrain/CIFAR100.param'))  

    # model.load_state_dict(torch.load(os.path.join(save_dir,'CIFAR100.param')))  

    # log file:
    test_result_file_name = 'test_results.txt'
    fp = open(os.path.join(save_dir, test_result_file_name), 'a+')

    val_cifar()
