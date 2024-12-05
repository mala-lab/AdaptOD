'''
Finetune auxiliary classification head.
'''
import argparse, os, warnings, datetime
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
import torchvision.transforms as transforms

from dataset.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100
from dataset.ImbalanceImageNet import LT_Dataset
from dataset.tinyimages_300k import TinyImages
from dataset.imagenet_ood import ImageNet_ood
from models.network_arch_resnet import ResnetEncoder
from skimage.filters import gaussian as gblur

from utils.utils import *
from utils.ltr_metrics import *

def get_args_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='Auxiliar branch finetuning.')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_workers', '--cpus', default=8, help='number of threads for data loader')
    parser.add_argument('--data_root_path', '--drp', default='./dataset', help='data root path')
    parser.add_argument('--dataset', '--ds', default='cifar100', choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--id_class_number', type=int, default=1000, help='for ImageNet subset')
    parser.add_argument('--model', '--md', default='ResNet18', choices=['ResNet18', 'ResNet34', 'ResNet50'], help='which model to use')
    parser.add_argument('--imbalance_ratio', '--rho', default=0.01, type=float)
    # training params:
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='input batch size for training')
    parser.add_argument('--test_batch_size', '--tb', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('--epochs', '-e', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='which lr decay method to use')
    parser.add_argument('--decay_epochs', '--de', default=[1,2], nargs='+', type=int, help='milestones for multisteps lr decay')
    parser.add_argument('--num_ood_samples', default=300000, type=float, help='Number of OOD samples to use.')
    # 
    parser.add_argument('--layer', default='fc', choices=['both', 'fc', 'bn'], 
        help='Which layers to finetune. both: both BN and the last fc layers | bn: all bn layers excluding fc layer | fc: the final fc layer excluding all bn layers')
    parser.add_argument('--pretrained_str', default='./pretrain', help='Starting point for finetuning')
    parser.add_argument('--save_root_path', '--srp', default='./result', help='data root path')
    parser.add_argument('--LA', default=False)
    args = parser.parse_args()

    return args

def set_mode_to_train(model, args, fp=None):
    if fp is not None:
        fp.write('All modules in training mode:\n')
    for name, m in model.named_modules():
        if len(list(m.children())) > 0: # not a leaf node
            continue
        if args.layer == 'all':
            m.train()
        else:
            m.eval()
        if args.layer == 'both':
            if any([_name in name for _name in ['bn', 'shortcut.1', 'downsample.1']]):
                m.train()
        elif args.layer == 'bn':
            if any([_name in name for _name in ['bn', 'shortcut.1', 'downsample.1']]):
                m.train()
        # print(name, m.training)
        if m.training and fp is not None:
            fp.write(name+'\n')

if __name__ == '__main__':
    # get args:
    args = get_args_parser()

    # fix random seed
    np.random.seed(22)
    torch.manual_seed(22)
    torch.cuda.manual_seed(22)
    torch.cuda.manual_seed_all(22)

    # mkdirs:
    save_dir = os.path.join(args.save_root_path, args.dataset, args.model)
    print('Saving to %s' % save_dir)
    create_dir(save_dir)

    fp = open(os.path.join(save_dir, 'train_log.txt'), 'a+')
    fp_val = open(os.path.join(save_dir, 'val_log.txt'), 'a+')

    # intialize device:
    device = 'cuda:'+str(args.gpu)
    torch.backends.cudnn.benchmark = True

    # get batch size:
    train_batch_size = args.batch_size 
    num_workers = args.num_workers 

    # data:
    if args.dataset in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if args.dataset == 'cifar10':
        num_classes = 10
        train_set = IMBALANCECIFAR10(train=True, transform=train_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR10(train=False, transform=test_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_set = IMBALANCECIFAR100(train=True, transform=train_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR100(train=False, transform=test_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
    elif args.dataset == 'imagenet':
        num_classes = args.id_class_number
        train_set = LT_Dataset(
            os.path.join(args.data_root_path, 'imagenet'), './datasets/ImageNet_LT_train.txt', transform=train_transform, 
            subset_class_idx=np.arange(0,args.id_class_number))
        test_set = LT_Dataset(
            os.path.join(args.data_root_path, 'imagenet'), './datasets/ImageNet_LT_val.txt', transform=test_transform,
            subset_class_idx=np.arange(0,args.id_class_number))
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers,
                                drop_last=True, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers, 
                                drop_last=False, pin_memory=False)
    print('Training on %s with %d images and %d validation images.' % (args.dataset, len(train_set), len(test_set)))

    if args.dataset in ['cifar10', 'cifar100']:
        ood_set = Subset(TinyImages(args.data_root_path, transform=train_transform), list(range(args.num_ood_samples)))
    elif args.dataset == 'imagenet':
        # use 1k classes (non-overlap with 1k classes in imagenet1k) from imagenet10k as OE training images
        ood_set = ImageNet_ood(os.path.join(args.data_root_path, 'ImageNet10k_eccv2010/imagenet10k'), transform=train_transform, txt="./datasets/imagenet_extra_1k_wnid_list_picture.txt")
    ood_loader = DataLoader(ood_set, batch_size=train_batch_size*2, shuffle=True, num_workers=num_workers,
                                drop_last=True, pin_memory=False)
    print('Training on %s with %d images and %d validation images | %d OOD training images.' % (args.dataset, len(train_set), len(test_set), len(ood_set)))


    # model:
    if args.model == 'ResNet18':
        encoder_num_layers = 18 # network architecture is ResNet34
        isPretrained = False
        model = ResnetEncoder(encoder_num_layers, isPretrained, embDimension=num_classes, poolSize=4).to(device)
    elif args.model == 'ResNet34':
        encoder_num_layers = 34 # network architecture is ResNet34
        isPretrained = False
        model = ResnetEncoder(encoder_num_layers, isPretrained, embDimension=num_classes, poolSize=4).to(device)
    elif args.model == 'ResNet50':
        encoder_num_layers = 50 # network architecture is ResNet34
        isPretrained = False
        model = ResnetEncoder(encoder_num_layers, isPretrained, embDimension=num_classes, poolSize=4).to(device)

        # model = torch.nn.DataParallel(model)

    # load model:
    if args.dataset == 'cifar10':
        model.load_state_dict(torch.load('./pretrain/CIFAR10.param'))   
    if args.dataset == 'cifar100':
        model.load_state_dict(torch.load('./pretrain/CIFAR100.param'))   
    
  
    # only update bn and fc layers:
    fp.write('ALl parameters requiring grad:'+'\n')    
    for name, p in model.named_parameters(): 
        if args.layer == 'all':
            p.requires_grad = True 
        else:
            p.requires_grad = False 
        if args.layer == 'both':
            if any([_name in name for _name in ['bn', 'shortcut.1', 'downsample.1']]): 
                p.requires_grad = True
            if 'fc' in name or 'linear' in name:
                p.requires_grad = True 
        elif args.layer == 'bn':
            if any([_name in name for _name in ['bn', 'shortcut.1', 'downsample.1']]):
                p.requires_grad = True
        elif args.layer == 'fc':
            if 'fc' in name or 'linear' in name:
                p.requires_grad = True 
        if p.requires_grad:
            print(name)
            fp.write(name+'\n')
    set_mode_to_train(model, args, fp)


    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=True)

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.lr))


    # train:
    training_losses, training_accs, test_clean_losses = [], [], []
    f1s, overall_accs, many_accs, median_accs, low_accs = [], [], [], [], []
    best_overall_acc = 0
    start_epoch = 0

    img_num_per_cls = np.array(train_set.img_num_per_cls)
    img_num_per_cls = torch.from_numpy(img_num_per_cls).to(device)
    prior = img_num_per_cls / torch.sum(img_num_per_cls)

    for epoch in range(start_epoch, args.epochs):

        set_mode_to_train(model, args)
        # model.train()
        train_acc_meter, training_loss_meter = AverageMeter(), AverageMeter()
        for batch_idx, ((in_data, label), (ood_data, ood_labels)) in enumerate(zip(train_loader, ood_loader)):
            in_data, label = in_data.to(device), label.to(device)
            ood_data, ood_labels = ood_data.to(device), ood_labels.to(device)
                
            # forward:
            all_data = torch.cat([in_data, ood_data], dim=0)
            all_logits = model(all_data)


            # ID loss
            in_logits = all_logits[:in_data.shape[0]]

            # adjust logits:
            if args.LA:
                adjusted_in_logits = in_logits + args.tau * prior.log()[None,:]
                in_loss = F.cross_entropy(adjusted_in_logits, label)
            else:
                in_loss = F.cross_entropy(in_logits, label)

            

            # OOD loss            
            all_probs = F.softmax(all_logits, dim=0)
            in_probs = all_probs[:in_data.shape[0]]
            ood_probs = all_probs[in_data.shape[0]:]

            # class-level
            m_in = 1 
            m_out = 0 
            energy_in = torch.sum(in_probs, dim=0)
            energy_out = torch.sum(ood_probs, dim=0)
            ood_loss1 = torch.pow(F.relu(m_in - energy_in), 2).mean() + torch.pow(F.relu(energy_out - m_out), 2).mean()

            # sample-level   
            m_in =  num_classes / train_batch_size
            m_out = 0   
            energy_in = torch.sum(in_probs, dim=1)
            energy_out = torch.sum(ood_probs, dim=1)
            ood_loss2 = torch.pow(F.relu(m_in - energy_in), 2).mean() + torch.pow(F.relu(energy_out - m_out), 2).mean()
                
            ood_loss = ood_loss1 + ood_loss2
            loss = ood_loss + in_loss

            # backward:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # append:
            training_loss_meter.append(loss.item())
            acc = (in_logits.argmax(1) == label).float().mean()
            train_acc_meter.append(acc.item())

            if batch_idx % 100 == 0:
                train_str = 'epoch %d batch %d (train): loss %.4f (in_loss %.4f, ood_loss %.4f (ood_loss1 %.4f, ood_loss2 %.4f))' % (epoch, batch_idx, loss.item(), in_loss.item(), ood_loss.item(), ood_loss1.item(), ood_loss2.item())
                print(train_str)
                fp.write(train_str + '\n')
                fp.flush()

        # lr update:
        scheduler.step()

        # eval on clean set:
        model.eval()
        test_acc_meter, test_loss_meter = AverageMeter(), AverageMeter()
        preds_list, labels_list = [], []
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                logits = model(data)
                pred = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                loss = F.cross_entropy(logits, labels)
                test_acc_meter.append((logits.argmax(1) == labels).float().mean().item())
                test_loss_meter.append(loss.item())
                preds_list.append(pred)
                labels_list.append(labels)

        preds = torch.cat(preds_list, dim=0).detach().cpu().numpy().squeeze()
        labels = torch.cat(labels_list, dim=0).detach().cpu().numpy()

        overall_acc= (preds == labels).sum().item() / len(labels)
        f1 = f1_score(labels, preds, average='macro')

        many_acc, median_acc, low_acc, _ = shot_acc(preds, labels, img_num_per_cls, acc_per_cls=True)

        test_clean_losses.append(test_loss_meter.avg)
        f1s.append(f1)
        overall_accs.append(overall_acc)
        many_accs.append(many_acc)
        median_accs.append(median_acc)
        low_accs.append(low_acc)

        training_acc = train_acc_meter.avg
        val_str = 'epoch %d (test): ACC %.4f (%.4f, %.4f, %.4f) | F1 %.4f | (train) ACC %.4f' % (epoch, overall_acc, many_acc, median_acc, low_acc, f1, training_acc) 
        print(val_str)
        fp_val.write(val_str + '\n')
        fp_val.flush()

        # save pth:
        torch.save(model.state_dict(), os.path.join(save_dir, 'CIFAR100.param'))