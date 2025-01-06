import argparse
import logging
import glob
import os
import pprint
from torchvision.utils import save_image
import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.supervised import SemiDataset, SemiDataset_Weight
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import *
from util.dist_helper import setup_distributed

import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Fully-Supervised Training in Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--pretrained-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

'''
bic_pru
card **
diw 
invoice **
passport
tam_tru **
'''
Category = ['bic_pru','card','diw','invoice','passport','tam_tru']

def weight_computation(sub_miou, local_rank):
    n_samples = np.array([935,11894,5100,1738,3982,27])
    inv_miou = 100 / sub_miou
    weight = inv_miou + 5/n_samples
    weight = torch.tensor(weight, dtype=torch.float32, device = local_rank)
    return weight

def evaluate(model, loader, cfg):
    model.eval()
    
    intersection_meter = MultiAverageMeter()
    union_meter = MultiAverageMeter()

    with torch.no_grad():
        for img, mask, id in loader:

            img = img.cuda()
                
            ori_h, ori_w = img.shape[-2:]
            img = F.interpolate(img, (288, 288), mode='bilinear', align_corners=True)
            pred = model(img)
            
            pred = F.interpolate(pred, (ori_h, ori_w), mode='bilinear', align_corners=True)
            pred = pred.argmax(dim=1)
            
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
            
            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy(), id_path = ''.join(id))
            union_meter.update(reduced_union.cpu().numpy(), id_path = ''.join(id))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    n = np.array([104,1322,567,193,442,3])
    mIOU_folder = np.mean(iou_class, axis = 1)
    mIoU = np.sum(n*mIOU_folder) / 2631.

    return mIoU, iou_class


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    cfg['batch_size'] *= 2
    
    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,                      # model output channels (number of classes in your dataset)
    )
    
    optimizer = AdamW(
        [
            {'params': [p for p in model.parameters()]},
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'Dice':
        criterion = DiceLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'DiceCE':
        criterion1 = DiceLoss(mode = 'multiclass', **cfg['criterion']['kwargs']).cuda(local_rank)
        criterion2 = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    
    n_upsampled = {
        'pascal': 3000, 
        'cityscapes': 3000, 
        'ade20k': 6000, 
        'coco': 30000,
        'doc': 3000,
    }
    trainset = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=n_upsampled[cfg['dataset']]
    )
    valset = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'val'
    )
    
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(
        trainset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler
    )
    
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler
    )
    print(len(trainsampler),len(valsampler))
    print(len(trainloader),len(valloader))
    iters = 0
    total_iters = len(trainloader) * (cfg['epochs1'] + cfg['epochs2'])

    previous_best = 0.0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    # checkpoint = torch.load('/kaggle/input/segformer_mitb2/pytorch/default/1/latest.pth', map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # epoch = checkpoint['epoch']
    # previous_best = checkpoint['previous_best']
    
    """
    Train with equal distribution
    """
    for epoch in range(epoch + 1, cfg['epochs1']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.7f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        total_loss = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img, mask,_) in enumerate(trainloader):

            img, mask = img.cuda(), mask.cuda()

            pred = model(img)

            loss = criterion1(pred, mask) + criterion2(pred, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.95
            optimizer.param_groups[0]["lr"] = lr
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss.item(), iters)
            
            if (i % (len(trainloader) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))
        
        mIoU, iou_class = evaluate(model, valloader, cfg)
        
        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info(f'{Category[cls_idx]:<8} IOU: {iou[0]:.2f} | {iou[1]:.2f} --> Mean: {np.mean(iou):.2f}%')
            logger.info('***** Evaluation ***** >>>> MeanIoU      : {:.2f}\n'.format(mIoU))         
            writer.add_scalar('eval/mIoU', mIoU, epoch)

            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (Category[i]), np.mean(iou), epoch)
        
        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best1.pth'))

    '''
    Reweight via iou score and n_sample
    '''

    sub_miou = np.mean(iou_class, axis = 1)
    folder_weight = weight_computation(sub_miou, local_rank)
    print(folder_weight)
    logger.info(folder_weight)
    trainset_new = SemiDataset_Weight(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=n_upsampled[cfg['dataset']], weight = folder_weight
    )
    
    trainsampler_new = torch.utils.data.WeightedRandomSampler(weights=trainset_new.sample_weight, num_samples=len(trainset_new), replacement=True)
    trainloader_new = DataLoader(
        trainset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_new
    )

    for epoch in range(epoch + 1, cfg['epochs1']+ cfg['epochs2']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.7f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))
            
        model.train()
        total_loss = AverageMeter()

        for i, (img, mask, _) in enumerate(trainloader_new):
            img, mask = img.cuda(), mask.cuda()

            pred = model(img)
            
            loss = criterion1(pred, mask) + criterion2(pred, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.95
            optimizer.param_groups[0]["lr"] = lr
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss.item(), iters)
            
            if (i % (len(trainloader) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))
        
        mIoU, iou_class = evaluate(model, valloader, cfg)
        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info(f'{Category[cls_idx]:<8} IOU: {iou[0]:.2f} | {iou[1]:.2f} --> Mean: {np.mean(iou):.2f}%')
            logger.info('***** Evaluation ***** >>>> MeanIoU      : {:.2f}\n'.format(mIoU))         
            writer.add_scalar('eval/mIoU', mIoU, epoch)

            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (Category[i]), np.mean(iou), epoch)
        
        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best2.pth'))

if __name__ == '__main__':
    main()
