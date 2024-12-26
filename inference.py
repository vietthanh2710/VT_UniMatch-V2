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

from dataset.supervised import SemiDataset
from model.semseg.dpt import DPT
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log, SegmentationMetrics
from util.dist_helper import setup_distributed

import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Fully-Supervised Training in Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--pretrained-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def evaluate(model, loader, mode, cfg, multiplier=None):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    dice_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()

    with torch.no_grad():
        i = 0
        for img, mask, id in loader:
            
            img = img.cuda()
                
            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()
                
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: row + grid, col: col + grid])
                        final[:, :, row: row + grid, col: col + grid] += pred.softmax(dim=1)
                        if col == w - grid:
                            break
                        col = min(col + int(grid * 2 / 3), w - grid)
                    if row == h - grid:
                        break
                    row = min(row + int(grid * 2 / 3), h - grid)
                    
                pred = final
            
            else:
                assert mode == 'original'
                
                if multiplier is not None:
                    ori_h, ori_w = img.shape[-2:]
                    if multiplier == 512:
                        new_h, new_w = 512, 512
                    else:
                        new_h, new_w = int(ori_h / multiplier + 0.5) * multiplier, int(ori_w / multiplier + 0.5) * multiplier
                    # img = F.interpolate(img, (new_h, new_w), mode='bilinear', align_corners=True)
                    img = F.interpolate(img, (266, 266), mode='bilinear', align_corners=True)
                pred = model(img)
            
                if multiplier is not None:
                    pred = F.interpolate(pred, (ori_h, ori_w), mode='bilinear', align_corners=True)

            metric_calculator = SegmentationMetrics(average=True, ignore_background=True, ignore_index = 255,activation='0-1')
            _, dice, precision, recall = metric_calculator(mask, pred.cpu())
            
            pred = pred.argmax(dim=1)
            
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
            
            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()
            
            reduced_dice = torch.from_numpy(np.array([dice])).cuda()
            reduced_precision = torch.from_numpy(np.array([precision])).cuda()
            reduced_recall = torch.from_numpy(np.array([recall])).cuda()
            print(f"{i} -- Dice: {dice}, Pre: {precision}, Recall: {recall}")
            
            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)
            dist.all_reduce(reduced_dice)
            dist.all_reduce(reduced_precision)
            dist.all_reduce(reduced_recall)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
            dice_meter.update(reduced_dice.cpu().numpy())
            precision_meter.update(reduced_precision.cpu().numpy())
            recall_meter.update(reduced_recall.cpu().numpy())

            i += 1

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)
    mdice = dice_meter.avg
    mprecision = precision_meter.avg
    mrecall = recall_meter.avg
    return mIOU, iou_class, mdice, mprecision, mrecall

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

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    state_dict = torch.load('/kaggle/input/doc_code/pytorch/default/1/1_Code/dinov2_vits14_pretrain.pth')
    model.backbone.load_state_dict(state_dict)
    
    if cfg['lock_backbone']:
        model.lock_backbone()
    
    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
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
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    
    valset = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'val'
    )
    
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler
    )
    print(len(valloader))

    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint = torch.load('/kaggle/input/uni2_model/pytorch/default/2/best.pth')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    model.eval()

    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    mIoU, iou_class, mdice, mprecision, mrecall  = evaluate(model, valloader, eval_mode, cfg, multiplier=14)
        
    if rank == 0:
        for (cls_idx, iou) in enumerate(iou_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                        'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
        logger.info('***** Evaluation {} ***** >>>> MeanIoU      : {:.2f}\n'.format(eval_mode, mIoU))
        logger.info('***** Evaluation {} ***** >>>> MeanDice     : {:.2f}\n'.format(eval_mode, mdice))
        logger.info('***** Evaluation {} ***** >>>> MeanPrecision: {:.2f}\n'.format(eval_mode, mprecision))
        logger.info('***** Evaluation {} ***** >>>> MeanRecall   : {:.2f}\n'.format(eval_mode, mrecall))
            
        writer.add_scalar('eval/mIoU', mIoU, epoch)
        writer.add_scalar('eval/mdice', mdice, epoch)
        writer.add_scalar('eval/mprecision', mprecision, epoch)
        writer.add_scalar('eval/mrecall', mrecall, epoch)
        for i, iou in enumerate(iou_class):
            writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

if __name__ == '__main__':
    main()
