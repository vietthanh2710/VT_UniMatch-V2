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


def inference(model, loader, mode, cfg, multiplier=None):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    pred_batch = []
    with torch.no_grad():
        i = 0
        for img, _, id in loader:
            
            img = img.cuda()
                
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

            pred = pred.argmax(dim=1)
            pred_batch.append(pred.cpu().numpy())
            i += 1

    return pred_batch

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
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )

    valset = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'infer', id_path = '/kaggle/working/VT_UniMatch-V2/infer.txt'
    )

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler
    )

    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint = torch.load('/kaggle/input/uni2_model/pytorch/default/2/best.pth')
    model.load_state_dict(checkpoint['model'])
    
    model.eval()

    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    pred_batch  = inference(model, valloader, eval_mode, cfg, multiplier=14)
    for idx, pred in enumerate(pred_batch):
        # Convert prediction to a PIL image
        pred_image = Image.fromarray((pred * 255).astype(np.uint8))
        
        # Save the image with the index as the file name
        pred_image.save(os.path.join(args.save_path, f"mask_{idx}.png"))
        

if __name__ == '__main__':
    main()
