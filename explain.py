# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from email.policy import default
import os
import time
import random
import argparse
# import datetimeii
import numpy as np


import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from vgg import VGGnet
import torchvision.models.resnet as resnet
import shap

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str,  metavar="FILE", help='path to config file',
        default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/configs/swin_base_patch4_window7_224_jiangnan.yaml' )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU",default=1)
    parser.add_argument('--data-path', type=str, help='path to dataset',default='/home/wcy/data/wcy86/harricane/data/envf+paki+tai')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight'
                        # )
                        # ,default='/exstorage/wcy/harricane/code/1-Swin-Transformer-main/pretrained/swin_base_patch4_window7_224.pth')
                        #  ,default='/exstorage/wcy/output/swin_base_patch4_window7_224_newset2_6b_1025/default/ckpt_epoch_66.pth')
                        # ,default='/home/wcy/data/wcy86/output/swin_base_patch4_window7_224_newset_6b_1028_0/default/ckpt_epoch_55.pth')
                        # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/swin_base_patch4_window7_224_paki_6b_0301_5/default/ckpt_epoch_50.pth')
                        # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/swin_base_patch4_window7_224_paki_6b_0307_6_1/default/ckpt_epoch_29.pth')
                        # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_paki_6b_0324_1/default/ckpt_epoch_13.pth')
                        # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_jnonly_6b_0613/default/ckpt_epoch_68.pth')
                        # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/pretrained/vgg16-397923af.pth')
                        # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_paki_6b_0328_1/default/ckpt_epoch_77.pth')
                        # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_paki_3b_0331_1/default/ckpt_epoch_22.pth')
                        # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_paki_6b_0408_1_mixonly/default/ckpt_epoch_78.pth')
                        # ,default='/home/wcy/.cache/torch/hub/checkpoints/resnext50_32x4d-7cdf4587.pth')
                        # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_paki_6b_0328_6/default/ckpt_epoch_66.pth')
                        # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_paki_6b_0410_1_mixoonly/default/ckpt_epoch_27.pth')
                        # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_SWINT_CF_PKTAI/default/ckpt_epoch_2.pth')
                        # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_SWINT_CFP_PKTAI_2/default/ckpt_epoch_47.pth')
                        # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_tai_build_0922/default/ckpt_epoch_10.pth')
                        ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_envpakitai_1014/default/ckpt_epoch_8.pth')
    parser.add_argument('--resume', help='resume from checkpoint'
    # )
    #,default='/home/wcy/data/wcy86/output/swin_base_patch4_window7_224_newset_6b_1025/default/ckpt_epoch_26.pth')
    # ,default="/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/swin_base_patch4_window7_224_paki_6b_0307_6_1/default/ckpt_epoch_21.pth")
    # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_paki_6b_0328_6/default/ckpt_epoch_66.pth')
    # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_paki_6b_0408_1/default/ckpt_epoch_2.pth')
    # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_paki_6b_0410_6_pyonly/default/ckpt_epoch_17.pth')
    # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_paki_6b_0410_1_mixonly/default/ckpt_epoch_17.pth')
    # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_SWINT_CF_PKTAI/default/ckpt_epoch_36.pth')
    # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_paki_6b_0328_6/default/ckpt_epoch_66.pth')
    # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_paki_6b_0410_1_mixoonly/default/ckpt_epoch_27.pth')
    # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_SWINT_CFP_PKTAI_3/default/ckpt_epoch_60.pth')
    # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_swint_taionly_0717/default/ckpt_epoch_70.pth')
    # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_SWINT_CFP_PKTAI_3/default/ckpt_epoch_37.pth')
    # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_envpakitai_1014/default/ckpt_epoch_8.pth')
    # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_envpakitai_1018/default/ckpt_epoch_13.pth')
    # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_SWINT_PKTAI/default/ckpt_epoch_28.pth')
    # ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_SWINT_CDA_PKTAI_1/default/ckpt_epoch_12.pth')
    ,default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputswin_base_patch4_window7_224_SWINT_FP_PKTAI_1/default/ckpt_epoch_37.pth')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', default=True,action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    # capsnet 参数
    parser.add_argument('--with_reconstruction', action='store_true', default=False)
    parser.add_argument('--routing_iterations', type=int, default=3)
    

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    # model = VGGnet(in_channels=config.MODEL.SWIN.IN_CHANS, num_classes=config.MODEL.NUM_CLASSES)
    # model = resnet.resnext50_32x4d(pretrained=False,num_classes=4)
    # model.cuda()
    logger.info(str(model))
    
    optimizer = build_optimizer(config, model)
    # if config.AMP_OPT_LEVEL != "O0":
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,find_unused_parameters=True)
    # model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
 
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
        # acc1, acc5, loss = validate(config, data_loader_val, model)
        # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        eval_model(config, model, data_loader_train)
        if config.EVAL_MODE:
            return


def eval_model(config, model, data_loader):
    model.eval()
    data=None
    target=None
    tt=None
    flag1=True
    toE=None
    for idx, (samples, targets,fn) in enumerate(data_loader):
        # samples = samples.cuda(non_blocking=True)
        # targets = targets.cuda(non_blocking=True)
        
        if 'post__2_x4_y26' in fn[0]:
        # if 'post__2_x4_y26' in fn[0]:
            data=torch.cat((data,samples),dim=0)
            target=torch.cat((target,targets),dim=0)
            tt=targets
            toE=samples
            if not flag1:
                break
        elif flag1:
            if data is not None:
                data=torch.cat((data,samples),dim=0)
                target=torch.cat((target,targets),dim=0)
                if data.shape[0]>200:
                    flag1=False
            else:
                data=samples
                target=targets
        else:
            if toE is None:
                continue
            else:
                break
            

    e = shap.GradientExplainer((model, model.layers[2]), data)#10min 
    shap_values, indexes = e.shap_values(
        toE, ranked_outputs=1, nsamples=100
    )

    # get the names for the classes
    index_names = np.array(tt)
    # index_names = np.vectorize(targets)(indexes)

    # plot the explanations
    # shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
    np.save('/home/wcy/data/wcy86/harricane/data/explain/result/fpswint-shapv-layers2-100-x4_y26.npy',np.array(shap_values))
    np.save('/home/wcy/data/wcy86/harricane/data/explain/result/fpswint-sample-layers2-100-x4_y26.npy',np.array(toE))
    np.save('/home/wcy/data/wcy86/harricane/data/explain/result/fpswint-index-layers2-100-x4_y26.npy',np.array(index_names))
    np.save('/home/wcy/data/wcy86/harricane/data/explain/result/fpswint-fn-layers2-100-x4_y26.npy',np.array(fn))

    # shap.image_plot(shap_values, samples, index_names)


@torch.no_grad()
def accuracy4train(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    _, targ = target.topk(1, 1, True, True)
    # targ = target

    pred = pred.t()
    correct = pred.eq(targ.t().expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

@torch.no_grad()
def accuracy4class(output, target, classnum):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    batch_size = target[target==classnum].size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    t=target.reshape(1, -1).expand_as(pred)
    correct = pred[t==classnum].eq(t[t==classnum])
    # print(correct.shape)
    if batch_size==0:
        return torch.Tensor([0]),0
    return (correct.float().sum() * 100. / batch_size),batch_size

@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    
    # img=torch.randn((1,6,224,224))
    # start = time.time()
    # for i in range(500):
    #     model(img)
    # end = time.time()
    # t_all=end-start
    # print(t_all,t_all/500)

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    acc4nodam_meter = AverageMeter()
    acc4minor_meter = AverageMeter()
    acc4major_meter = AverageMeter()
    acc4destr_meter = AverageMeter()
    # bn1=0
    # bn2=0
    # bn3=0
    # bn4=0
    # c1=0
    # c2=0
    # c3=0
    # c4=0
    os.makedirs('/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/'+config.MODEL.NAME+'/default',exist_ok=True)
    f=open('/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/'+config.MODEL.NAME+'/default/test_pred.csv','w')
    f.writelines('fn,target,pred\n')
    end = time.time()
    for idx, (images, target,fn) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # bs,_,_,_=images.shape
        # I=[[int(i.split('.')[0].split('x')[1].split('_y')[0]),int(i.split('.')[0].split('x')[1].split('_y')[1].split('_')[0])] for i in fn]
        # I=np.array(I)
        # B=((I.reshape(bs,-1,1) & (2**np.arange(8))) != 0).astype(int)
        # pos=torch.tensor(B).cuda(non_blocking=True)
        # compute output
        if config.MODEL.TYPE=="capsnet":
            _,output = model(images)
        else:
            output = model(images)
            # output = model(images,pos)
        b,p=output.topk(2, 1, True, True)
        for n in zip(fn,target,p):
            f.writelines(n[0]+','+str(n[1])+','+str(n[2][0])+'\n')

        # if target.sum()>0:
        #     print(111)
        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 2))
        acc4nodam,bs4nodam = accuracy4class(output, target,0)
        acc4minor,bs4minor = accuracy4class(output, target,1)
        acc4major,bs4major = accuracy4class(output, target,2)
        acc4destr,bs4destr = accuracy4class(output, target,3)
        # bn1+=bs4nodam
        # bn2+=bs4minor
        # bn3+=bs4major
        # bn4+=bs4destr
        # c1+=acc4nodam.cpu().numpy()*bs4nodam
        # c2+=acc4minor.cpu().numpy()*bs4nodam
        # c3+=acc4major.cpu().numpy()*bs4nodam
        # c4+=acc4destr.cpu().numpy()*bs4nodam
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)
        if bs4nodam!=0:
            acc4nodam = reduce_tensor(acc4nodam)
            acc4nodam_meter.update(acc4nodam.item(), bs4nodam)
        if bs4minor!=0:
            acc4minor = reduce_tensor(acc4minor)
            acc4minor_meter.update(acc4minor.item(), bs4minor)
        if bs4major!=0:
            acc4major = reduce_tensor(acc4major)
            acc4major_meter.update(acc4major.item(), bs4major)
        if bs4destr!=0:
            acc4destr = reduce_tensor(acc4destr)
            acc4destr_meter.update(acc4destr.item(), bs4destr)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                # f'Acc@nodam {acc4nodam_meter.val:.3f} ({acc4nodam_meter.avg:.3f})\t'
                # f'Acc@minor {acc4minor_meter.val:.3f} ({acc4minor_meter.avg:.3f})\t'
                # f'Acc@major {acc4major_meter.val:.3f} ({acc4major_meter.avg:.3f})\t'
                # f'Acc@destr {acc4destr_meter.val:.3f} ({acc4destr_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}\t'
    f' ({acc4nodam_meter.avg:.3f})\t'
    f' ({acc4minor_meter.avg:.3f})\t'
    f' ({acc4major_meter.avg:.3f})\t'
    f' ({acc4destr_meter.avg:.3f})\t'
    )
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return

def test(config,data_loader,model):
    os.makedirs('/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output'+config.MODEL.NAME+'/default',exist_ok=True)
    f=open('/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output'+config.MODEL.NAME+'/default/test_pred.csv','w')
    f.writelines('fn,target,pred\n')
    end = time.time()
    for idx, (images, target,fn) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        if config.MODEL.TYPE=="capsnet":
            _,output = model(images)
        else:
            output = model(images)
        _,p=output.topk(1, 1, True, True)
        for n in zip(fn,target,p):
            f.writelines(n[0]+','+str(n[1])+','+str(n[2])+'\n')

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,0" 
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        # rank = -1
        # world_size = -1
        rank = 0
        world_size = 1
    torch.cuda.set_device(config.LOCAL_RANK)
    # torch.cuda.set_device(0)

    # print(os.environ['MASTER_PORT'])
    os.environ['MASTER_ADDR']="127.0.0.1"
    os.environ['MASTER_PORT']="12345"

    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
