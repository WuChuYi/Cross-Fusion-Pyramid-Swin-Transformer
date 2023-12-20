from email.policy import default
import os
import time
import random
import argparse
import datetime
import numpy as np


import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
import models_t2t
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from vgg import VGGnet
import torchvision.models.resnet as resnet
from torchvision.models import densenet121,googlenet
from timm.models import create_model
from utilst2t import load_for_transfer_learning
import gvt
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
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU",default=16)
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
                        
    parser.add_argument('--resume', help='resume from checkpoint')
    
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='./output/', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval',action='store_true', help='Perform evaluation only')
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
    # model = googlenet(num_classes=4)
    # model = resnet.resnext50_32x4d(pretrained=False,num_classes=4)
    # model = densenet121(pretrained=False,num_classes=4)
    # model = create_model(
    # 't2t_vit_14', pretrained=False, num_classes=4, drop_rate=0, drop_connect_rate=None, drop_path_rate=0.1, 
    # drop_block_rate=None, global_pool=None, bn_tf=False, bn_momentum=None, bn_eps=None,checkpoint_path='', img_size=224)
    # load_for_transfer_learning(model, '/home/wcy/data/wcy86/harricane/code/1-Swin-Transformer-main/output/outputt2t_pakitai_1114_2/default/ckpt_epoch_0.pth', use_ema=True, strict=False, num_classes=4)
    # model = create_model('pcpvt_base_v0', pretrained=False)
    
    model.cuda()
    logger.info(str(model))
    
    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,find_unused_parameters=True,)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy(weight=torch.Tensor(config.TRAIN.LOSS_WEIGHT).cuda())
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor(config.TRAIN.LOSS_WEIGHT).cuda())
        print(config.TRAIN.LOSS_WEIGHT)
    max_accuracy = 0.0 

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return


    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc2_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets,fn) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        outputs = model(samples)
            

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion(outputs, targets)
            acc1, acc2 = accuracy4train(outputs, targets, topk=(1, 2))
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion(outputs, targets)
            acc1, acc2 = accuracy4train(outputs, targets, topk=(1, 2))
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        acc1_meter.update(reduce_tensor(acc1).item(), targets.size(0))
        acc2_meter.update(reduce_tensor(acc2).item(), targets.size(0))

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr*1e6:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val*1e3:.4f} ({loss_meter.avg*1e3:.4f})\t'
                f'acc1 {acc1_meter.val:.4f} ({acc1_meter.avg:.4f})\t'
                f'acc2 {acc2_meter.val:.4f} ({acc2_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

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
    
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    acc4nodam_meter = AverageMeter()
    acc4minor_meter = AverageMeter()
    acc4major_meter = AverageMeter()
    acc4destr_meter = AverageMeter()

    os.makedirs('./output/'+config.MODEL.NAME+'/default',exist_ok=True)
    f=open('./output/'+config.MODEL.NAME+'/default/test_pred.csv','w')
    f.writelines('fn,target,pred\n')
    end = time.time()
    for idx, (images, target,fn) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # compute output
        output = model(images)
        b,p=output.topk(2, 1, True, True)
        for n in zip(fn,target,p):
            f.writelines(n[0]+','+str(n[1])+','+str(n[2][0])+'\n')

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 2))
        acc4nodam,bs4nodam = accuracy4class(output, target,0)
        acc4minor,bs4minor = accuracy4class(output, target,1)
        acc4major,bs4major = accuracy4class(output, target,2)
        acc4destr,bs4destr = accuracy4class(output, target,3)

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
    f.close()
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
    os.makedirs('./output'+config.MODEL.NAME+'/default',exist_ok=True)
    f=open('./output'+config.MODEL.NAME+'/default/test_pred.csv','w')
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
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
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
