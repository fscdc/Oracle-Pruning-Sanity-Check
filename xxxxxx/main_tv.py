from option import args
import torch
import os
from loguru import logger
import sys
from data import Data
from data import num_classes_dict, input_size_dict
from utils import *
from model import model_dict
import torch.nn as nn
from torch.optim import AdamW, SGD
from transformers import get_scheduler
from tqdm.auto import tqdm
from accelerate import Accelerator
from ptflops import get_model_complexity_info

import warnings
warnings.filterwarnings("ignore")

# logger examples
# logger.debug("这是一个调试消息")
# logger.info("这是一个信息消息")
# logger.warning("这是一个警告消息")
# logger.error("这是一个错误消息")
# logger.critical("这是一个严重错误的消息")


# set up logger
logger_txt_path = os.path.join("./record/", f"{args.experiment_name}/{args.experiment_name}.txt")
logger.add(logger_txt_path, rotation="20MB", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")

logger.level("TEST-RESULT", no=52, color="<green>", icon="🐍")
logger.level("TRAIN", no=51, color="<blue>", icon="🐍")


# set up accelerator
accelerator = Accelerator()
if accelerator.is_local_main_process:
    logger.info(f"Set up accelerator!")
    logger.info(f"All params: {args}")

# set up device and criterion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)


# set up dataset
if accelerator.is_local_main_process:
    logger.info(f"Set up dataset {args.dataset}! ----------------- Start!")

if args.dataset == "imdb":
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
elif args.dataset == "imagenet":
    pass
else:
    raise ValueError("Unknown dataset")

loader = Data(args)

num_classes = num_classes_dict[args.dataset]
*_, num_channels, input_height, input_width = input_size_dict[args.dataset]

train_set, test_set, train_subset = loader.train_set, loader.test_set, loader.train_subset
train_loader, test_loader, train_subset_loader = loader.train_loader, loader.test_loader, loader.train_subset_loader
# use for pruned train loss
train_loader_nocollate = loader.train_loader_nocollate


if accelerator.is_local_main_process:
    logger.info(f"Set up dataset {args.dataset}! ----------------- Finish!")

# set up model
if args.arch.startswith("vit") and args.dataset == "imagenet":
    model = model_dict[args.arch]()
elif args.arch == "distilbert":
    model = model_dict[args.arch](
        num_classes=num_classes,
        id2label = id2label,
        label2id = label2id,
    )

model.to(device)

if accelerator.is_local_main_process:
    logger.debug(f"Pretrained Model: {model}")
    logger.info(f"Create model {args.arch}!")
    logger.info(f"Total parameters for pretrained model: {sum(p.numel() for p in model.parameters())}")
    model.eval()
    with torch.no_grad():
        macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=False)
        logger.info(f"Pretrained Model: Params: {params}, MACs: {macs}")
    top1, top5, loss = validate_tv(test_loader, model, criterion, device)
    logger.log("TEST-RESULT", f"Pretrained Model: Top1: {top1}, Top5: {top5}, Loss: {loss}")

# set up optimizer
wd = args.weight_decay

if args.optimizer == "adam":
    optimizer = AdamW(model.parameters(), lr=args.retrain_lr, weight_decay=wd, betas=(0.9, 0.999))
elif args.optimizer == "sgd":
    optimizer = SGD(model.parameters(), lr=args.retrain_lr, weight_decay=wd, momentum=0.9)


# prune model (refer to: https://github.com/VainF/Torch-Pruning/tree/master/examples/torchvision_models)
if accelerator.is_local_main_process:
    logger.info(f"Prune model {args.arch}! ----------------- Start!")

import torch_pruning as tp
import random

def prune_tv(model, example_inputs, importance, model_name):
    
    from torchvision.models.vision_transformer import VisionTransformer

    ori_size = tp.utils.count_params(model)
    model.eval()
    ignored_layers = []
    for p in model.parameters():
        p.requires_grad_(True)

    # Ignore unprunable modules
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m)

    # For ViT: Rounding the number of channels to the nearest multiple of num_heads
    round_to = None
    channel_groups = {}
    if isinstance( model, VisionTransformer): 
        for m in model.modules():
            if isinstance(m, nn.MultiheadAttention):
                channel_groups[m] = m.num_heads

    # (Optional) Register unwrapped nn.Parameters 
    # TP will automatically detect unwrapped parameters and prune the last dim for you by default.
    # If you want to prune other dims, you can register them here.
    unwrapped_parameters = None

    # Build network pruners
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=1,
        pruning_ratio=args.pruning_ratio,
        global_pruning=False,
        round_to=round_to,
        unwrapped_parameters=unwrapped_parameters,
        ignored_layers=ignored_layers,
        channel_groups=channel_groups,
    )

    # Pruning 
    layer_channel_cfg = {}
    for module in model.modules():
        if module not in pruner.ignored_layers:
            if isinstance(module, nn.Conv2d):
                layer_channel_cfg[module] = module.out_channels
            elif isinstance(module, nn.Linear):
                layer_channel_cfg[module] = module.out_features

    for g in pruner.step(interactive=True):
        g.prune()

    if isinstance(model, VisionTransformer):  # Torchvision relies on the hidden_dim variable for forwarding, so we have to modify this varaible after pruning
        model.hidden_dim = model.conv_proj.out_channels

# Reimplement the importance function for oracle pruning (@fengsicheng)
class OracleImportance(tp.importance.Importance):
    def __init__(self, oracle_seed, accelerator=None, logger=None):
        self.oracle_seed = oracle_seed
        self.accelerator = accelerator
        self.logger = logger

    @torch.no_grad()
    def __call__(self, group: tp.Group) -> torch.Tensor:
        _, idxs = group[0]  

        torch.manual_seed(self.oracle_seed)
        tensor = torch.rand(len(idxs))

        if self.accelerator.is_local_main_process:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            np.save(f"./record/{args.experiment_name}/oracle_importance_{timestamp}.npy", tensor.cpu().numpy())
            self.logger.info(f"Save oracle importance as ./record/{args.experiment_name}/oracle_importance_{timestamp}.npy")
        return tensor

if args.pruning_type == 'random':
    imp = tp.importance.RandomImportance()
elif args.pruning_type == 'l1':
    imp = tp.importance.MagnitudeImportance(p=1)
elif args.pruning_type == 'oracle':
    imp = OracleImportance(oracle_seed = args.oracle_seed, accelerator=accelerator, logger=logger)
else: raise NotImplementedError

example_inputs = torch.randn(1,3,224,224).to(device)

prune_tv(model, example_inputs=example_inputs, importance=imp, model_name='vit')

if accelerator.is_local_main_process:
    logger.debug(f"Pruned Model: {model}")
    logger.info(f"Prune model {args.arch}! ----------------- Finish!")
    torch.save(model.state_dict(), f"./record/{args.experiment_name}/{args.experiment_name}_pruned.pth")
    model.eval()
    with torch.no_grad():
        macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=False)
        logger.info(f"Pruned Model: Params: {params}, MACs: {macs}")


# prepare items
train_loader, test_loader, train_subset_loader, train_loader_nocollate, model, optimizer = accelerator.prepare(
    train_loader, test_loader, train_subset_loader, train_loader_nocollate, model, optimizer
)

top1, top5, test_loss = validate_tv_accelerator(test_loader, model, criterion, device, accelerator)
train_top1, train_top5, train_loss = validate_tv_accelerator(train_loader_nocollate, model, criterion, device, accelerator)

# test model
if accelerator.is_local_main_process:
    logger.log("TEST-RESULT", f"Just Pruned Model (Testset): Top1: {top1}, Top5: {top5}, Loss: {test_loss}" )
    logger.log("TEST-RESULT", f"Pruned Train Loss: {train_loss}")
    logger.info(f"Total parameters for pruned model: {sum(p.numel() for p in model.parameters())}")




##########################################################################################################################
# retrain model

if accelerator.is_local_main_process:
    logger.info(f"Retrain model {args.arch}! ----------------- Start!")

if args.optimizer == "adam":
    optimizer = AdamW(model.parameters(), lr=args.retrain_lr, weight_decay=wd, betas=(0.9, 0.999))
elif args.optimizer == "sgd":
    optimizer = SGD(model.parameters(), lr=args.retrain_lr, weight_decay=wd, momentum=0.9)

num_epochs = args.retrain_epoch
num_training_steps = num_epochs * len(train_loader)
num_warmpup_steps = 30 * len(train_loader) if args.lr_decay == "linear" else 0
lr_scheduler = get_scheduler(
    name=args.lr_decay, optimizer=optimizer, num_warmup_steps=num_warmpup_steps, num_training_steps=num_training_steps
)

if accelerator.is_local_main_process:
    logger.info(f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    for i, (images, target) in enumerate(train_loader):
        images = images.to(device=device)
        target = target.to(device=device).float()

        output = model(images)

        loss = criterion(output, target)
        accelerator.backward(loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    if epoch % 1 == 0:
        top1, top5, test_loss = validate_tv_accelerator(test_loader, model, criterion, device, accelerator)
        if accelerator.is_local_main_process:
            logger.log("TRAIN", f"Epoch: {epoch}, Testset_Top1: {top1}, Testset_Top5: {top5}, Testset_Loss: {test_loss}")


if accelerator.is_local_main_process:
    logger.info(f"Retrain model {args.arch}! ----------------- Finish!")
    torch.save(model.state_dict(), f"./record/{args.experiment_name}/{args.experiment_name}_retrained.pth")

##########################################################################################################################



##########################################################################################################################
# test retrained model

top1, top5, test_loss = validate_tv_accelerator(test_loader, model, criterion, device, accelerator)

if accelerator.is_local_main_process:
    logger.log("TEST-RESULT", f"Retrained Pruned Model: Top1: {top1}, Top5: {top5}, Loss: {test_loss}" )

##########################################################################################################################