import os
import time
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from accelerate import Accelerator

# Our packages
from smilelogging import Logger
from smilelogging.utils import Timer, get_lr, get_arg
from smilelogging.utils import (
    get_jacobian_singular_values,
    AverageMeter,
    ProgressMeter,
    accuracy,
)
from model import model_dict
from data import Data
from data import num_classes_dict, input_size_dict
from pruner.reinit_model import reinit_model, rescale_model
from option import args


def validate(test_loader, model, criterion):
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    train_state = model.training
    with torch.no_grad():
        model.eval()
        for i, (images, target) in enumerate(test_loader):
            images = images.to(dtype=accelerator.dtype,
                               device=accelerator.device)
            target = target.to(
                device=accelerator.device
            ).long()  # label should be long int
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    if train_state:
        model.train()
    return top1.avg, top5.avg, losses.avg


def set_up_optimizer(model):
    lr = args.lr_ft if hasattr(args, "lr_ft") else args.lr
    init_lr = list(lr.values())[0]
    if args.solver == "Adam":
        optim = torch.optim.Adam(model.parameters(), init_lr)
    else:
        optim = torch.optim.SGD(
            model.parameters(),
            init_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    return optim


def set_up_lr_scheduler(optimizer, start_epoch):
    if hasattr(args, "advanced_lr") and args.advanced_lr.lr_decay == "cos_v2":
        from torch.optim.lr_scheduler import CosineAnnealingLR

        min_lr = args.advanced_lr.min_lr
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=min_lr, last_epoch=start_epoch - 1
        )
        logger.info(
            f"Create lr scheduler: CosineAnnealingLR, eta_min={min_lr}, T_max: {args.epochs}, last_epoch: {start_epoch - 1}"
        )
    else:
        from smilelogging.utils import PresetLRScheduler

        lr = args.lr_ft if get_arg(args, "lr_ft") else args.lr
        lr_scheduler = PresetLRScheduler(lr)
        logger.info(f"Create lr scheduler: Step LR, {lr}")
        # TODO-@mst: Mimic pytorch lr scheduler, implement a new one; use --lr_schedule
    return lr_scheduler


# Set up DDP
accelerator = Accelerator(
    mixed_precision=args.mixed_precision,
    log_with="wandb",
)
if args.ddp:  # these vars are used ONLY in DDP
    args.world_size = accelerator.num_processes
    args.local_rank = accelerator.local_process_index
    args.global_rank = accelerator.process_index

# Set up logger
logger = Logger(args)
accelerator.init_trackers(args.experiment_name, config=vars(args))


def main():
    # Init
    pruner = None
    if get_arg(args, "prune_method"):
        prune_state = "prune"

    # Set up data
    args.distributed = args.ddp
    loader = Data(args)
    train_sampler = loader.train_sampler
    train_loader, test_loader = loader.train_loader, loader.test_loader
    train_loader_raw = loader.train_loader_raw
    num_classes = num_classes_dict[args.dataset]
    *_, num_channels, input_height, input_width = input_size_dict[args.dataset]

    # Get a neat test fn
    def validate_testset(net): return validate(test_loader, net, criterion)
    def validate_trainset(net): return validate(train_loader, net, criterion)
    def validate_trainset_raw(net): return validate(train_loader_raw, net, criterion)
    def retrain(model, train_loader, train_sampler, criterion, args):
        model = copy.deepcopy(model)
        start_epoch=0
        optimizer = set_up_optimizer(model)

        timer = Timer(args.epochs - start_epoch)
        best_acc1, best_epoch = 0, 0
        
        acc1_list, loss_train_list, loss_test_list = [], [], []
        # start retraining
        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            lr_scheduler(optimizer, epoch)

            lr = get_lr(optimizer)

            logger.info("Set lr = %s @ Epoch %d (Start)" % (lr, epoch + 1))

            one_epoch_train(train_loader, model, criterion, optimizer, epoch)

            acc1, acc5, loss_test = validate_testset(model)
            acc1_train, acc5_train, loss_train = -1, -1, -1

            if args.dataset not in ["imagenet"]:
                acc1_train, acc5_train, loss_train = validate_trainset(model)
            elif args.dataset == "imagenet":
                acc1_train, acc5_train, loss_train = validate_trainset_raw(model)
            acc1_list.append(acc1)
            loss_train_list.append(loss_train)
            loss_test_list.append(loss_test)
            
            # log
            num_steps_per_epoch = len(train_loader)
            step = (epoch + 1) * num_steps_per_epoch
            accelerator.log({"acc1": acc1}, step=step)
            accelerator.log({"acc5": acc5}, step=step)
            accelerator.log({"loss_test": loss_test}, step=step)

            # print metrics
            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                best_epoch = epoch
                best_loss_train = loss_train
                best_loss_test = loss_test            
            acclog = (
                f"Epoch {epoch} Acc1 {acc1:.4f} Acc5 {acc5:.4f} TestLoss {loss_test:.4f} "
                f"BestAcc1 {best_acc1:.4f} BestAcc1Epoch {best_epoch} LR {lr}"
            )
            if acc1_train != -1:
                acclog = acclog.replace(
                    "BestAcc1",
                    f"TrainAcc1 {acc1_train:.4f} TrainAcc5 {acc5_train:.4f} "
                    f"TrainLoss {loss_train:.4f} BestAcc1",
                )
            logger.info(acclog, color="green")
            logger.info(f"Predicted finish time: {timer()}")

        last5_acc_mean = np.mean([acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in acc1_list[-5:]])
        last5_acc_std = np.std([acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in acc1_list[-5:]])

        last5_loss_train_mean = np.mean([loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in loss_train_list[-5:]])
        last5_loss_train_std = np.std([loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in loss_train_list[-5:]])

        last5_loss_test_mean = np.mean([loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in loss_test_list[-5:]])
        last5_loss_test_std = np.std([loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in loss_test_list[-5:]])

        best = [best_acc1, best_loss_train, best_loss_test]
        last5 = [last5_acc_mean, last5_acc_std, last5_loss_train_mean, last5_loss_train_std, last5_loss_test_mean, last5_loss_test_std]    

        return best, last5

    # Set up model architecture
    if args.pretrained_ckpt is not None:
        ckpt = torch.load(args.pretrained_ckpt,
                          map_location=torch.device("cpu"))
        logger.info(f'Load pretrained_ckpt: "{args.pretrained_ckpt}"')
        if "model" in ckpt:
            model = ckpt["model"]
            dplog = ""
            if hasattr(model, "module"):
                model = model.module
                dplog = ". Found DataParallel in the model, removed"
            if hasattr(model, "features") and hasattr(
                model.features, "module"
            ):  # For back-compatibility with some old alexnet/vgg models
                model.features = model.features.module
                dplog = ". Found DataParallel in the model, removed"

            # Manually fix some compatibility issues
            if args.arch in ["resnet56_B"] and args.dataset == "cifar10":
                from model.resnet_cifar10 import resnet56_B

                model = resnet56_B(
                    num_classes=num_classes,
                    num_channels=num_channels,
                    use_bn=args.use_bn,
                    conv_type=args.conv_type,
                )

            logger.info(f"Use the model in ckpt{dplog}")
    else:
        model = model_dict[args.arch](
            num_classes=num_classes,
            num_channels=num_channels,
            use_bn=args.use_bn,
            conv_type=args.conv_type,
        )
        logger.info(f"Create model {args.arch}")

    # Load weights
    if args.pretrained_ckpt is not None:
        from collections import OrderedDict

        state_dict = OrderedDict()
        dplog = ""
        for k, v in ckpt["state_dict"].items():
            if "module." in k:
                k = k.replace("module.", "")
                dplog = ". Found DataParallel in the weights, removed"
            state_dict[k] = v
        model.load_state_dict(state_dict)
        logger.info(f"Load pretrained weights in ckpt successfully{dplog}")

    # Set weight dtype
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    accelerator.dtype = weight_dtype

    # Set to device
    model.to(device=accelerator.device, dtype=accelerator.dtype)
    model.dtype, model.device = accelerator.dtype, accelerator.device
    logger.info(
        f"Model device: {accelerator.device}  dtype: {accelerator.dtype}")

    # Check model accuracy
    criterion = nn.CrossEntropyLoss().to(accelerator.device)
    if args.test_pretrained or args.test_pretrained_only:
        acc1, acc5, loss_test = validate_testset(model)
        logger.info(
            f"Test pretrained ckpt. Acc1 {acc1:.4f} Acc5 {acc5:.4f} TestLoss {loss_test:.4f}"
        )
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(model, (num_channels, input_height, input_width), as_strings=True, print_per_layer_stat=False, verbose=False)
        logger.info(f"Pretrained model: macs: {macs}, params: {params}")
        if args.test_pretrained_only:
            exit(0)

    # Set up optimizer
    optimizer = set_up_optimizer(model)
    logger.info(f"Set up optimizer ({args.solver})")

    if args.ddp:
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader
        )
        logger.info(
            f"DDP: Accelerate set up model / optimizer / train_loader done!")
    else:
        model = nn.DataParallel(model)
        logger.info(f"DP: wrap DataParallel for the model done!")

    start_epoch = 0
    if args.resume:
        assert args.pretrained_ckpt
        start_epoch = ckpt["epoch"] + 1
        logstr = f"Resuming from epoch {start_epoch}"

        # Resume optimizer
        optim = ckpt["optimizer"]
        optimizer.load_state_dict(optim)
        logstr += f". Optimizer resumed"

        # Other resume options
        if get_arg(args, "prune_method"):
            prune_state = ckpt["prune_state"]
            logstr += f". prune_state = {prune_state}"
        logger.info(logstr)

    # Set up lr scheduler
    lr_scheduler = set_up_lr_scheduler(optimizer, start_epoch)

    # Save the model after initialization (useful in lottery ticket hypothesis)
    if args.save_init_model:
        model_save = copy.deepcopy(model).cpu()
        if hasattr(model_save, "module"):
            model_save = model_save.module
        state = {
            "arch": args.arch,
            "model": model_save,
            "state_dict": model_save.state_dict(),
            "ExpID": logger.ExpID,
        }
        save_model(state, mark="init")

    if hasattr(args, "utils") and args.utils.check_kernel_spatial_dist:
        from smilelogging.utils import check_kernel_spatial_dist

        check_kernel_spatial_dist(model)
        exit(0)

    # Structured pruning is basically equivalent to providing a new weight initialization before finetune,
    # so just before training, conduct pruning to obtain a new model.
    if get_arg(args, "prune_method"):
        # Get the original unpruned model statistics
        n_params_original_v2 = 1.0
        n_flops_original_v2 = 1.0

        # Finetune a model directly
        if args.directly_ft_weights:
            ckpt = torch.load(
                args.directly_ft_weights, map_location=torch.device("cpu")
            )
            model = ckpt["model"]
            model.load_state_dict(ckpt["state_dict"])
            model.to(device=accelerator.device, dtype=accelerator.dtype)
            model = nn.DataParallel(model)
            if args.ddp:
                raise NotImplementedError
            prune_state = "finetune"
            logger.info(
                "Load model successfully: '{}'. Epoch = {}. prune_state = '{}'".format(
                    args.directly_ft_weights, args.start_epoch, prune_state
                )
            )

            if args.wg == "weight":
                mask_key = (
                    "masks" if "masks" in ckpt else "mask"
                )  # keep back-compatible
                apply_mask_forward(model, ckpt[mask_key])
                logger.info("Masks restored for unstructured pruning")

        # Prune the model
        if prune_state in ["prune"]:

            class Passer:
                pass

            passer = Passer()  # to pass arguments
            passer.accelerator = accelerator
            passer.test = validate_testset
            passer.train = validate_trainset
            passer.train_raw = validate_trainset_raw
            passer.save = save_model
            passer.criterion = criterion
            passer.optimizer = optimizer
            passer.retrain = retrain
            
            passer.num_channels, passer.input_height, passer.input_width = num_channels, input_height, input_width

            # store the first batch of data to dummy_input
            for ix, (input, _) in enumerate(train_loader):
                input = input.to(device=accelerator.device,
                                 dtype=accelerator.dtype)
                passer.dummy_input = input[0].unsqueeze(0)
                del input
                break

            # ************************* Core pruning function *************************
            from importlib import import_module

            pruner_module = import_module(f"pruner.{args.pruner}_pruner")
            pruner = pruner_module.Pruner(
                model=model,
                loader=loader,
                logger=logger,
                accelerator=accelerator,
                args=args,
                passer=passer,
            )
            model = pruner.prune()  # Get the pruned model
            Logger.passer["pruner"] = pruner  # For later use

            if isinstance(model, tuple):
                model_before_removing_weights, model = model

            if args.wg == "weight":
                Logger.passer["masks"] = pruner.masks
                apply_mask_forward(model, Logger.passer["masks"])
                logger.info(
                    "Apply masks before finetuning to ensure the pruned weights are zero"
                )
            # *************************************************************************

            # Get model statistics of the pruned model
            n_params_now_v2 = 1.0
            n_flops_now_v2 = 1.0
            logger.info(
                "n_params_original_v2: {:>9.6f}M, n_flops_original_v2: {:>9.6f}G".format(
                    n_params_original_v2 / 1e6, n_flops_original_v2 / 1e9
                )
            )
            logger.info(
                "n_params_now_v2:      {:>9.6f}M, n_flops_now_v2:      {:>9.6f}G".format(
                    n_params_now_v2 / 1e6, n_flops_now_v2 / 1e9
                )
            )
            ratio_param = (
                n_params_original_v2 - n_params_now_v2
            ) / n_params_original_v2
            ratio_flops = (n_flops_original_v2 -
                           n_flops_now_v2) / n_flops_original_v2
            compression_ratio = 1.0 / (1 - ratio_param)
            speedup_ratio = 1.0 / (1 - ratio_flops)
            format_str = (
                "reduction ratio -- params: {:>5.2f}% (compression ratio {:>.2f}x), flops: {:>5.2f}% ("
                "speedup ratio {:>.2f}x) "
            )
            logger.info(
                format_str.format(
                    ratio_param * 100,
                    compression_ratio,
                    ratio_flops * 100,
                    speedup_ratio,
                )
            )

            from ptflops import get_model_complexity_info
            macs, params = get_model_complexity_info(model, (num_channels, input_height, input_width), as_strings=True, print_per_layer_stat=False, verbose=False)
            logger.info(f"Pruned model: macs: {macs}, params: {params}")

            # Test the just pruned model
            t1 = time.time()
            acc1, acc5, loss_test = validate_testset(model)
            logstr = "Acc1 %.4f Acc5 %.4f TestLoss %.4f" % (
                acc1, acc5, loss_test)
            
            # calculate the pruned_train_loss for the just pruned model
            acc1_train, acc5_train, pruned_train_loss = validate_trainset(model)
            logger.info("[fengsicheng]: pruned_train_loss %.6f" % (pruned_train_loss))
            
            if args.dataset not in ["imagenet"] and args.test_trainset:
                acc1_train, acc5_train, loss_train = validate_trainset(model)
                logstr += " TrainAcc1 %.4f TrainAcc5 %.4f TrainLoss %.4f" % (
                    acc1_train,
                    acc5_train,
                    loss_train,
                )
            logstr += " (test_time %.2fs) Just got pruned model, about to finetune" % (
                time.time() - t1
            )
            logger.info(logstr)

            # Save the just pruned model
            model_save = copy.deepcopy(model).cpu()
            if hasattr(model_save, "module"):
                model_save = model_save.module
            state = {
                "arch": args.arch,
                "model": model_save,
                "state_dict": model_save.state_dict(),
                "acc1": acc1,
                "acc5": acc5,
                "ExpID": logger.ExpID,
                "pruned_wg": pruner.pruned_wg,
                "kept_wg": pruner.kept_wg,
            }
            if args.wg == "weight":
                state["masks"] = Logger.passer["masks"]
            save_model(state, mark="just_finished_prune")
    # ---

    # Before retraining, we may reinit the weights by some rule
    if get_arg(args, "reinit"):
        mask = Logger.passer["masks"] if args.wg == "weight" else None
        model = reinit_model(model, args=args, mask=mask, print_fn=logger.info)
        acc1, acc5, loss_test = validate_testset(model)
        logger.info(
            f"Acc1 {acc1:.4f} Acc5 {acc5:.4f} TestLoss {loss_test:.4f} "
            f"-- after reinitializing the just pruned model"
        )

        # Save weights
        model_save = copy.deepcopy(model).cpu()
        if hasattr(model_save, "module"):
            model_save = model_save.module
        state = {
            "arch": args.arch,
            "model": model_save,
            "state_dict": model_save.state_dict(),
            "acc1": acc1,
            "acc5": acc5,
            "ExpID": logger.ExpID,
            "pruned_wg": pruner.pruned_wg,
            "kept_wg": pruner.kept_wg,
        }
        if args.wg == "weight":
            state["masks"] = Logger.passer["masks"]
        save_model(state, mark="reinit")
        logger.info(f"Reinitialized model saved")
        del model_save, state

    if args.rescale:
        logger.info(f"Rescale model weights, begin:")
        model = rescale_model(model, args.rescale)
        logger.info(f"Rescale model weights, done!")

    if get_arg(args, "prune_method"):
        optimizer = set_up_optimizer(model)
        logger.info(f"After pruning, about to retrain, reset optimizer")

    # Train
    timer = Timer(args.epochs - start_epoch)
    best_acc1, best_epoch = 0, 0
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # LR scheduling
        lr_scheduler(optimizer, epoch)

        # Get LR
        lr = get_lr(optimizer)

        # One epoch train
        logger.info("Set lr = %s @ Epoch %d (Start)" % (lr, epoch + 1))
        one_epoch_train(train_loader, model, criterion, optimizer, epoch)

        # Test
        acc1, acc5, loss_test = validate_testset(model)
        acc1_train, acc5_train, loss_train = -1, -1, -1

        if args.dataset not in ["imagenet"]:  # imagenet is too costly, not test
            acc1_train, acc5_train, loss_train = validate_trainset(model)
        elif args.dataset == "imagenet":
            acc1_train, acc5_train, loss_train = validate_trainset_raw(model)
        # Log down metrics
        num_steps_per_epoch = len(train_loader)
        step = (epoch + 1) * num_steps_per_epoch
        accelerator.log({"acc1": acc1}, step=step)
        accelerator.log({"acc5": acc5}, step=step)
        accelerator.log({"loss_test": loss_test}, step=step)

        # Print metrics
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
            best_epoch = epoch
            best_loss_train = loss_train
            best_loss_test = loss_test    

            model_save = copy.deepcopy(model).cpu()
            if hasattr(model_save, "module"):
                model_save = model_save.module
            state = {
                "arch": args.arch,
                "model": model_save,
                "state_dict": model_save.state_dict(),
                "acc1": best_acc1,
            }
            save_model(state, is_best=True)          
        acclog = (
            f"Epoch {epoch} Acc1 {acc1:.4f} Acc5 {acc5:.4f} TestLoss {loss_test:.4f} "
            f"BestAcc1 {best_acc1:.4f} BestAcc1Epoch {best_epoch} LR {lr}"
        )
        if acc1_train != -1:
            acclog.replace(
                "BestAcc1",
                f"TrainAcc1 {acc1_train:.4f} TrainAcc5 {acc5_train:.4f} "
                f"TrainLoss {loss_train:.4f} BestAcc1",
            )
        logger.info(acclog, color="green")
        logger.info(f"Predicted finish time: {timer()}")
        logger.info(
            "[fengsicheng]: final_train_loss %.6f final_test_loss %.6f final_test_acc %.6f"
            % (best_loss_train, best_loss_test, best_acc1)
        )

    accelerator.end_training()


def one_epoch_train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device=accelerator.device, dtype=accelerator.dtype)
        target = target.to(device=accelerator.device).long()

        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # Compute gradient and update params
        if args.ddp:
            accelerator.backward(loss)
        else:
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # After update, zero out pruned weights
        if get_arg(args, "prune_method") and args.wg == "weight":
            from smilelogging import Logger

            masks = Logger.passer["masks"]
            apply_mask_forward(model, masks)

        # Utils: check gradient norm
        if hasattr(args, "utils") and args.utils.check_grad_norm:
            from smilelogging.utils import check_grad_norm

            if i % args.print_interval == 0:
                logger.info("")
                logger.info(
                    f"(** Start check_grad_norm. Epoch {epoch} Step {i} **)")
                check_grad_norm(model)
                logger.info(f"(** End check_grad_norm **)")
                logger.info("")

        # Utils: check gradient stats
        if hasattr(args, "utils") and args.utils.check_grad_stats:
            from smilelogging.utils import check_grad_stats

            if i % args.print_interval == 0:
                logger.info("")
                logger.info(
                    f"(** Start check_grad_stats. Epoch {epoch} Step {i} **)")
                check_grad_stats(model)
                logger.info(f"(** End check_grad_stats **)")
                logger.info("")

        # Utils: Check grad history
        if hasattr(args, "utils") and args.utils.check_grad_history:
            grad_history = Logger.passer["grad_history"]
            pruner = Logger.passer["pruner"]
            assert args.wg == "weight"
            for name, module in model.named_modules():
                if name in grad_history:
                    grad = module.weight.grad.data.clone().cpu()
                    # grad = effective_grad[name].clone().cpu()
                    grad_history[name] += [grad]

        # Utils: check weight stats
        if hasattr(args, "utils") and args.utils.check_weight_stats:
            from smilelogging.utils import check_weight_stats

            if i % args.print_interval == 0:
                logger.info("")
                logger.info(
                    f"(** Start check_weight_stats. Epoch {epoch} Step {i} **)")
                check_weight_stats(model)
                logger.info(f"(** End check_weight_stats **)")
                logger.info("")

        # @mst: check Jacobian singular value (JSV)
        if args.jsv_interval == -1:
            args.jsv_interval = len(
                train_loader
            )  # default: check jsv at the last iteration
        if args.jsv_loop and (i + 1) % args.jsv_interval == 0:
            from smilelogging import Logger

            jsv, jsv_diff, cn = get_jacobian_singular_values(
                model,
                train_loader,
                num_classes=Logger.passer["num_classes"],
                n_loop=args.jsv_loop,
                rand_data=args.jsv_rand_data,
            )
            logger.info(
                "JSV_mean %.4f JSV_std %.4f JSV_std/mean %.4f JSV_max %.4f JSV_min %.4f "
                "Condition_Number_mean %.4f JSV_diff_mean %.4f JSV_diff_std %.4f -- Epoch %d Iter %d"
                % (
                    np.mean(jsv),
                    np.std(jsv),
                    np.std(jsv) / np.mean(jsv),
                    np.max(jsv),
                    np.min(jsv),
                    np.mean(cn),
                    np.mean(jsv_diff),
                    np.std(jsv_diff),
                    epoch,
                    i,
                )
            )

            # For easy report
            Logger.passer["JSV_mean"] += [np.mean(jsv)]
            Logger.passer["JSV_std/mean"] += [np.std(jsv) / np.mean(jsv)]
            if len(Logger.passer["JSV_mean"]) == 11:
                logstr = []
                for x, y in zip(
                    Logger.passer["JSV_mean"], Logger.passer["JSV_std/mean"]
                ):
                    logstr += ["%.4f/%.2f" % (x, y)]
                logstr = " | ".join(logstr) + " |"  # For markdown
                logger.info(f"First 10-epoch JSV_mean, JSV_std/mean: {logstr}")

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_interval == 0 and accelerator.is_main_process:
            progress.display(i)


def save_model(ckpt, is_best=False, mark=""):
    out = os.path.join(logger.weights_path, "ckpt.pth")
    torch.save(ckpt, out)
    if is_best:
        out_best = os.path.join(logger.weights_path, "ckpt_best.pth")
        torch.save(ckpt, out_best)
    if mark:
        out_mark = os.path.join(logger.weights_path,
                                "ckpt_{}.pth".format(mark))
        torch.save(ckpt, out_mark)


# Zero out pruned weights for unstructured pruning
def apply_mask_forward(model, masks):
    for name, m in model.named_modules():
        if name in masks:
            if isinstance(m, nn.MultiheadAttention):
                m.in_proj_weight.data.mul_(masks[name].cuda())
            else:
                m.weight.data.mul_(masks[name].cuda())


def adjust_learning_rate_v2(optimizer, epoch, iteration, num_iter):
    r"""More advanced LR scheduling. Refers to d-li14 MobileNetV2 ImageNet implementation:
    https://github.com/d-li14/mobilenetv2.pytorch/blob/1733532bd43743442077326e1efc556d7cfd025d/imagenet.py#L374
    """
    assert hasattr(args, "advanced_lr")

    warmup_iter = (
        args.advanced_lr.warmup_epoch * num_iter
    )  # num_iter: num_iter_per_epoch
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if epoch < args.advanced_lr.warmup_epoch:
        lr = args.lr * current_iter / warmup_iter
    else:
        if args.advanced_lr.lr_decay == "step":
            lr = args.lr * (
                args.advanced_lr.gamma
                ** ((current_iter - warmup_iter) / (max_iter - warmup_iter))
            )
        elif args.advanced_lr.lr_decay == "cos":
            lr = (
                args.lr
                * (
                    1
                    + math.cos(
                        math.pi
                        * (current_iter - warmup_iter)
                        / (max_iter - warmup_iter)
                    )
                )
                / 2
            )
        elif args.advanced_lr.lr_decay == "linear":
            lr = args.lr * (1 - (current_iter - warmup_iter) /
                            (max_iter - warmup_iter))
        elif args.advanced_lr.lr_decay == "schedule":
            count = sum([1 for s in args.advanced_lr.schedule if s <= epoch])
            lr = args.lr * pow(args.advanced_lr.gamma, count)
        else:
            raise ValueError("Unknown lr mode {}".format(
                args.advanced_lr.lr_decay))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

if __name__ == "__main__":
    # Check data
    data_script = "scripts/set_up_data.sh"
    if os.path.exists(data_script):
        os.system(f"sh {data_script} {args.dataset}")

    main()
