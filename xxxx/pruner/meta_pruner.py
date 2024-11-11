import copy
import numpy as np
import re
from fnmatch import fnmatch
from collections import OrderedDict

import torch
import torch.nn as nn

from .layer import LayerBuilder
from .prune_utils import (
    get_masks_for_pruned_layers,
    get_pr_for_model,
    get_kept_filter_channel,
    replace_module,
)
from .prune_utils import pick_pruned_weights_for_model
from smilelogging.slutils import red, green, yellow, blue


# TODO: This modulename attr may conflict with others. A safer way?
def register_modulename(model):
    for name, m in model.named_modules():
        m.module_name = name


class MetaPruner:
    def __init__(self, model, loader, logger, accelerator, args, passer=None):
        """The metaclass of pruners. Specific pruners should inherit this class.
        Args:
            model (nn.Module)
            loader (pytorch dataloader): it should have two attributes: .test_loader, .train_loader
            logger: Logger in smilelogging
            accelerator: HuggingFace Accelerate library; used to easily set device
            args (arguments): used to pass some important pruning-related information (e.g., layerwise pruning ratio)
            passer: other useful info (like test fn) is passed through this passer class
        """
        self.model = model
        self.loader = loader
        self.logger = logger
        self.args = self.__check_args(args)
        self.accelerator = accelerator
        if not hasattr(self.accelerator, "dtype"):
            self.accelerator.dtype = torch.float32
        register_modulename(self.model)

        # Set some attributes via passer.
        self.dummy_input = (
            passer.dummy_input if hasattr(passer, "dummy_input") else None
        )
        self.test = passer.test if hasattr(passer, "test") else None
        self.save = passer.save if hasattr(passer, "save") else None
        self.criterion = passer.criterion if hasattr(passer, "criterion") else None

        # Set up some constants to indicate which learnable learnable_layers are
        # considered for pruning.
        self.prunable_types = (
            nn.Conv2d,
            nn.Conv1d,
            nn.Linear,
            nn.Embedding,
            nn.MultiheadAttention,
        )
        self.norm_types = (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)
        self.learnable_types = self.prunable_types + self.norm_types
        self.logger.info(
            f"Prunable layer types: {[x.__name__ for x in self.prunable_types]}"
            + " -- learnable_layers of these types will be accounted for in pruning"
        )

        # Use hooks to build layer dependency; note this needs at least once model
        # forward to materialize the hooks, otherwise, these variables are empty.
        self.logger.info(green("[build layer dep]") + "...")
        self.layer_builder = LayerBuilder(
            model=model,
            learnable_types=self.learnable_types,
            prunable_types=self.prunable_types,
            dummy_input=self.dummy_input,
            print_fn=self.logger.info,
        )
        msg = (
            " (only hooks done, still wait for model forward)"
            if not self.layer_builder.layer_build_done
            else ""
        )
        self.logger.info(green("[build layer dep]") + f" -- done!{msg}")
        self.learnable_layers = self.layer_builder.learnable_layers
        self.constrained_groups = self.layer_builder.constrained_groups
        self._max_len_ix = self.layer_builder._max_len_ix  # TODO: optimize these?
        self._max_len_name = self.layer_builder._max_len_name
        self.layer_print_prefix = self.layer_builder.print_prefix
        self.logger.info("Constrained groups:", self.constrained_groups, color="green")

        # Anchor the layer index to module.
        if self.layer_builder.layer_build_done:
            for n, m in model.named_modules():
                if n in self.learnable_layers:
                    m.index = self.learnable_layers[n].index
        self.logger.info("Anchor layer index as an attribute of module.")

        # Set up prunable layers.
        self.prunable_layers = OrderedDict()
        for n, l in self.learnable_layers.items():
            if isinstance(l.module, self.prunable_types):
                self.prunable_layers[n] = l

        # Check module consistency.
        self._check_module_consistency(loc="After building layers in meta_pruner")

        # Others
        self.total_iter = None  # num of total training iterations (if any).

    def __check_args(self, args):
        """The pruner requires some basic arguments, so check them here to make sure
        they have been provided."""
        if not hasattr(args, "prune_schedule"):
            args.prune_schedule = "simu"
        if not hasattr(args, "reinit_layers"):
            args.reinit_layers = ""
        if not hasattr(args, "skip_layers"):
            args.skip_layers = ""
        if not hasattr(args, "compare_mode"):
            args.compare_mode = "local"
        return args

    def __set_up_pruned_layers(self):
        """All pruned layers form a list; prune_index means the index of each pruned
        layer in that list.

        prune_order: the order to prune different learnable_layers. If all learnable_layers are pruned simultaneously,
        prune_order = 0; if a layer is not prunable, prune_order = None.
        """
        pruned_layers, prune_interval = OrderedDict(), 0
        for name, layer in self.prunable_layers.items():
            if self.pr[name] > 0:
                layer.prune_index = len(pruned_layers)
                pruned_layers[name] = layer
            else:
                layer.prune_index = None
        num_pruned_layers = len(pruned_layers)

        # Set prune_order.
        for name, layer in self.prunable_layers.items():
            if self.pr[name] > 0:
                if self.args.prune_schedule in ["simu"]:
                    layer.prune_order = 0
                elif re.match("ascend_\d+", self.args.prune_schedule):
                    layer.prune_order = layer.prune_index
                    prune_interval = int(self.args.prune_schedule.split("_")[1])
                elif re.match("descend_\d+", self.args.prune_schedule):
                    layer.prune_order = num_pruned_layers - 1 - layer.prune_index
                    prune_interval = int(self.args.prune_schedule.split("_")[1])
                else:
                    raise NotImplementedError
            else:
                layer.prune_order = None
        return pruned_layers, num_pruned_layers, prune_interval

    def prune_init(self):
        r"""Set up pruning related initializations, such as PR."""
        # Get skip learnable_layers (if any)
        self.skip_layers = []
        for name in self.prunable_layers:
            for p in self.args.skip_layers:
                if fnmatch(name, p):  # TODO: replace all fnmatch with re.match
                    self.skip_layers += [name]
        if len(self.skip_layers):
            self.logger.info(f"Get skipped learnable_layers: {self.skip_layers}")

        # Set up pr for all prunable learnable_layers
        self.raw_pr = get_pr_for_model(
            prunable_layers=self.prunable_layers,
            base_pr=self.args.stage_pr,
            skip_layers=self.skip_layers,
            compare_mode=self.args.compare_mode,
            print_fn=self.logger.info,
        )

        # Handle the case of constrained learnable_layers
        if hasattr(self.args, "not_prune_cnst") and self.args.not_prune_cnst:
            cnst_layers = self.constrained_groups.values()
            for k, v in self.raw_pr.items():
                if k in cnst_layers:
                    self.raw_pr[k] = 0
            self.logger.info(
                f"Set the pruning ratio of all constrained learnable_layers to 0, "
                f"given --not_prune_cnst is used"
            )

        # The official interface of pruning ratio
        self.pr = copy.deepcopy(self.raw_pr)

        # For final to-prune learnable_layers (prunable and pr > 0), set up their related attributes
        (
            self.pruned_layers,
            self.num_pruned_layers,
            self.layerwise_prune_interval,
        ) = self.__set_up_pruned_layers()

    def get_kept_weight_groups(
        self, align_constrained=False, criterion="mag", sort_mode=None, is_print=True
    ):
        """Get kept/pruned weight groups for the model."""
        self._check_module_consistency(loc="Before picking pruned weights for model")

        # Update args
        if sort_mode is None:
            sort_mode = self.args.pick_pruned

        if hasattr(self.args, "inherit_pruned") and self.args.inherit_pruned == "index":
            import os

            assert os.path.exists(self.args.stage_pr)
            ckpt = torch.load(self.args.stage_pr, map_location=torch.device("cpu"))
            pr = self.raw_pr
            pruned_wg, kept_wg = ckpt["pruned_wg"], ckpt["kept_wg"]
            scheme = f"inheriting_existing_pruned_indices"
            if is_print:
                self.logger.info(
                    f"==> Load base_pr model successfully and inherit "
                    f"its pruned indices: '{self.args.stage_pr}'"
                )
        else:
            # ************************* Core pruning function **************************
            pr, pruned_wg, kept_wg = pick_pruned_weights_for_model(
                self.prunable_layers,
                self.raw_pr,
                wg=self.args.wg,
                criterion=criterion,
                compare_mode=self.args.compare_mode,
                sort_mode=sort_mode,
                constrained=self.constrained_groups,
                align_constrained=align_constrained,
                print_fn=self.logger.info if is_print else None,
            )

            scheme = f"{self.args.prune_method, criterion, self.args.pick_pruned}"
            # ***************************************************************************

        # Print
        if is_print:
            self.logger.info(f"*********** Get pruned wg ***********")
            for name, layer in self.pruned_layers.items():
                logtmp = f"{self.layer_print_prefix[name]} -- Got pruned wg by {scheme}, pr {pr[name]}"
                ext = f" -- Constrained layer" if layer.cnst_grp != -1 else ""
                self.logger.info(logtmp + ext)
            self.logger.info(f"*************************************")
            # TODO-@mst: here, the printed info should be improved

        # Make pruning ratio an attr of layer
        for name, v in pr.items():
            self.prunable_layers[name].pr = v
            self.learnable_layers[name].pr = v
            if v > 0:
                self.pruned_layers[name].pr = v
        return pr, pruned_wg, kept_wg

    def prune_and_build_new_model(self, name_include=""):
        """Physically remove unimportant weights and build a new slimmer model."""
        if self.test is not None:
            acc1_before, *_ = self.test(self.model)

        # Unstructured pruning.
        if self.args.wg in ["weight", "maskfilter"]:
            self.masks = get_masks_for_pruned_layers(
                pruned_layers=self.pruned_layers,
                pruned_wg=self.pruned_wg,
                wg=self.args.wg,
            )

            # Reinit designated learnable_layers
            for name, m in self.model.named_modules():
                if name in self.learnable_layers:
                    reinit = False
                    for rl in self.args.reinit_layers:
                        if fnmatch(name, rl):
                            reinit = True
                            break
                    if reinit:
                        m.reset_parameters()
                        self.logger.info(
                            f"Layer {name} is reinitialized when building the new model!"
                        )
            return

        # Structured pruning
        new_model = copy.deepcopy(self.model)

        # Get all modules
        all_modules = {n: m for n, m in self.model.named_modules()}

        # Iterate all learnable_layers (including norm layers like BN)
        for name, layer in self.learnable_layers.items():
            # Used for pruning a part of the model
            if name_include not in name:
                continue

            # Get the module
            m = all_modules[name]

            shape = self.learnable_layers[name].shape
            n_filters, n_channels = shape[:2] if len(shape) > 1 else (shape[0], 0)
            current_layer_pr = self.pr[name] if name in self.pr else self.pr[layer.last]

            # Core fn: Get the indices of kept filters and channels
            logstr = f"[Rebuild] {self.layer_print_prefix[name]}"
            if self.args.prune_with_hooks:
                kept_filter, kept_chl = (
                    self.kept_filters[name],
                    self.kept_channels[name],
                )
            if len(kept_filter) == n_filters and len(kept_chl) == n_channels:
                new_layer = m
            else:
                logstr += (
                    f"  [#F: {n_filters:>4d} -> {len(kept_filter):<4d}]"
                    f"  [#C: {n_channels:>4d} -> {len(kept_chl):<4d}]"
                    f"  [PR: {current_layer_pr:.4f}]"
                    f"  [last: {layer.last}]"
                )
                self.logger.info(logstr, unprefix=True)
                if len(kept_chl) > n_channels:
                    self.logger.info(
                        f"#kept_chls ({len(kept_chl)}) > #original_chls ({n_channels})! Please check!",
                        level="error",
                    )

                # Reinit the current layer, TODO-@mst: move this to layer building and set layer.reinit
                reinit = False
                for rl in self.args.reinit_layers:
                    if fnmatch(name, rl):
                        reinit = True
                        break

                # Copy weight and bias, need case-by-case implementation!
                IMPLEMENTED_TYPES = (
                    nn.Conv2d,
                    nn.Linear,
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.GroupNorm,
                    nn.LayerNorm,
                )
                assert isinstance(
                    m, IMPLEMENTED_TYPES
                ), "This module type has not been implemented. You need to implement this yourself!"
                bias = hasattr(m, "bias") and m.bias is not None
                if isinstance(m, nn.Conv2d):
                    new_layer = nn.Conv2d(
                        len(kept_chl),
                        len(kept_filter),
                        m.kernel_size,
                        m.stride,
                        m.padding,
                        m.dilation,
                        m.groups,
                        bias,
                    )

                    self.logger.info(f"kept_filter: {kept_filter}")
                    self.logger.info(f"kept_chl: {kept_chl}")
                    # if max(kept_filter) < m.weight.data.shape[0] and max(kept_chl) < m.weight.data.shape[1]:
                    #     kept_weights = m.weight.data[kept_filter][:, kept_chl, :, :]
                    #     print(f"kept_weights shape: {kept_weights.shape}")
                    # else:
                    #     print("Error: Index out of range or data type issue")

                    kept_weights = m.weight.data[kept_filter][:, kept_chl, :, :]

                    if not reinit:
                        new_layer.weight.data.copy_(
                            kept_weights
                        )  # load weights into the new module
                        if bias:
                            kept_bias = m.bias.data[kept_filter]
                            new_layer.bias.data.copy_(kept_bias)
                    else:
                        self.logger.info(
                            f"Layer {name} is reinited when building the new model!"
                        )

                elif isinstance(m, nn.Linear):
                    new_layer = nn.Linear(
                        in_features=len(kept_chl),
                        out_features=len(kept_filter),
                        bias=bias,
                    )
                    kept_weights = m.weight.data[kept_filter][:, kept_chl]

                    if not reinit:
                        new_layer.weight.data.copy_(
                            kept_weights
                        )  # load weights into the new module
                        if bias:
                            kept_bias = m.bias.data[kept_filter]
                            new_layer.bias.data.copy_(kept_bias)
                    else:
                        self.logger.info(
                            f"Layer {name} is reinited when building the new model!"
                        )

                elif isinstance(m, nn.BatchNorm1d):
                    # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
                    new_layer = nn.BatchNorm1d(
                        num_features=len(kept_filter),
                        eps=m.eps,
                        momentum=m.momentum,
                        affine=m.affine,
                        track_running_stats=m.track_running_stats,
                    )

                    # Copy weight and bias.
                    new_layer.weight.data.copy_(m.weight.data[kept_filter])
                    new_layer.bias.data.copy_(m.bias.data[kept_filter])

                    # Copy running stats.
                    new_layer.running_mean.data.copy_(m.running_mean[kept_filter])
                    new_layer.running_var.data.copy_(m.running_var[kept_filter])
                    new_layer.num_batches_tracked.data.copy_(m.num_batches_tracked)

                elif isinstance(m, nn.BatchNorm2d):
                    # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
                    new_layer = nn.BatchNorm2d(
                        num_features=len(kept_filter),
                        eps=m.eps,
                        momentum=m.momentum,
                        affine=m.affine,
                        track_running_stats=m.track_running_stats,
                    )

                    # Copy weight and bias.
                    new_layer.weight.data.copy_(m.weight.data[kept_filter])
                    new_layer.bias.data.copy_(m.bias.data[kept_filter])

                    # Copy running stats.
                    new_layer.running_mean.data.copy_(m.running_mean[kept_filter])
                    new_layer.running_var.data.copy_(m.running_var[kept_filter])
                    new_layer.num_batches_tracked.data.copy_(m.num_batches_tracked)

                elif isinstance(m, nn.GroupNorm):
                    kept_filter = list(range(m.num_channels))
                    new_layer = nn.GroupNorm(
                        num_groups=m.num_groups,
                        num_channels=len(kept_filter),
                        eps=m.eps,
                        affine=m.affine,
                    )

                    # copy bn weight and bias
                    new_layer.weight.data.copy_(m.weight.data[kept_filter])
                    new_layer.bias.data.copy_(m.bias.data[kept_filter])

                elif isinstance(m, nn.LayerNorm):
                    # See https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
                    new_layer = nn.LayerNorm(
                        len(kept_filter),
                        eps=m.eps,
                        elementwise_affine=m.elementwise_affine,
                    )

                    # copy bn weight and bias
                    new_layer.weight.data.copy_(m.weight.data[kept_filter])
                    new_layer.bias.data.copy_(m.bias.data[kept_filter])

                    raise NotImplementedError

            # Load the new_layer into the new_model
            replace_module(new_model, name, new_layer)

        new_model.to(device=self.accelerator.device, dtype=self.accelerator.dtype)
        
        del all_modules
        register_modulename(new_model)

        if self.test is not None:
            acc1_after, *_ = self.test(new_model)
            self.logger.info(
                "Acc1 %11.8f -- Before prune_and_build_new_model" % (acc1_before),
                color="green",
            )
            self.logger.info(
                "Acc1 %11.8f -- After  prune_and_build_new_model" % (acc1_after),
                color="green",
            )

        return new_model

    def _check_weight(self, model=None, layer_prefix=""):
        r"""Check weights and their masks."""
        if model is None:
            model = self.model

        # Select learnable_layers to print
        selected_layers = self.select_layers_for_print()
        masks = get_masks_for_pruned_layers(
            pruned_layers=self.pruned_layers, pruned_wg=self.pruned_wg, wg=self.args.wg
        )
        num_print = 10
        step = "(Step %s)" % self.total_iter if self.total_iter is not None else ""
        self.logger.info(f"Check the weights (mask) of each layer: {step}")
        for name, m in model.named_modules():
            name = layer_prefix + name
            if name in selected_layers:
                weight_ = m.weight.data.flatten()  # TODO-@mst: Add MHA support
                mask_ = masks[name].flatten()
                np.random.seed(0)
                ix = np.random.choice(len(mask_), num_print)
                wstr = " ".join(
                    [
                        f"{x:.5f}({int(y.item())})".rjust(11)
                        for x, y in zip(weight_[ix], mask_[ix])
                    ]
                )
                self.logger.info(
                    f"{self.layer_print_prefix[name].strip()}: {wstr}", unprefix=True
                )

    def apply_masks(self, model):
        """Apply masks (derived by this pruner) to a <model>.

        Args:
            model: input dense model

        Returns:
            model: output sparse model
        """
        if self.args.wg in ["weight", "maskfilter"]:
            masks = get_masks_for_pruned_layers(
                pruned_layers=self.pruned_layers,
                pruned_wg=self.pruned_wg,
                wg=self.args.wg,
            )

            for n, m in model.named_modules():
                if n in self.pruned_layers:
                    # TODO-@mst: It seems this is a generic problem to decide the param name in a layer (module)
                    if hasattr(
                        m, "weight"
                    ):  # For CNN: naive modules like Conv2d/Linear/BN
                        data = m.weight.data
                    elif (
                        hasattr(m, "in_proj_weight") and m.in_proj_weight is not None
                    ):  # For MHA: qkv same dim
                        data = m.in_proj_weight.data
                    elif (
                        hasattr(m, "q_proj_weight") and m.q_proj_weight is not None
                    ):  # For MHA: qkv not same dim
                        assert None not in (m.k_proj_weight, m.v_proj_weight)
                        raise NotImplementedError
                    else:
                        raise NotImplementedError

                    mask_ = masks[n].to(device=data.device)
                    data.copy_(data * mask_)
            self.logger.info(f"==> Masks applied to model weights!")
        return model

    def _check_bn(self, model=None):
        if model is None:
            model = self.model

        selected_layers = self.select_layers_for_print()
        assert self.args.wg == "filter"
        num_print = 10
        all_layers = [
            n for n, _ in self.model.named_modules()
        ]  # Conv, ReLU, BN, FC, etc.
        self.logger.info("Check the BN learnable_layers:")
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                # Get the associating conv layer of this BN layer
                ix = all_layers.index(name)
                for k in range(ix - 1, -1, -1):
                    if all_layers[k] in self.learnable_layers:
                        last_conv = all_layers[k]
                        break
                if last_conv not in selected_layers:
                    continue
                mask_ = [0] * m.weight.data.size(0)
                for i in self.kept_wg[last_conv]:
                    mask_[i] = 1
                wstr = " ".join(
                    [
                        f"{x:.3f}({y})".rjust(9)
                        for x, y in zip(m.weight.data[:num_print], mask_[:num_print])
                    ]
                )
                bstr = " ".join(
                    [
                        f"{x:.3f}({y})".rjust(9)
                        for x, y in zip(m.bias.data[:num_print], mask_[:num_print])
                    ]
                )
                self.logger.info(
                    f"{self.learnable_layers[last_conv].index} {last_conv} BN weight: {wstr}",
                    unprefix=True,
                )
                self.logger.info(
                    f"{self.learnable_layers[last_conv].index} {last_conv} BN bias  : {bstr}",
                    unprefix=True,
                )

    def _check_mask_overlap_with_MP(self):
        selected_layers = self.select_layers_for_print()

        _, pruned_wg_MP_temp, kept_wg_MP_temp = self.get_kept_weight_groups(
            self.args.align_constrained, sort_mode="min", is_print=False
        )
        self.logger.info("-" * 20 + " Check mask overlap " + "-" * 20)
        for name, layer in self.learnable_layers.items():
            if name in selected_layers:
                overlap = [
                    x for x in pruned_wg_MP_temp[name] if x in self.pruned_wg[name]
                ]
                r = len(overlap) / len(pruned_wg_MP_temp[name])
                self.logger.info(
                    f"{layer.index} {name} -- Overlapped pruned_wg: {len(overlap)} ({r * 100:.2f}%) "
                    f"PR: {self.pr[name]}"
                )
        self.logger.info("-" * 20 + "--------------------" + "-" * 20)

    def select_layers_for_print(self):
        """A model typically has so many layer. We do not want to print info of all layers, so here select some layers
        for print.

        Returns:
            selected_layers: a list of selected layer names
        """
        num = self.num_pruned_layers
        pruned_layers = list(self.pruned_layers.keys())
        selected_ix = [0, int(num * 0.25), int(num * 0.5), int(num * 0.75), num - 1]
        selected_layers = [pruned_layers[i] for i in selected_ix]
        return selected_layers

    def _check_module_consistency(self, model=None, layers=None, loc=""):
        """Check if the modules in self.model and those in self.layers are the same
        Args:
            model: model
            layers: learnable_layers
            loc: check location
        """
        if model is None:
            model = self.model
        if layers is None:
            layers = self.learnable_layers

        consistent = True
        for name, module in model.named_modules():
            if name in layers:
                same = module is layers[name].module
                if not same:
                    consistent = False
                    self.logger.info(
                        f"Module consistency check failed: {name}", color="red"
                    )

        consistent = "[%s]" % consistent
        consistent = green(consistent) if consistent else red(consistent)
        msg = blue(f"{loc}, module consistency check passed?")
        self.logger.info(f"{msg} {consistent}")
        if not consistent:
            exit(1)
