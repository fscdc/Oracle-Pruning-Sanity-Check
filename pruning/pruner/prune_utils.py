import os.path
from collections import OrderedDict
from fnmatch import fnmatch
import re
import sys
from typing import Callable
import math
import copy

import numpy as np
from scipy.linalg import circulant

import torch
import torch.nn as nn

tensor2list = lambda x: x.data.cpu().numpy().tolist()
tensor2array = lambda x: x.data.cpu().numpy()
totensor = lambda x: torch.Tensor(x)


def _get_pr_by_name_matching(pr_patterns: dict, name: str):
    """
    Args:
        pr_patterns: a dict whose key is layer name pattern, whose value is the associated pruning ratio
        name: layer name

    Returns:
        pr: pruning ratio, float, in range [0, 1)

    """
    pr = 0  # Default pr = 0
    for p in pr_patterns:
        if fnmatch(
            name, p
        ):  # Note, if a layer matches multiple pattern, its pr will be up to the last pattern
            pr = pr_patterns[p]
    assert (
        0 <= pr < 1
    ), f"Wrong pruning ratio for layer '{name}': {pr} -- must be in range [0, 1)"
    return pr


def get_pr_for_model(
    prunable_layers, base_pr, skip_layers=[], compare_mode="local", print_fn=print
):
    r"""Get layer-wise pruning ratio for a model. There are multiple choices to set up layerwise pr, as addressed
    exclusively below.
    """
    pr = OrderedDict()
    assert compare_mode in ["global", "local"]
    if isinstance(base_pr, str):
        assert os.path.exists(
            base_pr
        ), f'Not found ckpt "{base_pr}" used to specify layerwise pruning ratio'
        ckpt = torch.load(base_pr, map_location=torch.device("cpu"))
        pruned, kept = ckpt["pruned_wg"], ckpt["kept_wg"]
        for name in pruned:
            num_pruned, num_kept = len(pruned[name]), len(kept[name])
            pr[name] = float(num_pruned) / (num_pruned + num_kept)
        print_fn(
            f"==> Load base_pr model successfully and inherit its pruning ratios: '{base_pr}'."
        )

    elif isinstance(base_pr, (float, list)):
        if compare_mode in ["global"]:
            assert isinstance(base_pr, float) and 0 < base_pr < 1
            pr["model"] = base_pr
        for name, layer in prunable_layers.items():
            # A small positive value to indicate this layer will be considered for pruning; will be replaced
            pr[name] = 1e-20 if compare_mode == "global" else base_pr[layer.index]
            if name in skip_layers:
                pr[name] = 0
        print_fn(
            f"==> Get layerwise pruning ratios for the whole model (they may be updated later)."
        )

    elif isinstance(base_pr, dict):  # Index layer by name matching
        for name, layer in prunable_layers.items():
            pr[name] = _get_pr_by_name_matching(base_pr, name)
            if name in skip_layers:
                pr[name] = 0
        print_fn(
            f"==> Get layerwise pruning ratios for the whole model (they may be updated later)."
        )

    else:
        raise NotImplementedError

    # Make pr an attribute, for later use
    for name, layer in prunable_layers.items():
        layer.pr = pr[name]
    return pr


# TODO: This fn may not be 100% correct. Deprecated. Will be removed. Keep it here
# for now.
def align_pruned_indices_for_cnst_layers(
    prunable_layers: OrderedDict,
    pruned_wg: OrderedDict,
    kept_wg: OrderedDict,
    constrained_groups: OrderedDict,
    wg: str = "filter",
):
    """Set pruned indices of constrained prunable_layers to the same."""
    pruned_wg, kept_wg = copy.deepcopy(kept_wg), copy.deepcopy(kept_wg)
    for _, v in constrained_groups.items():
        # Get kept/pruned for the 1st cnst layer.
        score = get_score_for_layer(
            module=prunable_layers[v[0]].module, wg=wg, criterion="mag"
        )["score"]
        pruned, kept = pick_pruned_weights_for_layer(
            score=score,
            pr=pr[v[0]],
            sort_mode="min",
        )
        # Set the rest kept/pruned same as the 1st cnst layer
        for name in v[1:]:
            pruned_wg[name], kept_wg[name] = pruned, kept
    return pruned_wg, kept_wg


def get_score_for_layer(module: nn.Module, wg: str = "filter", criterion: str = "mag"):
    r"""Get importance score for a layer. Define any scoring scheme here as you like.

    Args:
        module: module of the layer
        wg: weight group
        criterion: pruning criterion

    Returns:
        out: A dict that must have key 'score', whose value is a numpy array
    """
    if hasattr(module, "in_proj_weight"):  # MHA
        assert not hasattr(module, "weight")
        w = module.in_proj_weight.data
    elif hasattr(module, "weight"):
        w = module.weight.data
    else:
        raise NotImplementedError

    if wg == "channel":
        reduce_dim = [0, 2, 3] if len(w.shape) == 4 else 0
        l1 = w.abs().mean(dim=reduce_dim)

    elif wg in ["filter", "maskfilter"]:
        if len(w.shape) > 1:
            reduce_dim = list(range(1, len(w.shape)))
            l1 = w.abs().mean(dim=reduce_dim)
        else:
            l1 = w.abs()
        if criterion == "taylor-fo":
            g = module.accu_grad
            taylor_fo = ((w * g) ** 2).mean(
                dim=reduce_dim
            )  # Eq. 8 in Importance Estimation for Neural
            # Network Pruning (CVPR 2019)

    elif wg == "weight":
        l1 = w.abs().flatten()

    out = {"mag": tensor2array(l1)}
    out["l1-norm"] = out[
        "mag"
    ]  # 'mag' = 'l1-norm'. Keep the 'l1-norm' key for back-compatibility
    if criterion == "taylor-fo":
        out["taylor-fo"] = tensor2array(taylor_fo)
    out["score"] = out[criterion]
    return out


def get_pruned_index_inward(num_total: int, num_pruned: int):
    assert 0 <= num_pruned < num_total
    pruned = []
    all_indices = list(range(num_total))
    while len(pruned) < num_pruned:
        if len(pruned) % 2 == 0:
            pruned.append(all_indices.pop(0))
        else:
            pruned.append(all_indices.pop(-1))
    kept = list(set(all_indices) - set(pruned))
    return pruned, kept


def pick_pruned_weights_for_layer(
    score: list,
    pr: float = None,
    threshold: float = None,
    sort_mode: str = "min",
    weight_shape=None,
):
    r"""Get the indices of pruned weight groups in a layer.
    Args:
        score: a 1-d list, importance score of this layer
        pr: pruning ratio of this layer
        threshold: another way to decide which weights to prune, those with score < threshold will be discarded
        sort_mode: normally for a weight, the larger score, the better; this can be reverted by setting sort_mode
        weight_shape: shape of the original weights

    Returns:
        pruned (list): pruned indices
        kept (list): kept indices
    """
    score = np.array(score)
    num_total = len(score)
    max_pruned = num_total - 1  # to avoid pruning all
    if sort_mode in ["rand"] or re.match("rand_\d+", sort_mode):
        assert pr is not None
        seed = 42
        if "_" in sort_mode:  # e.g., rand_2023
            seed = int(sort_mode.split("_")[1])
        num_pruned = min(math.ceil(pr * num_total), max_pruned)
        np.random.seed(seed)
        order = np.random.permutation(num_total).tolist()
        pruned, kept = order[:num_pruned], order[num_pruned:]
        return pruned, kept

    elif sort_mode in ["min", "max", "ascending", "descending"]:
        num_pruned = (
            math.ceil(pr * num_total)
            if threshold is None
            else len(np.where(score < threshold)[0])
        )
        num_pruned = min(num_pruned, max_pruned)
        if sort_mode in ["min", "ascending"]:
            order = np.argsort(score).tolist()
        elif sort_mode in ["max", "descending"]:
            order = np.argsort(score)[::-1].tolist()
        pruned, kept = order[:num_pruned], order[num_pruned:]
        return pruned, kept

    elif re.match("min_\d+:\d+", sort_mode):
        # See https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/
        # Currently, only unstructured pruning supports such M:N sparsity pattern
        # E.g., 'mode' = 'min_2:4'
        M, N = [int(x) for x in sort_mode.split("_")[1].split(":")]
        score = score.reshape(
            -1, N
        )  # Risky, will throw an error if the total #elements is not a multiple of N
        indices = np.argsort(score, axis=-1)[:, :M]  # [M, N]
        out = []
        for row, col in enumerate(indices):
            out += (row * N + col).tolist()
        out = np.array(out)
        # TODO-@mst: Check this 2:4 impl.

    elif sort_mode in ["circu_sparsity", "cs"] or re.match("cs_\d+", sort_mode):
        pos_1 = 0
        if "_" in sort_mode:  # e.g., cs_2
            pos_1 = int(sort_mode.split("_")[1])
        shape = (
            weight_shape
            if len(weight_shape) == 2
            else (weight_shape[0], np.prod(list(weight_shape[1:])))
        )
        if pr > 0:
            mask = _get_sparse_circulant_matrix(shape, pr, pos_1)
            mask = mask.flatten()
            pruned = np.where(mask == 0)[0].tolist()
            kept = np.where(mask == 1)[0].tolist()
        else:
            pruned = []
            kept = list(range(len(score)))
        return pruned, kept

    elif sort_mode in ["evenly-spaced", "es"]:
        r"""Evenly pick kept weight groups"""
        raise NotImplementedError
        # TODO-@mst: add this impl.

    elif sort_mode in ["inward"]:
        r"""Pick pruned weights from outside to inside"""
        num_pruned = min(round(pr * num_total), max_pruned)
        pruned, kept = get_pruned_index_inward(num_total, num_pruned)
        return pruned, kept

    else:
        raise NotImplementedError


def pick_pruned_weights_for_model(
    prunable_layers: OrderedDict,
    raw_pr: OrderedDict,
    get_score_for_layer: Callable = get_score_for_layer,
    wg: str = "filter",
    criterion: str = "mag",
    compare_mode: str = "local",
    sort_mode: str = "min",
    constrained: OrderedDict = OrderedDict(),
    align_constrained: bool = False,
    print_fn=None,
):
    """Pick pruned weight groups for a model. This is the most central fn in a pruning algorithm.
    Args:
        prunable_layers: an OrderedDict, key is layer name, value is a Layer instance

    Returns:
        pruned (OrderedDict): key is layer name, value is the pruned indices for the layer
        kept (OrderedDict): key is layer name, value is the kept indices for the layer
    """
    assert compare_mode in ["global", "local"]
    pruned_wg, kept_wg = OrderedDict(), OrderedDict()
    all_scores = []

    # Get importance score for each layer
    for name, layer in prunable_layers.items():
        module = layer.module
        out = get_score_for_layer(module, wg=wg, criterion=criterion)
        score = out["score"]
        layer.score = score  # Make it an attribute for later use
        if raw_pr[name] == 0:
            pruned_wg[name], kept_wg[name] = [], list(range(len(score)))
            continue

        if print_fn is not None:
            print_fn(
                f"{prunable_layers[name].print_prefix} Pruning this layer, raw pr {layer.pr}"
            )
        if isinstance(
            module, nn.MultiheadAttention
        ):  # TODO: this MHA interface needs to be unified
            weight_shape = module.in_proj_weight.shape
        else:
            weight_shape = module.weight.shape

        if raw_pr[name] > 0:  # raw_pr > 0 indicates we want to prune this layer
            all_scores = np.append(all_scores, score)

        # Local pruning: weights are compared within the layer to decide which to discard
        if compare_mode in ["local"]:
            assert isinstance(raw_pr, dict)
            pruned_wg[name], kept_wg[name] = pick_pruned_weights_for_layer(
                score,
                raw_pr[name],
                sort_mode=sort_mode,
                weight_shape=weight_shape,
            )

    # Global pruning
    if compare_mode in ["global"]:
        num_total = len(all_scores)
        num_pruned = min(
            math.ceil(raw_pr["model"] * num_total), num_total - 1
        )  # do not prune all
        if sort_mode == "min":
            threshold = sorted(all_scores)[num_pruned]  # in ascending order
        elif sort_mode == "max":
            threshold = sorted(all_scores)[::-1][num_pruned]  # in descending order
        if print_fn is not None:
            print_fn(
                f"Global pruning: #all_scores {len(all_scores)}, threshold {threshold:.20f}"
            )

        for name, layer in prunable_layers.items():
            pruned_wg[name], kept_wg[name] = [], list(range(len(layer.score)))
            if raw_pr[name] > 0:
                if sort_mode in ["rand"]:
                    pass
                elif sort_mode in ["min", "max"]:
                    pruned_wg[name], kept_wg[name] = pick_pruned_weights_for_layer(
                        layer.score,
                        pr=None,
                        threshold=threshold,
                        sort_mode=sort_mode,
                    )

    # Adjust pr/pruned/kept
    pr = copy.deepcopy(raw_pr)
    for name, layer in prunable_layers.items():
        num_pruned = len(pruned_wg[name])
        pr[name] = num_pruned / len(layer.score)

    if wg == "filter" and align_constrained:
        pruned_wg, kept_wg = align_pruned_indices_for_cnst_layers(
            prunable_layers,
            pr,
            pruned_wg,
            kept_wg,
            constrained,
            wg=wg,
            criterion=criterion,
            sort_mode=sort_mode,
        )

    # Print pruned indices
    if wg != "weight" and print_fn is not None:
        print_fn(f"-" * 30 + " Print pruned indices (Start) " + "-" * 30)
        print_fn(
            f"(Note the pruned indices of the learnable_layers of the same "
            f"constrained (cnst) group should be the same)"
        )
        max_name_length = max([len(n) for n in prunable_layers])
        max_cnst_length = max(
            [len(str(prunable_layers[n].cnst_grp)) for n in prunable_layers]
        )
        for name in prunable_layers:
            if pr[name] > 0:
                cnst_group_id = str(prunable_layers[name].cnst_grp)
                print_fn(
                    f"{name.ljust(max_name_length)} "
                    f"cnst_group {cnst_group_id.rjust(max_cnst_length)} "
                    f"first_10_pruned_wg {pruned_wg[name][:10]}"
                )
        print_fn("-" * 30 + " Print pruned indices (End) " + "-" * 30)

    return pr, pruned_wg, kept_wg


def register_hooks_for_pruning(
    model: nn.Module,
    pr: OrderedDict,
    constrained_groups: OrderedDict = OrderedDict(),
    print_fn: Callable = print,
    align_constrained: bool = False,
):
    """Main fn for pruning with hooks. Very important!
    Args:
        model: Unpruned model.
        pr: Layerwise pruning ratio.
        constrained_groups: Groups of constrained layers.
        print_fn: Print function.

    Returns:
        pr (OrderedDict): Updated layerwise pruning ratio.
        kept_filters (OrderedDict): Key is layer name, value is the indices of kept filters.
        kept_channels (OrderedDict): Key is layer name, value is the indices of kept channels.
        handles (List): Handles of the created hooks.
    """
    kept_filters = OrderedDict()
    kept_channels = OrderedDict()
    handles = []

    def _hook(m, input, output):
        """This hook fn is to decide the kept_channels for output, based on the module
        type. If the module is conv/linear etc., do pruning; otherwise, just pass on
        the kept_channels.

        Args:
            m: nn.Module.
            input: Tuple of tensors.
            output: Tensor.
        """
        input_shape = [list(i.shape) for i in input]
        if isinstance(m, nn.LSTM):
            output_shape = list(output[0].shape)
        else:
            output_shape = list(output.shape)

        # Check the input channels.
        num_chls = input[0].shape[1]
        if len(input[0].shape) == 4:  # [B, C, H, W], e.g., Conv
            abs_input = input[0].abs().mean(dim=[0, 2, 3])
        elif len(input[0].shape) == 3:  # [B, L, D], e.g., LSTM
            abs_input = input[0].abs().mean(dim=[0, 1])
        elif len(input[0].shape) == 2:  # [B, C], e.g., Linear
            abs_input = input[0].abs().mean(dim=[0])
        else:
            raise NotImplementedError(
                f"The input shape should be 2D/3D/4D. Now it is {list(input[0].shape)}."
            )
        kept_chls = torch.nonzero(abs_input, as_tuple=True)[0].tolist()
        pruned_chls = list(set(range(num_chls)) - set(kept_chls))

        # Learnable layers. [TODO-@huanwangx: Add LSTM case.]
        if len(m._parameters) and not isinstance(m, nn.LSTM):
            if hasattr(m, "weight"):  # For CNN: naive modules like Conv2d/Linear/BN
                w = m.weight
            elif (
                hasattr(m, "in_proj_weight") and m.in_proj_weight is not None
            ):  # For MHA: qkv same dim
                w = m.in_proj_weight
            elif (
                hasattr(m, "q_proj_weight") and m.q_proj_weight is not None
            ):  # For MHA: qkv not same dim
                assert None not in (m.k_proj_weight, m.v_proj_weight)
                raise NotImplementedError
            else:
                raise NotImplementedError
            shape = list(w.size())
            # print_fn(f"{m.module_name} weight shape: {shape}", unprefix=True)

            # Make the output non-zero. This cannot be applied to in-place ops.
            output.data.copy_(torch.ones_like(output))

            # For norm layers, pass the kept channels.
            if isinstance(
                m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)
            ):
                kept_filters[m.module_name] = kept_chls
                kept_channels[m.module_name] = []  # No channel axis in norm layers.
                pruned_chls = list(set(range(shape[0])) - set(kept_chls))
                output[:, pruned_chls, ...] = 0

            # For conv/linear layers, do the pruning.
            elif isinstance(m, (nn.Conv2d, nn.Linear)):
                kept_channels[m.module_name] = kept_chls

                # Get kept_filters.
                if hasattr(m, "info") and "prune_scheme" in m.info:
                    # [TODO-@huanwangx] This is ad-hoc design.
                    if m.info["prune_scheme"] in ["chunk2"]:
                        N = shape[0]
                        assert N % 2 == 0
                        w_abs = w[: N // 2].abs().view(N // 2, -1).cpu().data
                        score = w_abs.mean(dim=-1)
                        x, indices = torch.sort(score)
                        num_pruned = math.ceil(pr[m.module_name] * len(score))
                        kept = indices[num_pruned:].data.cpu().tolist()
                        kept = kept + [x + N // 2 for x in kept]
                else:
                    w_abs = (w.abs().view(shape[0], -1).cpu().data)# Structured pruning by magnitude.
                    score = w_abs.mean(dim=-1)
                    _, indices = torch.sort(score)
                    num_pruned = math.ceil(pr[m.module_name] * len(score))
                    kept = indices[num_pruned:].data.cpu().tolist()

                    # If current layer is a constrained layer, may adjust the kept:
                    # if any layer in the same constrained group has been assigned kept,
                    # just use its kept.
                    if align_constrained:
                        for group_id, layers_same_group in constrained_groups.items():
                            if m.module_name in layers_same_group:
                                for l in layers_same_group:
                                    if l in kept_filters:
                                        kept = kept_filters[l]
                                        print_fn(
                                            f"{m.module_name} constrained, kept adjusted!"
                                        )
                                        break

                    pruned = list(set(list(range(len(score)))) - set(kept))

                kept.sort()
                pruned.sort()
                kept_filters[m.module_name] = kept
                pr[m.module_name] = num_pruned / len(score)

                # Apply pruning to the output tensor. Very important!
                output[:, pruned, ...] = 0

            # For MHA layers, do the pruning.
            elif isinstance(m, (nn.MultiheadAttention, nn.Embedding)):
                pass

            else:
                print_fn(f"NotImplemented: {m.module_name} {m._get_name()} {shape}")
                raise NotImplementedError

    def _register(m, handles):
        """Recursively register hook for each learnable layer. A layer is defined as
        the node in a computation graph, which has no children, and has parameters.
        """
        children = list(m.children())
        if len(children) == 0 or isinstance(m, nn.MultiheadAttention):
            # MHA has children
            # TODO-@mst: this MultiheadAttention manual check is ad-hoc, improve it?
            handles += [m.register_forward_hook(_hook)]
        else:
            [_register(c, handles) for c in children]

    _register(model, handles)

    return pr, kept_filters, kept_channels, handles


def replace_module(model, name, new_m):
    r"""Replace the module <name> in <model> with <new_m>
    E.g., 'module.layer1.0.conv1'
    ==> model.__getattr__('module').__getattr__("layer1").__getitem__(0).__setattr__('conv1', new_m)
    """
    obj = model
    segs = name.split(".")
    for ix, s in enumerate(segs):
        if ix == len(segs) - 1:  # the last one
            if s.isdigit():
                obj.__setitem__(int(s), new_m)
            else:
                obj.__setattr__(s, new_m)
            return
        if s.isdigit():
            obj = obj.__getitem__(int(s))
        else:
            obj = obj.__getattr__(s)


def get_kept_filter_channel(
    layers, layer_name, kept_wg, norm_types, wg="filter", pick_scheme=None
):
    r"""Considering layer dependency, get the kept filters and channels for the layer of <layer_name>."""
    current_layer = layers[layer_name]
    shape = current_layer.shape
    assert wg in ["filter", "channel"]

    if wg in ["channel"]:
        raise NotImplementedError

    elif wg in ["filter"]:
        last_layer_name = (
            current_layer.last
        )  # TODO: how to handle multi last learnable_layers
        last_layer_pr = layers[last_layer_name].pr if last_layer_name != "None" else -1
        if isinstance(current_layer.module, norm_types):
            total_filters = shape[0]
            kept_filter, kept_chl = list(range(total_filters)), []
            if last_layer_pr > 0:
                kept_filter = kept_wg[last_layer_name]
                if pick_scheme in ["evenly-spaced", "es"]:
                    num_kept = total_filters - math.ceil(last_layer_pr * total_filters)
                    np.random.seed(42)
                    order = np.random.permutation(total_filters)
                    kept_filter = order[:num_kept]
        else:
            total_channels = shape[1]
            kept_filter = kept_wg[layer_name]
            kept_chl = list(range(total_channels))
            if last_layer_pr > 0:
                kept_chl = kept_wg[last_layer_name]
                if pick_scheme in ["evenly-spaced", "es"]:
                    num_kept = total_channels - math.ceil(
                        last_layer_pr * total_channels
                    )
                    np.random.seed(42)
                    order = np.random.permutation(total_channels)
                    kept_chl = order[:num_kept]

    # Sort to make the indices be in ascending order
    kept_filter, kept_chl = list(kept_filter), list(kept_chl)  # Make sure they are list
    kept_filter.sort()
    kept_chl.sort()
    return kept_filter, kept_chl


def get_masks_for_pruned_layers(
    pruned_layers: OrderedDict, pruned_wg: OrderedDict, wg: str = "weight"
):
    """Get masks for all pruned layers in the model.
    Args:
        pruned_layers: a dict of pruned layers.
        pruned_wg: a dict, key is a layer name, and value is a list of pruned indices
        wg: weight group

    Returns:
        masks: a dict, key is layer name, value is the mask of the layer
    """
    masks = OrderedDict()
    for name, layer in pruned_layers.items():
        pruned = pruned_wg[name]
        if wg == "weight":  # Unstructured pruning
            mask = torch.ones(layer.shape).flatten()
            mask[pruned] = 0
            mask = mask.view(layer.shape)
        elif wg in ["filter", "maskfilter"]:  # Structured pruning
            mask = torch.ones(layer.shape)
            mask[pruned, ...] = 0
        else:
            raise NotImplementedError
        masks[name] = mask
    return masks


def _get_sparse_circulant_matrix(shape, sparsity_ratio, pos_1=0):
    if len(shape) == 2:
        H, W = shape
    else:
        raise NotImplementedError
    num_ = W

    # Get base sparse list for 'sparsity_ratio'
    sparsity = np.round(num_ * sparsity_ratio) / num_
    if sparsity_ratio != sparsity:
        print(f"Designated sparsity ratio rounded from {sparsity_ratio} to {sparsity}")
    sparsity_ratio = sparsity

    # Get the base sparse list for sparsity_ratio
    # E.g., sparsity_ratio = 0.75, num_ = 10, then first the sparsity_ratio is rounded to 0.8.
    # N0 = 8, N1 = 2, minibase = [1, 0, 0, 0, 0], base = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    # E.g., sparsity_ratio = 0.625, num_ = 8, then the sparsity_ratio will still be 0.625.
    # N0 = 5, N1 = 3, minibase = [1, 0], base = [1, 0, 1, 0, 1, 0, 0, 0]
    # In each minibase, there would be only one 1 or 0
    N0, N = int(np.round(num_ * sparsity_ratio)), num_  # N0: number of zeros
    N1 = N - N0
    if N0 > N1:  # Network pruning typically goes this way
        minibase = [0] + [0] * (N0 // N1)
        assert pos_1 < len(minibase)
        minibase[pos_1] = 1
        base = minibase * N1
        left = 0
    else:
        minibase = [0] + [1] * (N1 // N0)
        base = minibase * N0
        left = 1

    # Append the left 0 or 1's to the end
    num_left = N - len(base)
    base += [left] * num_left
    print(
        f"Sparsity ratio: {sparsity_ratio}, its minibase sparse list length: {len(minibase)}, pos_1: {pos_1}"
    )

    # Get circulant matrix
    circ_matrix = circulant(base)

    # Crop or expand matrix
    if H > W:
        circ_matrix_ = circ_matrix
        for _ in range(H // W - 1):
            circ_matrix_ = np.concatenate([circ_matrix_, circ_matrix], axis=0)
        circ_matrix = np.concatenate([circ_matrix_, circ_matrix[: H % W]], axis=0)
    else:
        circ_matrix = circ_matrix[:H, :]

    assert circ_matrix.shape == (
        H,
        W,
    ), f"circ_matrix shape: {circ_matrix.shape}; should be {(H, W)}"
    return circ_matrix


def set_up_prune_args(parser):
    """Set up the args for pruning.
    Args:
        parser: the parser in argparse

    Returns:
        args: parsed args
        unknown: unknown args

    """
    from importlib import import_module
    from smilelogging.utils import get_arg

    argv = np.array(sys.argv)
    if f"--prune_method" in argv[1:]:
        ix = np.where(argv == f"--prune_method")[0][-1]
        method = argv[ix + 1]
        if method and not method.startswith("-"):  # TODO: Add some method name check?
            # Add shared args about pruning
            from pruner.meta_args import add_args

            parser = add_args(parser)

            # Get pruner name
            if "--pruner" in argv[1:]:
                ix = np.where(argv == f"--pruner")[0][-1]
                pruner_name = argv[ix + 1]
            else:
                pruner_name = method.lower()

            # Add args that are specific to the pruning method
            prune_module = import_module(f"pruner.{pruner_name}_args")
            parser = prune_module.add_args(parser)

    # Parse
    args, unknown = parser.parse_known_args()

    # Check args for pruning method
    if get_arg(args, "prune_method"):
        from pruner.meta_args import check_args

        args = check_args(args)
        args = prune_module.check_args(args)

    return args, unknown
