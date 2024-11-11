import sys
from collections import OrderedDict
from typing import Union
import numpy as np
from typing import Tuple, List
import torch
import torch.nn as nn

MODULE_ID_BASE = 1e10


# [TODO-@huanwangx] This scheme is not general enough.
def encode_module_ids(module_ids: List[int], tensor: torch.Tensor):
    """A tensor can come from different modules. Encode the source module ids to the
    tensor.

    Args:
        module_ids: Module ids, >= 0.
        tensor: Pytorch Tensor.
    """
    values = torch.zeros_like(tensor).flatten()
    if max(module_ids) + 1 < len(values):
        for mid in module_ids:
            values[mid + 1] = mid + 1  # Hash! mid starts from -1.
    # else:  # TODO: This range issue. For most layers this is okay. But not entirely correct!
    #     raise NotImplementedError
    values = values.view(tensor.shape)
    tensor.data.copy_(values)  # Note this has to be in-place!


def decode_module_ids(tensor: torch.Tensor):
    """Decode the source module ids from the tensor.

    Args:
        tensor: Pytorch Tensor.

    Returns:
        A list of source module ids.
    """
    tensor = tensor.data.cpu().flatten().numpy()
    module_ids = np.where(tensor > 0)[0]
    if len(module_ids) == 0:
        return [-1]
    out = []
    for i in module_ids:
        if tensor[i] == i:  # Ideally, position is same as value.
            out.append(int(i) - 1)
        else:  # Otherwise, trust the value -- but this is not always correct!
            out.append(int(tensor[i]) - 1)
    return out


class Layer:
    def __init__(
        self,
        name: str,
        shape: Tuple[list, tuple],
        index: int,
        module: nn.Module,
        type: str = None,
        last: str = None,
        next: str = None,
        prunable: bool = True,
        cnst_grp: int = -1,
    ):
        self.name = name
        self.module = module
        self.shape = shape
        self.index = index
        self.type = type
        self.last = last
        self.next = next
        self.prunable = prunable
        self.cnst_grp = cnst_grp


def register_hooks_for_model(model, learnable_types, prunable_types, print_fn):
    """Register hooks for a model."""
    learnable_layers = (
        OrderedDict()
    )  # All learnable layers - those with parameters, including BN etc.
    prunable_layers = OrderedDict()  # Usually, Conv & Linear, no BN.
    direct_father = (
        OrderedDict()
    )  # Direct dependency layers, including non-learnable layers like ReLU.
    real_father = OrderedDict()
    constrained_groups = (
        OrderedDict()
    )  # Constrained groups due to Add/Mul ops (e.g., in residual and attention blocks).

    max_len_name = [0]  # TODO-@mst: refactor this name
    max_len_shape = [0]  # Use list because this variable will be modified in hook fn
    max_len_ix = [0]
    handles = []  # For removing hooks later.
    cnst_grp = [-1]
    all_modules = []

    def _get_real_father_layer(dire_last: str) -> str:
        r"""Get real last layer based on direct last layer."""
        dfather = dire_last.split(",") if "," in dire_last else [dire_last]
        rfather = []
        for k in dfather:
            while k != "None" and (
                k not in learnable_layers or not learnable_layers[k].prunable
            ):
                k = direct_father[k]  # TODO-@mst: There is a problem here for Cat op.
                if "," in k:
                    k = k.split(",")[0]
            rfather.append(k)
        return ",".join(rfather)

    def _hook(m, input, output):
        """Hook fn to build layer dependency."""
        module_id = len(all_modules)  # Module id starts from 0!
        all_modules.append(m.module_name)
        op = m._get_name()
        m_name = f"{m.module_name}/{op}"

        # Logging when it is the 1st layer.
        if module_id == 0:
            print_fn("=" * 25 + " Building Layer Dependency " + "=" * 25, unprefix=True)

        # Source the last layer from input.
        i = input[
            0
        ]  # Only handle one-tensor input now. [TODO-@huanwangx-20240106: This holds for most cases. Even sometimes, we use Cat.]
        src_module_ids = decode_module_ids(
            i
        )  # For each i, should decode its module ids.
        print_fn(f"src_module_ids: {src_module_ids}", color="red", unprefix=1)

        if len(src_module_ids) == 1:
            dfather = (
                "None" if src_module_ids[0] == -1 else all_modules[src_module_ids[0]]
            )
        else:
            dfather = ",".join([all_modules[i] for i in src_module_ids])
        direct_father[m.module_name] = dfather

        # Get the real father layer.
        rfather = _get_real_father_layer(dfather)
        real_father[m.module_name] = rfather
        logstr = (
            f"[Dep] {module_id:>3d} {m_name:50s}"
            f"real_father/direct_father: {rfather + '/' + dfather:110s}"
        )

        # Update constrained layers. (TODO: This part is wrong for resnet56 + cifar10)
        if "," in rfather:
            all_cnst_layers = []
            for k, v in constrained_groups.items():
                all_cnst_layers += v

            # See if the cnst layer are already registered in constrained_groups.
            found = None
            for layer in rfather.split(","):
                if layer in all_cnst_layers:
                    found = layer
                    break

            if (
                found is None
            ):  # These layers have never been registered in constrained_groups.
                cnst_grp[0] = len(constrained_groups)  # cnst_id starts from 0
                constrained_groups[cnst_grp[0]] = rfather.split(",")
            else:
                for k, v in constrained_groups.items():
                    if found in v:
                        break
                constrained_groups[k] += rfather.split(",")

        # Pass the module ids to output.
        if isinstance(output, tuple):
            for o in output:
                if isinstance(o, torch.Tensor):
                    encode_module_ids([module_id], o)
        elif isinstance(output, torch.Tensor):
            encode_module_ids([module_id], output)
        else:
            raise NotImplementedError(
                "The output is supposed to be tensor or tuple of tensors."
            )

        # Register this layer if it is learnable.
        if (
            isinstance(m, learnable_types) and m.module_name not in learnable_layers
        ):  # to avoid replicated registration
            # Get layer parameter shape
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
            shape = w.size()

            layer_ix = len(learnable_layers)
            shape = list(shape)
            max_len_name[0] = max(max_len_name[0], len(m.module_name))
            max_len_shape[0] = max(max_len_shape[0], len(str(shape)))
            max_len_ix[0] = max(max_len_ix[0], len(str(layer_ix)))

            logstr += f" {shape}"

            # Register the layer.
            # [TODO-@huanwangx] last needs to be double-checked.
            learnable_layers[m.module_name] = Layer(
                name=m.module_name,
                shape=shape,
                index=layer_ix,
                module=m,
                type=m.__class__.__name__,
                last=rfather,
                prunable=isinstance(m, prunable_types),
            )
            if isinstance(m, prunable_types):
                prunable_layers[m.module_name] = learnable_layers[m.module_name]

        print_fn(logstr, unprefix=True)

    def _register(m, handles):
        """Recursively register hook for each layer. A layer is defined as the node in
        a computation graph that has no children.
        """
        children = list(m.children())
        if len(children) == 0 or isinstance(m, nn.MultiheadAttention):
            # MHA has children
            # [TODO-@huanwangx] MultiheadAttention manual check is ad-hoc, to improve.
            handles += [m.register_forward_hook(_hook)]
        else:
            [_register(c, handles) for c in children]

    _register(model, handles)

    return (
        learnable_layers,
        prunable_layers,
        constrained_groups,
        max_len_ix,
        max_len_name,
        max_len_shape,
        handles,
    )


class LayerBuilder:
    """Build the learnable_types in a model. This is the central function to figure out
    the layer dependency in a model.
    """

    def __init__(
        self,
        model,
        learnable_types=(
            nn.Conv1d,
            nn.Conv2d,
            nn.Linear,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.GroupNorm,
            nn.LayerNorm,
        ),
        prunable_types=(nn.Conv1d, nn.Conv2d, nn.Linear),
        dummy_input=None,
        print_fn=print,
    ):
        self.model = model
        self.learnable_types = learnable_types
        self.prunable_types = prunable_types
        self.dummy_input = dummy_input
        self.print = print_fn

        # Register hooks.
        (
            self.learnable_layers,
            self.prunable_layers,
            self.constrained_groups,
            self._max_len_ix,
            self._max_len_name,
            self._max_len_shape,
            self.handles,
        ) = register_hooks_for_model(
            self.model, self.learnable_types, self.prunable_types, print_fn=print_fn
        )
        self.print(f"Register hooks for model to build layer dependency -- done!")
        self.num_layers = len(self.learnable_layers)
        self.print_prefix = OrderedDict()
        self.layer_build_done = False

        # Model forward to materialize the hooks.
        if self.dummy_input is not None:
            if isinstance(self.dummy_input, torch.Tensor):
                dummy_input = self.dummy_input.clone()
                encode_module_ids([-1], dummy_input)
            else:
                raise NotImplementedError
            self.model_forward(dummy_input=dummy_input)
            self.finish()

        # Set up the next layer.
        for n, l in self.learnable_layers.items():
            if l.last != "None":
                if "," in l.last:
                    for i in l.last.split(","):
                        self.learnable_layers[i].next = n
                else:
                    self.learnable_layers[l.last].next = n

    def model_forward(self, model=None, dummy_input=None):
        """Model forward to make hooks physically work."""
        self.print(f"Model forward in layer build ...")
        if model is None:
            model = self.model
        if dummy_input is None:
            dummy_input = self.dummy_input

        is_train = model.training
        if is_train:
            model.eval()

        with torch.no_grad():
            if isinstance(dummy_input, dict):
                model(**dummy_input)
            elif isinstance(dummy_input, (tuple, list)):
                model(*dummy_input)
            elif isinstance(dummy_input, torch.Tensor):
                model(dummy_input)
            else:
                raise NotImplementedError

        if is_train:
            model.train()
        self.print(f"Model forward in layer build -- done!")

    def finish(self):
        self._max_len_ix = self._max_len_ix[0]
        self._max_len_name = self._max_len_name[0]
        self._max_len_shape = self._max_len_shape[0]
        self._update_cnst_id()
        self._get_print_prefix()
        self.print("Layer profiles are:")
        self._print_layer_stats()
        self._rm_hooks()
        self.layer_build_done = True
        self.print(
            f"Hooks for building layer dependency enforced. Now layer build is done!"
        )

    def _rm_hooks(self):
        [x.remove() for x in self.handles]

    def _update_cnst_id(self):
        for layer_name, layer in self.learnable_layers.items():
            for k, v in self.constrained_groups.items():
                if layer_name in v:
                    layer.cnst_grp = k

    def _get_print_prefix(self):
        for name, layer in self.learnable_layers.items():
            format_str = f"[%-{self._max_len_ix}d] %-{self._max_len_name}s %-{self._max_len_shape}s %-15s"  # align left
            prefix = format_str % (layer.index, name, layer.shape, layer.type)
            self.print_prefix[name] = prefix
            layer.print_prefix = prefix

    def _print_layer_stats(self):
        self.print(
            "------------------------- Layer Profile -------------------------",
            unprefix=True,
        )
        for name, layer in self.learnable_layers.items():
            format_str = f"%s  last: %-50s  cnst_grp: %3s"
            self.print(
                format_str % (self.print_prefix[name], layer.last, layer.cnst_grp),
                unprefix=True,
            )
        self.print(
            "-----------------------------------------------------------------",
            unprefix=True,
        )
