import torch
from peft.tuners.lora import LoraLayer
import torch.nn as nn
import torch.nn.functional as F
import math
from peft.utils.other import transpose
import random
import numpy as np
import peft.tuners.lora as Lora
from peft import PeftModel
# from peft.tuners.lora import Linear, Embedding, LoraModel
import torch
import pynvml
import os
import warnings
from torch.autograd import Function
from torch.utils.data import IterableDataset
import netifaces
from trl import SFTTrainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from transformers.trainer import TRAINING_ARGS_NAME
import safetensors.torch
from functools import partial
from torch.optim.lr_scheduler import LambdaLR

def get_local_ip():
    for interface in netifaces.interfaces():
        addresses = netifaces.ifaddresses(interface)
        # Check for IPv4 addresses
        if netifaces.AF_INET in addresses:
            for link in addresses[netifaces.AF_INET]:
                ip_address = link['addr']
                # Assuming your local network IPs start with '192.168.'
                if ip_address.startswith('192.168.'):
                    return ip_address
    return "Local IP address not found"

def get_lora_parameter_names(model):
    result = []
    for param_name, param in model.named_parameters():
        if "lora" in param_name:
            result.append(param_name)
    return result

def _get_iterative_polynomial_decay_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    update_interval: int,
    num_warmup_per_interval: int,
    lr_end: float,
    power: float,
    lr_init: int,
):
    if current_step < num_warmup_steps:
        global_lr = float(current_step) / float(max(1, num_warmup_steps))
    elif current_step > num_training_steps:
        global_lr = lr_end / lr_init  # as LambdaLR multiplies by lr_init
    else:
        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
        decay = lr_range * pct_remaining**power + lr_end
        global_lr = decay / lr_init  # as LambdaLR multiplies by lr_init

    local_step = current_step % update_interval
    if local_step < num_warmup_per_interval:
        local_lr = local_step / num_warmup_per_interval
    else:
        local_lr = 1

    return global_lr * local_lr

def get_iterative_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, update_interval, num_warmup_per_interval, lr_end=1e-7, power=1.0, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        lr_end (`float`, *optional*, defaults to 1e-7):
            The end LR.
        power (`float`, *optional*, defaults to 1.0):
            Power factor.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    lr_init = optimizer.defaults["lr"]
    # if not (lr_init > lr_end):
    #     raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    lr_lambda = partial(
        _get_iterative_polynomial_decay_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        update_interval=update_interval,
        num_warmup_per_interval=num_warmup_per_interval,
        lr_end=lr_end,
        power=power,
        lr_init=lr_init,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def _get_warm_iterative_cosine_lr_lambda(
    current_step: int, *, steps_per_cycle: int
):
    current_cycle_step = current_step % steps_per_cycle
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (float(current_cycle_step / steps_per_cycle) + 1.0))))

def get_warm_iterative_cosine(
    optimizer, steps_per_cycle: int, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_warm_iterative_cosine_lr_lambda,
        steps_per_cycle=steps_per_cycle
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def replace_with_svd(model, config, lora_head, mode):
    for n, m in model.named_modules():
        if isinstance(m, Lora.Linear):
            parent = model.get_submodule(".".join(n.split(".")[:-1]))
            child_name = n.split(".")[-1]
            active_adapter = m.active_adapter[0]
            new_module = SVDLinear(m.base_layer, active_adapter, config.r, lora_head, config.lora_alpha, config.lora_dropout, init_lora_weights=False, mode=mode)
            model._replace_module(parent, child_name, new_module, m)
            new_module.to(m.weight.device)
        elif isinstance(m, Lora.Embedding):
            raise Exception("no svd embedding")

def replace_embedding(model):
    for n, m in model.named_modules():
        if isinstance(m, Lora.Embedding):
            parent = model.get_submodule(".".join(n.split(".")[:-1]))
            child_name = n.split(".")[-1]
            # bias = hasattr(m, "bias") and m.bias is not None
            active_adapter = m.active_adapter[0]
            new_module = TransposedEmbedding(m.base_layer, active_adapter, m.r[active_adapter], m.lora_alpha[active_adapter], init_lora_weights=False)
            model._replace_module(parent, child_name, new_module, m)


class SVDLinear(nn.Module, LoraLayer):
    adapter_layer_names = ()
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        head: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        mode: str = None,
        **kwargs,
    ):
        assert(init_lora_weights is False)
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self.mode = mode

        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.lora_E = nn.ParameterDict({})
        self.lora_active_A = nn.ParameterDict({})
        self.lora_active_B = nn.ParameterDict({})

        self.head = head
        assert (r % head == 0)
        self.block_size = r // head

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

        active_adapter = self.active_adapter[0]
        # self.register_buffer("lora_mask", torch.zeros(self.r[active_adapter]))
        # self.update_mask(0)
        self.cur_head = 0
        epsilon = 1e-9

        self.lora_active_A[active_adapter].register_hook(lambda grad: grad / (torch.abs(self.lora_E[active_adapter][:self.block_size, :]) + epsilon))
        self.lora_active_B[active_adapter].register_hook(lambda grad: grad / (torch.abs(self.lora_E[active_adapter][:self.block_size, :]) + epsilon).T)


    def update_mask(self, cur_head):
        adapter = self.active_adapter[0]
        with torch.no_grad():
            self.cur_head = cur_head
            # self.lora_mask.zero_()
            if self.mode == "svd_init" or self.mode == "svd_shuffle":
                new_active_A = self.lora_A[adapter][:self.block_size, :].data
                new_A = torch.cat([self.lora_A[adapter][self.block_size:, :].data, self.lora_active_A[adapter].data], dim=0)
                self.lora_active_A[adapter].data = new_active_A
                self.lora_A[adapter].data = new_A

                new_active_B = self.lora_B[adapter][:, :self.block_size].data
                new_B = torch.cat([self.lora_B[adapter][:, self.block_size:].data, self.lora_active_B[adapter].data],
                                  dim=1)
                self.lora_active_B[adapter].data = new_active_B
                self.lora_B[adapter].data = new_B

                self.lora_E[adapter].data = torch.cat([self.lora_E[adapter].data[self.block_size:, :], self.lora_E[adapter].data[:self.block_size, :]], dim=0)
            if self.mode == "svd_adaptive":
                p = torch.abs(self.lora_E[adapter].squeeze(1))
                p = p + torch.mean(p)
                p = p / p.sum()
                indices = torch.multinomial(p, num_samples=self.block_size, replacement=False)
                full_A = torch.cat([self.lora_active_A[adapter].data, self.lora_A[adapter].data], dim=0)
                full_B = torch.cat([self.lora_active_B[adapter].data, self.lora_B[adapter].data], dim=1)
                mask = torch.zeros_like(p, dtype=torch.bool, device=self.lora_E[adapter].device)
                mask[indices] = True

                self.lora_active_A[adapter].data = full_A[mask]
                self.lora_A[adapter].data = full_A[~mask]
                self.lora_active_B[adapter].data = full_B[:, mask]
                self.lora_B[adapter].data = full_B[:, ~mask]
                self.lora_E[adapter].data = torch.cat([self.lora_E[adapter].data[mask], self.lora_E[adapter].data[~mask]], dim=0)


            # elif self.mode == "svd_adaptive":
            #     p = torch.abs(self.lora_E[self.active_adapter[0]].squeeze(1))
            #     p = p / p.sum()
            #     indices = torch.multinomial(p, num_samples=self.block_size, replacement=False)
            #     self.lora_mask[indices] = 1
            # elif self.mode == "svd_random":
            #     p = torch.ones_like(self.lora_E[self.active_adapter[0]].squeeze(1))
            #     p = p / p.sum()
            #     indices = torch.multinomial(p, num_samples=self.block_size, replacement=False)
            #     self.lora_mask[indices] = 1
            else:
                raise Exception("invalid svd init in update_mask")


    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A[adapter_name] = nn.Parameter(torch.randn(r - self.block_size, self.in_features), requires_grad=False)
            self.lora_E[adapter_name] = nn.Parameter(torch.randn(r, 1))
            self.lora_B[adapter_name] = nn.Parameter(torch.randn(self.out_features, r - self.block_size), requires_grad=False)
            self.lora_active_A[adapter_name] = nn.Parameter(torch.randn(self.block_size, self.in_features))
            self.lora_active_B[adapter_name] = nn.Parameter(torch.randn(self.out_features, self.block_size))
            self.scaling[adapter_name] = lora_alpha / r


        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)


    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter):
        return (
            transpose(
                torch.cat([self.lora_active_B[adapter], self.lora_B[adapter]], dim=1) @ (torch.cat([self.lora_active_A[adapter], self.lora_A[adapter]], dim=0) * self.lora_E[adapter]),
                self.fan_in_fan_out,
            )
            * self.scaling[adapter]
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                # lora_A = self.lora_A[active_adapter]
                # lora_B = self.lora_B[active_adapter]
                # dropout = self.lora_dropout[active_adapter]
                # scaling = self.scaling[active_adapter]
                x = x.to(self.lora_A[active_adapter].dtype)
                # result += lora_B(lora_A(dropout(x))) * scaling
                # if x.dim() == 3:
                #     dim_0, dim_1, dim_2 = x.shape
                #     x = x.view(-1, dim_2)
                #     svd_result = SVDFunction.apply(self.lora_dropout[active_adapter](x), self.lora_A[active_adapter],self.lora_E[active_adapter], self.lora_B[active_adapter]) * self.scaling[active_adapter]
                #     svd_result = svd_result.view(dim_0, dim_1, -1)
                #     result += svd_result
                # else:
                #     result += SVDFunction.apply(self.lora_dropout[active_adapter](x), self.lora_A[active_adapter],self.lora_E[active_adapter], self.lora_B[active_adapter]) * self.scaling[active_adapter]
                result += self.lora_dropout[active_adapter](x) @ (torch.cat([self.lora_active_A[active_adapter], self.lora_A[active_adapter]], dim=0) * self.lora_E[active_adapter]).T @ torch.cat([self.lora_active_B[active_adapter], self.lora_B[active_adapter]], dim=1).T

        result = result.to(previous_dtype)
        return result

class TransposedEmbedding(nn.Module, LoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer_embedding(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        if r > 0:
            weight_A = torch.randn((self.in_features, r))
            weight_B = torch.randn((r, self.out_features))
            self.lora_embedding_A[adapter_name] = nn.Parameter(weight_A)
            self.lora_embedding_B[adapter_name] = nn.Parameter(weight_B)
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        base_layer = self.get_base_layer()
        weight = getattr(base_layer, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(base_layer.weight.device, dtype=weight.dtype)
        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self.lora_embedding_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.copy()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_embedding_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B.T @ weight_A.T, True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_embedding_A[adapter] = weight_A.to(dtype)
            self.lora_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor

    def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter]
                embedding_B = self.lora_embedding_B[active_adapter]
                scaling = self.scaling[active_adapter]
                after_A = self._embed(x, embedding_A)
                result += (after_A @ embedding_B) * scaling

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# def get_delta_weight(m):
#     return transpose(
#         m.lora_B[m.active_adapter].weight @ m.lora_A[m.active_adapter].weight,
#         False,
#     ) * m.scaling[m.active_adapter]

def print_lora(m, mode):
    active_adapter = m.active_adapter[0]
    if isinstance(m, Lora.Linear) or isinstance(m, Lora.Conv2d):
        print(f"{mode} lora_A: {torch.mean(torch.abs(m.lora_A[active_adapter].weight.data))}")
        print(f"{mode} lora_B: {torch.mean(torch.abs(m.lora_B[active_adapter].weight.data))}")
    elif isinstance(m, Lora.Embedding):
        print(f"{mode} lora_embedding_A: {torch.mean(torch.abs(m.lora_embedding_A[active_adapter].data))}")
        print(f"{mode} lora_embedding_B: {torch.mean(torch.abs(m.lora_embedding_B[active_adapter].data))}")
    elif isinstance(m, SVDLinear):
        print(f"{mode} lora_A: {torch.mean(torch.abs(m.lora_A[active_adapter]))}, top 5 lora_A: {torch.mean(torch.abs(m.lora_A[active_adapter][:5, :]), dim=1)}")
        print(f"{mode} lora_active_A: {torch.mean(torch.abs(m.lora_active_A[active_adapter]))}")
        print(f"{mode} lora_E: {torch.mean(torch.abs(m.lora_E[active_adapter]))}, min: {torch.min(m.lora_E[active_adapter])}, max: {torch.max(m.lora_E[active_adapter])}")
        print(f"{mode} lora_B: {torch.mean(torch.abs(m.lora_B[active_adapter]))}")
        print(f"{mode} lora_active_B: {torch.mean(torch.abs(m.lora_active_B[active_adapter]))}")
        print(f"{mode} weight: {torch.mean(torch.abs(m.base_layer.weight))}")
        print(f"{mode} merged weight: {torch.mean(torch.abs(m.base_layer.weight + m.get_delta_weight(active_adapter)))}")

def merge_refresh(model: torch.nn.Module, args, optimizer, last_lora_A, last_lora_B, key_list):
    for n, m in model.named_modules():
        # print(f"{n}: {m}, {type(m)}")
        if isinstance(m, LoraLayer) and (not isinstance(m, TransposedEmbedding)):
            print(f"{n}")
            active_adapter = m.active_adapter[0]
            if isinstance(m, SVDLinear):
                cur_head = (m.cur_head + 1) % m.head
                if cur_head == 0:
                    print("resvd")
                    print_lora(m, "before")
                    m.weight.data += m.get_delta_weight(active_adapter)
                    init_layer(m, 1, args, last_lora_A[n], last_lora_B[n])
                    m.weight.data -= m.get_delta_weight(active_adapter)
                    m.cur_head = cur_head
                else:
                    m.update_mask(cur_head)
                print_lora(m, "after")
            else:
                m.weight.data += m.get_delta_weight(active_adapter)
                print_lora(m, "before")
                init_layer(m, 1, args, last_lora_A[n], last_lora_B[n])
                print_lora(m, "after")
                m.weight.data -= m.get_delta_weight(active_adapter)
            if isinstance(m, Lora.Linear):
                if not set(optimizer.state[m.lora_A[active_adapter].weight].keys()).issubset(set(key_list)):
                    print(optimizer.state[m.lora_A[active_adapter].weight].keys())
                    raise Exception("invalid optimizer key")
            for key in key_list:
                if isinstance(m, Lora.Linear) or isinstance(m, Lora.Conv2d):
                    parameters = [m.lora_A[active_adapter].weight, m.lora_B[active_adapter].weight]
                elif isinstance(m, Lora.Embedding):
                    parameters = [m.lora_embedding_A[active_adapter], m.lora_embedding_B[active_adapter]]
                elif isinstance(m, SVDLinear):
                    parameters = [m.lora_active_A[active_adapter], m.lora_active_B[active_adapter], m.lora_E[active_adapter]]
                else:
                    raise Exception("invalid type of loralayer")
                for parameter in parameters:
                    if isinstance(optimizer.state[parameter][key], int):
                        optimizer.state[parameter][key] = 0
                    else:
                        optimizer.state[parameter][key].zero_()

def init_layer(m, beta, args, last_lora_A=None, last_lora_B=None, clear=False, first=False):
    with torch.no_grad():
        if clear == True:
            nn.init.zeros_(m.weight)
        if isinstance(m, Lora.Linear) or isinstance(m, Lora.Conv2d):
            active_adapter = m.active_adapter[0]
            fan_in = m.in_features
            r = m.r[active_adapter]
            fan_out = m.out_features
            if (args.init in ["lora_half", "lora_momentum"] and first == True) or args.init == "lora":
                nn.init.kaiming_uniform_(m.lora_A[active_adapter].weight, a=math.sqrt(5))
                nn.init.zeros_(m.lora_B[active_adapter].weight)
            elif args.init == "lora_B":
                nn.init.zeros_(m.lora_A[active_adapter].weight)
                nn.init.kaiming_uniform_(m.lora_B[active_adapter].weight, a=math.sqrt(5))
            elif args.init == "lora_half":
                bound = 1 / math.sqrt(fan_in)
                value = torch.rand(r // 2, fan_in) * 2 * bound - bound
                m.lora_A[active_adapter].weight[r // 2:, :] = value
                m.lora_A[active_adapter].weight.data = torch.flip(m.lora_A[active_adapter].weight, [0])
                value = torch.zeros(fan_out, r // 2)
                m.lora_B[active_adapter].weight[:, r // 2:] = value
                m.lora_B[active_adapter].weight.data = torch.flip(m.lora_B[active_adapter].weight, [1])
            elif (args.init == "momentum" and first == True) or args.init == "random":
                weight_bound_A = 1 / math.sqrt(math.sqrt(fan_in * r * beta))
                weight_bound_B = 1 / math.sqrt(math.sqrt(fan_out * r * beta))
                torch.nn.init.uniform_(m.lora_A[active_adapter].weight, -weight_bound_A, weight_bound_A)
                torch.nn.init.uniform_(m.lora_B[active_adapter].weight, -weight_bound_B, weight_bound_B)
            elif args.init == "momentum":
                weight_bound_A = 1 / math.sqrt(math.sqrt(fan_in * r * beta))
                weight_bound_B = 1 / math.sqrt(math.sqrt(fan_out * r * beta))
                lora_A_init = m.lora_A[active_adapter].weight.data - last_lora_A
                variance = lora_A_init.pow(2).sum()
                lora_A_init = torch.clamp(
                    lora_A_init * torch.rsqrt(variance + 1e-8), -weight_bound_A,
                    weight_bound_A)
                m.lora_A[active_adapter].weight.data = lora_A_init
                lora_B_init = m.lora_B[active_adapter].weight.data - last_lora_B
                variance = lora_B_init.pow(2).sum()
                lora_B_init = torch.clamp(
                    lora_B_init * torch.rsqrt(variance + 1e-8), -weight_bound_B,
                    weight_bound_B)
                m.lora_B[active_adapter].weight.data = lora_B_init
            elif args.init == "lora_momentum":
                weight_bound_A = 1 / math.sqrt(fan_in * beta)
                lora_A_init = m.lora_A[active_adapter].weight.data - last_lora_A
                variance = lora_A_init.pow(2).sum()
                lora_A_init = torch.clamp(
                    lora_A_init * torch.rsqrt(variance + 1e-8), -weight_bound_A,
                    weight_bound_A)
                m.lora_A[active_adapter].weight.data = lora_A_init
                nn.init.zeros_(m.lora_B[active_adapter].weight)
            else:
                raise Exception("invalid init")
        elif isinstance(m, SVDLinear):
            active_adapter = m.active_adapter[0]
            r = m.r[active_adapter]
            U, S, Vh = torch.linalg.svd(m.base_layer.weight, full_matrices=False)
            if args.init == "svd_shuffle":
                rand_index = torch.randperm(r)
                U = U[:, rand_index]
                Vh = Vh[rand_index, :]
                S = S[rand_index]
            m.lora_active_B[active_adapter][:, :] = U[:, :m.block_size]
            m.lora_B[active_adapter][:, :] = U[:, m.block_size:r]
            m.lora_active_A[active_adapter][:, :] = Vh[:m.block_size, :]
            m.lora_A[active_adapter][:, :] = Vh[m.block_size:r, :]
            m.lora_E[active_adapter][:, :] = S[:r].unsqueeze(1)
            if args.init == "svd_adaptive":
                m.update_mask(m.cur_head)
            if first:
                m.weight.data -= m.get_delta_weight(active_adapter)

        elif isinstance(m, Lora.Embedding):
            active_adapter = m.active_adapter[0]
            nn.init.zeros_(m.lora_embedding_A[active_adapter])
            nn.init.kaiming_uniform_(m.lora_embedding_B[active_adapter], mode="fan_out", a=math.sqrt(5))
        else:
            raise Exception("invalid type of loralayer")


def get_visible_devices():
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        return list(map(int, cuda_visible_devices.split(',')))
    else:
        return list(range(pynvml.nvmlDeviceGetCount()))

def get_free_memory():
    visible_devices = get_visible_devices()
    print(f"visible_devices: {visible_devices}")
    memory_info = []
    for virtual_index, real_index in enumerate(visible_devices):
        handle = pynvml.nvmlDeviceGetHandleByIndex(real_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = info.free
        memory_info.append((virtual_index, free_memory))
    return memory_info


def select_device_with_most_free_memory():
    pynvml.nvmlInit()
    free_memory = get_free_memory()
    device_with_max_memory = max(free_memory, key=lambda x: x[1])
    return device_with_max_memory[0]


class CustomConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
    The dataset also formats the text before tokenization with a specific format that is provided
    by the user.

        Args:
            tokenizer (`transformers.PreTrainedTokenizer`):
                The processor used for processing the data.
            dataset (`dataset.Dataset`):
                Dataset with text files.
            dataset_text_field (`str`, **optional**):
                Name of the field in the dataset that contains the text. Used only if `formatting_func` is `None`.
            formatting_func (`Callable`, **optional**):
                Function that formats the text before tokenization. Usually it is recommended to have follows a certain
                pattern such as `"### Question: {question}\n ### Answer: {answer}\n"`
            infinite (`bool`, *optional*, defaults to `False`):
                If True the iterator is reset after dataset reaches end else stops.
            seq_length (`int`, *optional*, defaults to `1024`):
                Length of token sequences to return.
            num_of_sequences (`int`, *optional*, defaults to `1024`):
                Number of token sequences to keep in buffer.
            chars_per_token (`int`, *optional*, defaults to `3.6`):
                Number of characters per token used to estimate number of tokens in text buffer.
            eos_token_id (`int`, *optional*, defaults to `0`):
                Id of the end of sequence token if the passed tokenizer does not have an EOS token.
            shuffle ('bool', *optional*, defaults to True)
                Shuffle the examples before they are returned
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        dataset_text_field=None,
        formatting_func=None,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        eos_token_id=0,
        shuffle=True,
    ):
        self.tokenizer = tokenizer

        if tokenizer.eos_token_id is None:
            warnings.warn(
                "The passed tokenizer does not have an EOS token. We will use the passed eos_token_id instead which corresponds"
                f" to {eos_token_id}. If this is not the correct EOS token, make sure to pass the correct eos_token_id."
            )

        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.shuffle = shuffle
        if formatting_func is None:
            self.formatting_func = lambda x: x[dataset_text_field]
        else:
            self.formatting_func = formatting_func

        self.current_epoch = 0
        if formatting_func is not None:
            formatting_func_signature = formatting_func.__code__.co_varnames
            if len(formatting_func_signature) > 1:
                warnings.warn(
                    "The passed formatting_func has more than one argument. Usually that function should have a single argument `example`"
                    " which corresponds to the dictionary returned by each element of the dataset. Make sure you know what you are doing."
                )

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(self.formatting_func(next(iterator)))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        self.current_epoch += 1
                        iterator = iter(self.dataset)
                        warnings.warn("The dataset reached end and the iterator is reset to the start.")
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


class MySFTTrainer(SFTTrainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            raise Exception("invalid model type")
            # if state_dict is None:
            #     state_dict = self.model.state_dict()
            #
            # if isinstance(unwrap_model(self.model), supported_classes):
            #     if isinstance(unwrap_model(self.model), PeftModel):
            #         print("save all peft model")
            #         unwrap_model(self.model).base_model.model.save_pretrained(
            #             output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            #         )
            #     else:
            #         print("save full parameter model")
            #         unwrap_model(self.model).save_pretrained(
            #             output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            #         )
            # else:
            #     print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            #     if self.args.save_safetensors:
            #         safetensors.torch.save_file(state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME))
            #     else:
            #         torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.model, PeftModel):
                print("save all peft model")
                torch.save(state_dict, os.path.join(output_dir, "all_model.pt"))
                # self.model.base_model.model.save_pretrained(
                #     output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                # )
            else:
                print("save full parameter model")
                torch.save(state_dict, os.path.join(output_dir, "all_model.pt"))
            # self.model.save_pretrained(
            #     output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            # )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

