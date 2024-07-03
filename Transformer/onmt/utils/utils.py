import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from peft.utils.other import transpose
import random
import numpy as np
from peft import PeftModel
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

