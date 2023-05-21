"""
Instruction-tuning on the Alpaca dataset using a regular finetuning procedure (updating all layers).

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import os
import time
import logging
from functools import partial

import lightning as L
from lightning.fabric.strategies import FSDPStrategy
import numpy as np
import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from generate import generate
from lit_llama.model import Block, LLaMA, LLaMAConfig
from lit_llama.tokenizer import Tokenizer
from lit_llama.utils import save_model_checkpoint
from scripts.prepare_alpaca import generate_prompt

VALIDATE_PERCENTAGE = 0.85 # percentage of data to validate on
eval_interval = 2000
log_loss_iters = 1000
eval_iters = 4000
log_interval = 100
devices = 2

# Hyperparameters
learning_rate = 3e-5
batch_size = 256 / devices
micro_batch_size = 32
gradient_accumulation_steps = batch_size // micro_batch_size
epoch_size = 130000  # train dataset size
num_epochs = 10
max_iters = num_epochs * epoch_size // micro_batch_size // devices
weight_decay = 0.0
block_size = 512
warmup_steps = 100


def main(
    data_dir: str = "data/squad2",
    pretrained_path: str = "checkpoints/lit-llama/7B/lit-llama.pth",
    out_dir: str = "out/full/squad2_trial1",
):

    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, activation_checkpointing=Block)

    fabric = L.Fabric(accelerator="cuda", devices=devices, precision="bf16-mixed", strategy=strategy)
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets(data_dir=data_dir)

    config = LLaMAConfig.from_name("7B")
    config.block_size = block_size

    checkpoint = torch.load(pretrained_path)

    with fabric.device:
        torch.set_default_tensor_type(torch.HalfTensor)
        model = LLaMA(config).bfloat16()
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False) 

    model = fabric.setup_module(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = fabric.setup_optimizers(optimizer)

    train(fabric, model, optimizer, train_data, val_data, out_dir)

    # Save the final checkpoint at the end of training
    save_model_checkpoint(fabric, model, os.path.join(out_dir, "lit-llama-full-finetuned.pth"))


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    out_dir: str,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0
    prev_val_loss = 9999
    train_loss = 0
    train_loss_logger = setup_logger("Train Loss Log", os.path.join(out_dir,"train_loss.log"))
    val_loss_logger = setup_logger("Val Loss Log", os.path.join(out_dir,"val_loss.log"))
    validate_output_logger = setup_logger("Val output log", os.path.join(out_dir,"val_out.log"))
    
    model.train()

    for iter_num in range(max_iters):

        is_accumulating = (iter_num + 1) % gradient_accumulation_steps == 0

        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            input_ids, targets = get_batch(fabric, train_data)
            logits = model(input_ids)
            loss = loss_fn(logits, targets)
            train_loss += loss
            fabric.backward(loss)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            if step_count % log_loss_iters == 0 and fabric.global_rank == 0:
                train_loss_logger.info(f"step: {step_count}, training loss: {train_loss}")

            if step_count % eval_interval == 0:
                val_loss = validate_all_data(fabric, model, val_data, logger = validate_output_logger)
                if fabric.global_rank==0:
                    val_loss_logger.info(f"step: {step_count}, val loss: {val_loss}")
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()

            if step_count % eval_interval == 0 and prev_val_loss > val_loss:
                prev_val_loss = val_loss
                print(f"Saving weights to {out_dir}")
                save_model_checkpoint(fabric, model, os.path.join(out_dir, f"iter-{iter_num:06d}-loss-{val_loss:0.4f}-ckpt.pth"))
            
            train_loss = 0
        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")

            
@torch.no_grad()
def validate_all_data(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray, logger) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    ceiling_division = lambda a,b : -(a//-b)
    data_portion = int(len(val_data)*VALIDATE_PERCENTAGE)
    data_splits = ceiling_division(data_portion, devices)
    all_losses_main = []
    for i in range(devices):
        if fabric.global_rank == i:
            start = i*data_splits
            end = (i+1)*data_splits
            data_chunk = val_data[start:end]

            # get the loss for the data on device
            all_loss = []
            input_ids = []
            labels = []
            for count, data in enumerate(data_chunk,1):

                # stack up to micro batch size
                input_ids.append(data["input_ids"].type(torch.int64))
                labels.append(data["labels"].type(torch.int64))

                if count % micro_batch_size == 0:
                    max_len = max(len(s) for s in input_ids)


                    def pad_right(x, pad_id):
                        # pad right based on the longest sequence
                        n = max_len - len(x)
                        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

                    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
                    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
                    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))

                    logits = model(x)
                    loss = loss_fn(logits, y)
                    all_loss.append(loss.item())

                    # reset input and output
                    input_ids  = []
                    labels = []
            
            # every device will add the mean loss here:
            all_losses_main.append(sum(all_loss) / len(all_loss))
    
    # ---- wait for all devices to complete (use fabric barrier) -----------
    fabric.barrier()
    losses = fabric.all_gather(all_losses_main)[0]
    assert losses.shape[0] == devices, "Each device will calculate the mean of loss. The current amount of loss data does not match amount of devices "
    model.train()
    return losses.mean().item()

def generate_response(model, instruction):
    tokenizer = Tokenizer("checkpoints/lit-llama/tokenizer.model")
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=block_size,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output)
    return output # output.split("### Response:")[1].strip()


# Setting up logger
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')      
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger



def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss


def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse.cli import CLI

    CLI(main)
