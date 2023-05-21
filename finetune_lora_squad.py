"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import os
import time
import json
import logging

import lightning as L
import numpy as np
import torch

from generate import generate
from lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from lit_llama.model import LLaMA, LLaMAConfig
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt

VALIDATE_PERCENTAGE = 0.8
eval_interval = 500
log_loss_iters = 250
eval_iters = 2500
log_interval = 100
devices = 2
# Hyperparameters
learning_rate = 3e-4
batch_size = 128 // devices
micro_batch_size = 1
gradient_accumulation_steps = batch_size // micro_batch_size
epoch = 15
training_data_size = 120000
max_iters = (training_data_size * epoch) // devices // micro_batch_size
weight_decay = 0.0
max_seq_length = 512  
lora_r = 16
lora_alpha = 16
lora_dropout = 0.05
warmup_steps = 100

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

def main(
    data_dir: str = "data/squad2", 
    pretrained_path: str = "checkpoints/lit-llama/7B/lit-llama.pth",
    out_dir: str = "out/lora/squad2_r16_new_loss",
):

    fabric = L.Fabric(accelerator="cuda", devices=devices, precision="bf16-true",strategy="ddp")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets(data_dir=data_dir)

    config = LLaMAConfig.from_name("7B")
    config.block_size = max_seq_length

    checkpoint = torch.load(pretrained_path)

    with fabric.init_module(), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
        model = LLaMA(config)
        # strict=False because missing keys due to LoRA weights not contained in checkpoint state
        model.load_state_dict(checkpoint, strict=False)
    
    mark_only_lora_as_trainable(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, train_data, val_data, out_dir)

    # Save the final LoRA checkpoint at the end of training
    checkpoint = lora_state_dict(model)
    fabric.save(os.path.join(out_dir, "lit-llama-lora-finetuned.pth"), checkpoint)


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
    
    for iter_num in range(max_iters):

        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        is_accumulating = (iter_num + 1) % gradient_accumulation_steps != 0
    
        input_ids, targets = get_batch(fabric, train_data)

        with fabric.no_backward_sync(model,enabled=is_accumulating):
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

            if step_count % eval_interval == 0: #and val_loss < prev_val_loss:
                prev_val_loss = val_loss 
                # print(f"Saving LoRA weights to {out_dir}")
                # We are only saving the LoRA weights
                # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
                checkpoint = lora_state_dict(model)
                fabric.save(os.path.join(out_dir, f"iter-{iter_num}-loss-{val_loss:.4f}-ckpt.pth"), checkpoint)
                fabric.barrier()
            
            train_loss = 0 # reset the train loss (accumulate and reset)

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")


def generate_response(model, instruction):
    tokenizer = Tokenizer("checkpoints/lit-llama/tokenizer.model")
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=max_seq_length,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output)
    return output # output.split("### Response:")[1].strip()

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
