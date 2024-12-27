import os
import random
import numpy as np
import argparse
from tqdm import tqdm
from copy import deepcopy
import pickle
from datasets import load_dataset

def str2bool(s):
    return s.lower() == 'true'

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',           default='0',            type=str)
    parser.add_argument('--dir',           default='runs',         type=str)
    parser.add_argument('--name',          default='test',         type=str)
    parser.add_argument('--dataset',       default='20newsgroups', type=str)
    parser.add_argument('--eval-dataset',  default='20newsgroups', type=str)
    parser.add_argument('--seed',          default=0,              type=int)
    parser.add_argument('--eval-freq',     default=1,              type=int)
    parser.add_argument('--num_layers_rem',default=0,              type=int)
    # optimization
    parser.add_argument('--lr',            default=1e-4,           type=str)
    parser.add_argument('--batch_size',    default=16,             type=int)
    parser.add_argument('--epochs',        default=10,             type=int)
    parser.add_argument('--loss',          default='MSE',          type=str)
    parser.add_argument('--loadseqmodel',  default='false',        type=str)
    parser.add_argument('--columns_rem',   default=0,              type=int)
    parser.add_argument('--by_norm',       default='true',        type=str)
    return parser.parse_args()

# Must set CUDA_VISIBLE_DEVICES before importing any GPU-related libraries
args = parse()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch
import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    print("CUDA is available! Training on GPU.")
else:
    print("CUDA is not available. Training on CPU.")

# Create run folder and args.json; raise error if folder exists.
args.name = f'train_width_norm_heuristic_{args.columns_rem}_columns_rem'
run_dir = f"{args.dir}/{args.name}"
os.makedirs(run_dir)
import json
with open(f"{run_dir}/args.json", 'w') as f:
    json.dump(vars(args), f, indent=4)
    print(f"Saved args to {run_dir}")

from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
model_name = "gpt2"
model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=20,pad_token_id=50256)
model.cuda()

K = args.num_layers_rem
keep_layers = len(model.transformer.h) - K

assert keep_layers > 0, "Cannot remove all layers. The model must retain at least one layer."
print(f'Number of layers remaining: {keep_layers}')
model.transformer.h = model.transformer.h[:keep_layers]
model.config.num_hidden_layers -= K

cols_rem = args.columns_rem

def zero_out_grad_weight(cols_indices):
    def hook(grad):
        grad[:, cols_indices] = 0
        return grad
    return hook

def zero_out_grad_bias(cols_indices):
    def hook(grad):
        grad[cols_indices] = 0
        return grad
    return hook

def zero_and_freeze_k_features(model, k, by_norm=False):
    # Iterate through the transformer blocks
    for block in model.transformer.h:
        if by_norm:
            # Compute the norms of the columns for both mlp and attn weights
            mlp_weight_norms = torch.norm(block.mlp.c_proj.weight.data, dim=0)
            attn_weight_norms = torch.norm(block.attn.c_proj.weight.data, dim=0)

            # Get the indices of the k smallest norms
            mlp_smallest_indices = torch.topk(mlp_weight_norms, k, largest=False).indices
            attn_smallest_indices = torch.topk(attn_weight_norms, k, largest=False).indices
        else:
            mlp_smallest_indices = torch.arange(k)
            attn_smallest_indices = torch.arange(k)

        # Zero out the corresponding columns for mlp and attn weights
        block.mlp.c_proj.weight.data[:, mlp_smallest_indices] = 0
        block.attn.c_proj.weight.data[:, attn_smallest_indices] = 0

        # Optionally zero out the corresponding biases
        block.attn.c_proj.bias.data[attn_smallest_indices] = 0
        block.mlp.c_proj.bias.data[mlp_smallest_indices] = 0

        # Register gradient hook to prevent updates
        block.mlp.c_proj.weight.register_hook(zero_out_grad_weight(mlp_smallest_indices))
        block.attn.c_proj.weight.register_hook(zero_out_grad_weight(attn_smallest_indices))
        block.attn.c_proj.bias.register_hook(zero_out_grad_bias(attn_smallest_indices))
        block.mlp.c_proj.bias.register_hook(zero_out_grad_bias(mlp_smallest_indices))

    for name, param in model.named_parameters():
        if 'wte' in name or 'wpe' in name:
            if by_norm:
                embed_norms = torch.norm(param.data, dim=0)
                embed_smallest_indices = torch.topk(embed_norms, k, largest=False).indices
            else:
                embed_smallest_indices = torch.arange(k)

            param.data[:, embed_smallest_indices] = 0
            param.register_hook(zero_out_grad_weight(embed_smallest_indices))
        if 'ln_f' in name:
            if by_norm:
                abs_param = torch.abs(param.data)
                ln_f_smallest_indices = torch.topk(abs_param, k, largest=False).indices
            else:
                ln_f_smallest_indices = torch.arange(k)
            param.data[ln_f_smallest_indices] = 0
            param.register_hook(zero_out_grad_bias(ln_f_smallest_indices))


# Zero and freeze parameters
zero_and_freeze_k_features(model, k=cols_rem, by_norm=str2bool(args.by_norm))


import data_utils
trainloader, testloader = data_utils.build_dataset(args.dataset, args.batch_size)
# _, testloader = data_utils.build_dataset(args.eval_dataset, args.batch_size)

from train_utils import test_batch, eval_loop
def save_model(model, name):
    print("Saving checkpoint...")
    with open(f'checkpoints/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)


def train(model, lr, epochs, trainloader, testloader, eval_freq):
    writer = tf.summary.create_file_writer(run_dir)
    pbar = tqdm(range(epochs))

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.99)

    eval_accu = 0
    for epoch in pbar:
        prev_eval_accu = eval_accu
        prev_model = deepcopy(model)
        for x,y in tqdm(trainloader):
            loss, correct, total = test_batch(model, x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            

        sched.step()
        if (epoch+1) % eval_freq == 0:
            eval_accu = eval_loop(model, testloader)
            with writer.as_default():
                tf.summary.scalar('eval/accuracy', eval_accu, step=epoch+1)
        pbar.set_description(f"eval: {eval_accu}")
        if prev_eval_accu > eval_accu:
            # Save checkpoint
            save_model(prev_model, f'{model_name}_{args.dataset}_{args.columns_rem}_columns_rem_by_norm_{args.by_norm}')
            break

train(
    model = model,
    lr = args.lr,
    epochs = args.epochs,
    trainloader = trainloader,
    testloader = testloader,
    eval_freq=args.eval_freq
)
