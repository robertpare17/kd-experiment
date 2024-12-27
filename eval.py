import os
import random
import numpy as np
import argparse
from tqdm import tqdm
from copy import deepcopy
import pickle
from datasets import load_dataset
from transformers import GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPast

class GPT2ForCausalLMFromSeqClass(GPT2Model):
    def __init__(self, original_model):
        super().__init__(original_model.config)
        self.transformer = original_model.transformer  # Keep the transformer layers
        self.lm_head = torch.nn.Linear(original_model.config.hidden_size, original_model.config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # Tie weights with the input embeddings

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Pass inputs through the transformer layers
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        hidden_states = transformer_outputs.last_hidden_state

        # Use the lm_head to get logits over the vocabulary
        lm_logits = self.lm_head(hidden_states)

        # Create an output object with a logits attribute
        output = BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=None, hidden_states=None, attentions=None)
        output.logits = lm_logits

        return output

def str2bool(s):
    return s.lower() == 'true'

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',           default='0',            type=str)
    parser.add_argument('--dir',           default='runs',         type=str)
    parser.add_argument('--name',          default='test',         type=str)
    parser.add_argument('--dataset',       default='openwebtext',  type=str)
    parser.add_argument('--eval-dataset',  default='20newsgroups',  type=str)
    parser.add_argument('--seed',          default=0,              type=int)
    parser.add_argument('--eval-freq',     default=1,              type=int)
    parser.add_argument('--num_layers_rem',default=0,              type=int)
    # optimization
    parser.add_argument('--lr',            default=1e-4,           type=str)
    parser.add_argument('--batch_size',    default=16,             type=int)
    parser.add_argument('--epochs',        default=1,              type=int)
    parser.add_argument('--loss',          default='MSE',          type=str)
    parser.add_argument('--loadseqmodel',  default='false',        type=str)
    parser.add_argument('--columns_rem',   default=0,              type=int)
    parser.add_argument('--by_norm',       default='true',         type=str)
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
args.name = f'eval_width_norm_heuristic_{args.columns_rem}_columns_rem'
run_dir = f"{args.dir}/{args.name}"
os.makedirs(run_dir)
import json
with open(f"{run_dir}/args.json", 'w') as f:
    json.dump(vars(args), f, indent=4)
    print(f"Saved args to {run_dir}")

# with open(f'checkpoints/kd_{args.num_layers_rem}_layers_rem_gpt2_openwebtext_epoch-1_kd.pkl', 'rb') as f:
with open(f'checkpoints/gpt2_20newsgroups_{args.columns_rem}_columns_rem_by_norm_{args.by_norm}.pkl', 'rb') as f:
        tuned_model = pickle.load(f)
tuned_model.cuda()


from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
model_name = "gpt2"
model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=20,pad_token_id=50256)
model.cuda()

by_norm = str2bool(args.by_norm)
k = args.columns_rem
for name, param in model.named_parameters():
    if name in tuned_model.state_dict():
        # model.state_dict()[name].copy_(tuned_model.state_dict()[name])
        # assert(torch.equal(model.state_dict()[name], tuned_model.state_dict()[name]))
        if k > 0 and ('mlp.c_proj.weight' in name or 'attn.c_proj.weight' in name or 'wte' in name or 'wpe' in name):
            if by_norm:
                weight_norms = torch.norm(param.data, dim=0)
                smallest_indices = torch.topk(weight_norms, k, largest=False).indices
            else:
                smallest_indices = torch.arange(k)
            assert(torch.sum(tuned_model.state_dict()[name][:, smallest_indices]) == 0)
            num_cols = param.shape[1]
            all_cols = torch.arange(num_cols).cuda()
            select_cols = ~torch.isin(all_cols, smallest_indices)
            model.state_dict()[name][:, select_cols].copy_(tuned_model.state_dict()[name][:, select_cols])

        elif k > 0 and ('attn.c_proj.bias' in name or 'mlp.c_proj.bias' in name):
            if by_norm:
                assert(torch.sum(tuned_model.state_dict()[name][smallest_indices]) == 0)
                model.state_dict()[name][select_cols].copy_(tuned_model.state_dict()[name][select_cols])
            else:
                assert(torch.sum(tuned_model.state_dict()[name][:k]) == 0)
                model.state_dict()[name][k:].copy_(tuned_model.state_dict()[name][k:])

        elif k > 0 and 'ln_f' in name:
            if by_norm:
                abs_param = torch.abs(param.data)
                ln_f_smallest_indices = torch.topk(abs_param, k, largest=False).indices
            else:
                ln_f_smallest_indices = torch.arange(k)
            assert(torch.sum(tuned_model.state_dict()[name][ln_f_smallest_indices]) == 0)
            num_cols = param.shape[0]
            all_cols = torch.arange(num_cols).cuda()
            select_cols = ~torch.isin(all_cols, smallest_indices)
            model.state_dict()[name][select_cols].copy_(tuned_model.state_dict()[name][select_cols])

        else:
            model.state_dict()[name].copy_(tuned_model.state_dict()[name])
            assert(torch.equal(model.state_dict()[name], tuned_model.state_dict()[name]))


import data_utils
# trainloader, testloader = data_utils.build_dataset(args.dataset, args.batch_size)
trainloader, testloader = data_utils.build_dataset(args.eval_dataset, args.batch_size)

from train_utils import eval_loop


def evaluate(model, testloader, loadseqmodel):
    if loadseqmodel:
       model = GPT2ForCausalLMFromSeqClass(model)

    writer = tf.summary.create_file_writer(run_dir)
    eval_accu = eval_loop(model, testloader)
    with writer.as_default():
        tf.summary.scalar('eval/accuracy', eval_accu, step=1)

evaluate(
    model = model,
    testloader = testloader,
    loadseqmodel = str2bool(args.loadseqmodel)
)
