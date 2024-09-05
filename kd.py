import os
import random
import numpy as np
import argparse
from tqdm import tqdm
from copy import deepcopy
import pickle
from datasets import load_dataset


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',           default='0',            type=str)
    parser.add_argument('--dir',           default='runs',         type=str)
    parser.add_argument('--name',          default='test',         type=str)
    parser.add_argument('--dataset',       default='openwebtext',  type=str)
    parser.add_argument('--eval-dataset',  default='20newsgroups', type=str)
    parser.add_argument('--seed',          default=0,              type=int)
    parser.add_argument('--eval-freq',     default=1,              type=int)
    parser.add_argument('--num_layers_rem',default=-1,              type=int)
    # optimization
    parser.add_argument('--lr',            default=1e-4,           type=str)
    parser.add_argument('--batch_size',    default=16,             type=int)
    parser.add_argument('--epochs',        default=1,             type=int)
    parser.add_argument('--loss',          default='MSE',          type=str)
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
args.name = f'kd_{args.num_layers_rem}_layers_rem'
run_dir = f"{args.dir}/{args.name}"
os.makedirs(run_dir)
import json
with open(f"{run_dir}/args.json", 'w') as f:
    json.dump(vars(args), f, indent=4)
    print(f"Saved args to {run_dir}")

from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
# model_name = "gpt2"
# model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=20,pad_token_id=50256)
# model.cuda()

model_name = "gpt2"
student_model = AutoModelForCausalLM.from_pretrained(model_name,pad_token_id=50256)
student_model.cuda()

with open(f'checkpoints/gpt2_20newsgroups_{args.num_layers_rem}_layers_rem.pkl', 'rb') as f:
        teacher_model = pickle.load(f)
teacher_model.cuda()

# Freeze teacher model for inference
for name, param in teacher_model.named_parameters():
    param.requires_grad = False
teacher_model.eval()

import data_utils
trainloader, _ = data_utils.build_dataset(args.dataset, args.batch_size)
_, testloader = data_utils.build_dataset(args.eval_dataset, args.batch_size)
# trainloader, testloader = data_utils.build_dataset(args.dataset, args.batch_size)


def save_model(model, name):
    print("Saving checkpoint...")
    with open(f'checkpoints/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

from train_utils import test_batch, eval_loop, eval_loop_kd
# def train(model, lr, epochs, trainloader, testloader, eval_freq):
#     writer = tf.summary.create_file_writer(run_dir)
#     pbar = tqdm(range(epochs))

#     opt = torch.optim.AdamW(model.parameters(), lr=lr)
#     sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.99)

#     eval_accu = 0
#     for epoch in pbar:
#         prev_eval_accu = eval_accu
#         prev_model = deepcopy(model)
#         for x,y in tqdm(trainloader):
#             loss, correct, total = test_batch(model, x, y)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()

#         sched.step()

#         if (epoch+1) % eval_freq == 0:
#             eval_accu = eval_loop(model, testloader)
#             with writer.as_default():
#                 tf.summary.scalar('eval/accuracy', eval_accu, step=epoch+1)
#         pbar.set_description(f"eval: {eval_accu}")
#         if prev_eval_accu > eval_accu:
#             # Save checkpoint
#             save_model(prev_model, f'{model_name}_{args.dataset}')
#             break

def knowledge_distill(student_model, teacher_model, trainloader, testloader, criterion, lr, eval_freq, epochs=10):
    '''
    Instantiate GPT2 model without any particular head
    Use MSE loss to train fresh GPT2 model
    Attach score weight matrix from teacher model onto fresh GPT2 model and evaluate model on 20newsgroups
    '''
    writer = tf.summary.create_file_writer(run_dir)
    pbar = tqdm(range(epochs))

    opt = torch.optim.AdamW(student_model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.99)

    for epoch in pbar:
        for idx, (x,y) in enumerate(tqdm(trainloader)):
            output = student_model(x.cuda(), output_hidden_states=True)
            teacher_output = teacher_model(x.cuda(), output_hidden_states=True)
            loss = criterion(output.hidden_states[-1], teacher_output.hidden_states[-1].detach().cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()

            if idx % 1000 == 0:
                with writer.as_default():
                    tf.summary.scalar('train/loss', loss.cpu().detach().numpy(), step=idx//1000)

        sched.step()


        student_model.score = deepcopy(teacher_model.score)
        if (epoch+1) % eval_freq == 0:
            eval_accu = eval_loop_kd(student_model, testloader)
        with writer.as_default():
            tf.summary.scalar('eval/accuracy', eval_accu, step=epoch+1)
        pbar.set_description(f"eval: {eval_accu}")

        save_model(student_model, f'{args.name}_{model_name}_{args.dataset}_epoch-{epoch+1}_kd')


# train(
#     model = model,
#     lr = args.lr,
#     epochs = args.epochs,
#     trainloader = trainloader,
#     testloader = testloader,
#     eval_freq=args.eval_freq
# )

knowledge_distill(
    student_model = student_model,
    teacher_model = teacher_model,
    trainloader = trainloader,
    testloader = testloader,
    criterion = torch.nn.MSELoss(),
    lr = args.lr,
    eval_freq = args.eval_freq,
    epochs = args.epochs
)

# NOTE: runs/test_kd contains run for one epoch of kd with no pruning -- change this name later
