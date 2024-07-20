import torch
import tensorflow as tf
ce = torch.nn.CrossEntropyLoss()
def test_batch_cls(model, x, y): # classification
    outputs = model(x, labels=y)
    logits = outputs.logits.detach()
    loss = outputs.loss # huggingface loss is already averaged
    correct = (logits.argmax(dim=1) == y.cuda()).sum().item()
    total = len(y)
    return loss, correct, total

def test_batch_kd(student_model, x, y):
    # Note: logits from test_batch_cls are shape [batch_size, n_classes] while logits from test_batch_kd are shape [batch_size, seq_len, n_classes]
    assert hasattr(student_model, 'score')
    outputs = student_model(x.cuda(), output_hidden_states=True)
    non_pad_mask = (x != 50256)
    last_non_pad_indices = non_pad_mask.sum(dim=1) - 1
    last_hidden = outputs.hidden_states[-1]
    last_hidden = last_hidden[torch.arange(len(x)), last_non_pad_indices, :]
    logits = student_model.score(last_hidden).detach()
    loss = ce(logits, y.cuda())
    correct = (logits.argmax(dim=1) == y.cuda()).sum().item()
    total = len(y)
    return loss, correct, total


def test_batch_nwp(model, x): # next word (token) prediction
    output = model(x)
    logits = output.logits[:, :-1]
    flat_logits = logits.reshape(-1, 50257) # exclude last token
    loss = torch.nn.functional.nll_loss(
        torch.nn.functional.log_softmax(flat_logits, dim=-1), # flat predictions
        x[:, 1:].reshape(-1), # flat tokens
        ignore_index=50256,
        reduction='sum'
    )
    non_pad_idx = x[:, 1:] != 50256                       # [B, S]: bool
    total = non_pad_idx.sum().item()                      # [sentences]: int
    with torch.no_grad():
        pred_toks = logits.argmax(dim=-1)                 # [sentences, tokens]: 0...50256
        correct_toks = pred_toks == x[:, 1:]              # [sentences, tokens]: bool
        correct = (non_pad_idx*correct_toks).sum().item() # [sentences]: int
    return loss, correct, total

def test_batch(model, x, y):
    if y[0] == -1:
        loss, correct, total = test_batch_nwp(model, x.cuda())
        loss /= total
    else:
        loss, correct, total = test_batch_cls(model, x.cuda(), y.cuda())
    return loss, correct, total

def eval_loop(model, testloader):
    model.eval()
    total_count, total_correct = 0, 0
    for x,y in testloader:
        with torch.no_grad():
            _, correct, count = test_batch(model, x, y)
        total_count += count
        total_correct += correct
    model.train()
    return total_correct / total_count

def eval_loop_kd(student_model, testloader):
    student_model.eval()
    total_count, total_correct = 0, 0
    for x,y in testloader:
        with torch.no_grad():
            _, correct, count = test_batch_kd(student_model, x, y)
        total_count += count
        total_correct += correct

    student_model.train()
    return total_correct / total_count
