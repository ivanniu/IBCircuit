import os, sys
sys.path.append(os.getcwd())
from transformers import AutoTokenizer, AutoModelForCausalLM
import hydra
from omegaconf import DictConfig,OmegaConf
import torch
import random
import numpy as np
import torch.nn.functional as F
from models.gpt2_IB import IB_GPT2LMHeadModel
from dsets.ioi_dataset import IOIDataset
from dsets.greaterthan_dataset import get_year_data
from utils.neel_plotly import imshow
import time
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(cfg):
    if 'gpt2' in cfg.model.model_name:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_path)

    if 'gpt2' in cfg.model.model_name:
        model = AutoModelForCausalLM.from_pretrained(cfg.model.pretrained_model_path).to(cfg.train.device)

    tokenizer.pad_token = tokenizer.eos_token
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer

def get_data(tokenizer, cfg):
    if cfg.train.data_name == 'ioi':
        ioi_dataset = IOIDataset(
            prompt_type=["When [A] and [B] went to the [PLACE], [B] gave a [OBJECT] to [A]"],
            N=cfg.train.dataset_size_limit,
            tokenizer=tokenizer,
            prepend_bos=False,
        )
        input_ids = ioi_dataset.toks.long().to(cfg.train.device)
        ans_ids = torch.tensor(ioi_dataset.io_tokenIDs).to(cfg.train.device)
        sem_tok_idx = ioi_dataset.sem_tok_idx
        end_idxs = sem_tok_idx['end']
        max_len = ioi_dataset.max_len
    elif cfg.train.data_name == 'greaterthan':
        input_ids, end_idxs, max_len = get_year_data(cfg.train.dataset_size_limit, tokenizer, cfg.train.device)
        ans_ids = None
    return input_ids, ans_ids, end_idxs, max_len

@hydra.main(version_base='1.1', config_path='./configs', config_name='config')
def main(cfg: DictConfig):
    stime = time.time()
    set_seed(cfg.train.seed)
    pretrained_model, tokenizer = get_model(cfg)
    input_ids, ans_ids, end_idxs, max_len = get_data(tokenizer, cfg)

    loss_fn = torch.nn.CrossEntropyLoss().to(cfg.train.device)
    outputs = pretrained_model(input_ids=input_ids)
    reshaped_last_logits = [outputs.logits[idx][end_idxs[idx]] for idx in range(cfg.train.dataset_size_limit)]
    reshaped_last_logits = torch.stack(reshaped_last_logits).to(cfg.train.device)
    if cfg.train.data_name == 'greaterthan':
        ans_ids = torch.argmax(reshaped_last_logits, dim=-1).to(cfg.train.device)
    target_cls_loss = loss_fn(reshaped_last_logits.squeeze(1), ans_ids)
    base_model_logprobs = F.log_softmax(reshaped_last_logits.squeeze(1), dim=-1)

    if 'gpt2' in cfg.model.model_name:
        IB_model = IB_GPT2LMHeadModel(pretrained_model, cfg, len(input_ids), cfg.train.mask_type)

    learning_params = []
    if len(IB_model.IB_weight[0]) == 2:
        for IB_params in IB_model.IB_weight:
            learning_params.append(IB_params[0])
            learning_params.append(IB_params[1])
    elif len(IB_model.IB_weight[0]) == 1:
        for IB_params in IB_model.IB_weight:
                learning_params.append(IB_params[0])
    else:
        print("length: ", len(IB_model.IB_weight[0]))

    optimizer = torch.optim.Adam(learning_params, lr=cfg.train.learning_rate)

    num_layers = cfg.model.n_layer
    for i in range(cfg.train.num_epochs):
        outputs, node_IB_lamda_list, KL_loss = IB_model(input_ids=input_ids)
        reshaped_last_logits = [outputs.logits[idx][end_idxs[idx]] for idx in range(cfg.train.dataset_size_limit)]
        reshaped_last_logits = torch.stack(reshaped_last_logits).to(cfg.train.device)
        cls_loss = loss_fn(reshaped_last_logits.squeeze(1), ans_ids)

        logprobs = F.log_softmax(reshaped_last_logits.squeeze(1), dim=-1)
        kl_div = F.kl_div(logprobs, base_model_logprobs, log_target=True, reduction="none").sum(dim=-1).mean()

        avg_node_weight = 0
        avg_node_lambda = 0

        for j in range(num_layers):
            avg_node_weight += torch.mean(IB_model.IB_weight[j][0])
            avg_node_lambda += torch.mean(node_IB_lamda_list[j])

        avg_node_weight = avg_node_weight/num_layers
        avg_node_lambda = avg_node_lambda/num_layers

        if cfg.train.alpha_loss == 'ce':
            loss = cfg.train.alpha * torch.abs(cls_loss - target_cls_loss) + cfg.train.beta * KL_loss
        elif cfg.train.alpha_loss == 'kl':
            loss = cfg.train.alpha * kl_div + cfg.train.beta * KL_loss

        if cfg.train.alpha_loss == 'ce':
            print("epoch: {} cls loss: {:.5f} IB_loss: {:.5f} avg_node_weight: {:.3f} "
                "avg_node_lambda: {:.3f}".format(
                    i, torch.abs(cls_loss - target_cls_loss), KL_loss, avg_node_weight, avg_node_lambda))
        elif cfg.train.alpha_loss == 'kl':
            print("epoch: {}  kl_div: {:.5f}  IB_loss: {:.5f} avg_node_weight: {:.3f} "
                "avg_node_lambda: {:.3f}".format(
                    i, kl_div, KL_loss, avg_node_weight, avg_node_lambda))

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    for threshold in [0.8, 0.6, 0.4, 0.2]:
        print("threshold: " + str(threshold))
        dir_name = str(cfg.train.data_name) + '_Beta_' + str(cfg.train.beta)
        dir_name += '_ts' + str(threshold)
        if not os.path.exists(cfg.train.saved_res_dir + dir_name + '/'):
            os.makedirs(cfg.train.saved_res_dir + dir_name + '/', exist_ok=True)

        bs_node_layer_head = []
        for bs in range(cfg.train.dataset_size_limit):
            node_layer_head = []
            for j in range(num_layers):
                node_IB_lambda = node_IB_lamda_list[j]
                node_layer_head.append(node_IB_lambda[bs].squeeze(1).squeeze(1))

            node_tensor = torch.stack(node_layer_head, dim=0)
            bs_node_layer_head.append(node_tensor)

        bs_node_tensor = torch.mean(torch.stack(bs_node_layer_head, dim=0), dim=0)
        imshow(bs_node_tensor - threshold, filename="batch_Node",
                plot_dir=cfg.train.saved_res_dir + dir_name + '/', xaxis="Head", yaxis="Layer",
                title="threshold:" + str(threshold))

        results = {}
        for layer_idx in range(bs_node_tensor.shape[0]):
            for head_idx in range(bs_node_tensor.shape[1]):
                key = f"attn_{layer_idx}.{head_idx}"
                results[key] = bs_node_tensor[layer_idx, head_idx].item()

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        print("results:", sorted_results)

    print("Time for runing the code (min): ", (time.time() - stime) / 60)

if __name__ == '__main__':
    main()