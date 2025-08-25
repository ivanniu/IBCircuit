from typing import ClassVar, Optional
from transformer_lens import ActivationCache
import torch
import torch.nn.functional as F
import numpy as np
hook_filter = lambda name: name.endswith("ln1.hook_normalized") or name.endswith("attn.hook_result")
def get_greatthan_3_caches(model, clean_input, corrupted_input, mode):
    model.reset_hooks()
    clean_cache = {}

    def forward_cache_hook(act, hook):
        clean_cache[hook.name] = act.detach()

    edge_acdcpp_outgoing_filter = lambda name: name.endswith(("hook_result", "hook_mlp_out", "blocks.0.hook_resid_pre", "hook_q", "hook_k", "hook_v"))
    model.add_hook(hook_filter if mode == "node" else edge_acdcpp_outgoing_filter, forward_cache_hook, "fwd")

    clean_grad_cache = {}

    def backward_cache_hook(act, hook):
        clean_grad_cache[hook.name] = act.detach()

    incoming_ends = ["hook_q_input", "hook_k_input", "hook_v_input", f"blocks.{model.cfg.n_layers-1}.hook_resid_post"]
    if not model.cfg.attn_only:
        incoming_ends.append("hook_mlp_in")
    edge_acdcpp_back_filter = lambda name: name.endswith(tuple(incoming_ends + ["hook_q", "hook_k", "hook_v"]))
    model.add_hook(hook_filter if mode=="node" else edge_acdcpp_back_filter, backward_cache_hook, "bwd")
    model=model.to(clean_input.device)
    value = greaterthan_metric(model(clean_input), clean_input, model)
    value.backward()

    # cache the activations of the corrupted inputs
    model.reset_hooks()
    corrupted_cache = {}

    def forward_corrupted_cache_hook(act, hook):
        corrupted_cache[hook.name] = act.detach()

    model.add_hook(hook_filter if mode == "node" else edge_acdcpp_outgoing_filter, forward_corrupted_cache_hook, "fwd")
    model(corrupted_input)
    model.reset_hooks()

    clean_cache = ActivationCache(clean_cache, model)
    corrupted_cache = ActivationCache(corrupted_cache, model)
    clean_grad_cache = ActivationCache(clean_grad_cache, model)
    return clean_cache, corrupted_cache, clean_grad_cache

NOUNS = [
    "abduction", "accord", "affair", "agreement", "appraisal",
    "assaults", "assessment", "attack", "attempts", "campaign",
    "captivity", "case", "challenge", "chaos", "clash",
    "collaboration", "coma", "competition", "confrontation", "consequence",
    "conspiracy", "construction", "consultation", "contact",
    "contract", "convention", "cooperation", "custody", "deal",
    "decline", "decrease", "demonstrations", "development", "disagreement",
    "disorder", "dispute", "domination", "dynasty", "effect",
    "effort", "employment", "endeavor", "engagement",
    "epidemic", "evaluation", "exchange", "existence", "expansion",
    "expedition", "experiments", "fall", "fame", "flights",
    "friendship", "growth", "hardship", "hostility", "illness",
    "impact", "imprisonment", "improvement", "incarceration",
    "increase", "insurgency", "invasion", "investigation", "journey",
    "kingdom", "marriage", "modernization", "negotiation",
    "notoriety", "obstruction", "operation", "order", "outbreak",
    "outcome", "overhaul", "patrols", "pilgrimage", "plague",
    "plan", "practice", "process", "program", "progress",
    "project", "pursuit", "quest", "raids", "reforms",
    "reign", "relationship",
    "retaliation", "riot", "rise", "rivalry", "romance",
    "rule", "sanctions", "shift", "siege", "slump",
    "stature", "stint", "strikes", "study",
    "test", "testing", "tests", "therapy", "tour",
    "tradition", "treaty", "trial", "trip", "unemployment",
    "voyage", "warfare", "work",
]

class GreaterThanConstants:
    YEARS: list[str]
    YEARS_BY_CENTURY: dict[str, list[str]]
    TOKENS: list[int]
    INV_TOKENS: dict[int, int]
    TOKENS_TENSOR: torch.Tensor
    INV_TOKENS_TENSOR: torch.Tensor

    _instance: ClassVar[Optional["GreaterThanConstants"]] = None

    @classmethod
    def get(cls: type["GreaterThanConstants"], tokenizer, device) -> "GreaterThanConstants":
        if cls._instance is None:
            cls._instance = cls(tokenizer, device)
        return cls._instance

    def __init__(self, tokenizer, device):
        _TOKENIZER = tokenizer

        self.YEARS = []
        self.YEARS_BY_CENTURY = {}

        for century in range(11, 18):
            all_success = []
            for year in range(century * 100 + 2, (century * 100) + 99):
                a = _TOKENIZER.encode(f" {year}")
                if a == [_TOKENIZER.encode(f" {str(year)[:2]}")[0], _TOKENIZER.encode(str(year)[2:])[0]]:
                    all_success.append(str(year))
                    continue
            self.YEARS.extend(all_success[1:-1])
            self.YEARS_BY_CENTURY[century] = all_success[1:-1]

        TOKENS = {
            i: _TOKENIZER.encode(f"{'0' if i<=9 else ''}{i}")[0] for i in range(0, 100)
        }
        self.INV_TOKENS = {v: k for k, v in TOKENS.items()}
        self.TOKENS = TOKENS

        TOKENS_TENSOR = torch.as_tensor([TOKENS[i] for i in range(0, 100)], dtype=torch.long)
        INV_TOKENS_TENSOR = torch.zeros(50290, dtype=torch.long)
        for i, v in enumerate(TOKENS_TENSOR):
            INV_TOKENS_TENSOR[v] = i

        self.TOKENS_TENSOR = TOKENS_TENSOR.to(device)
        self.INV_TOKENS_TENSOR = INV_TOKENS_TENSOR.to(device)

def greaterthan_metric(logits, tokens, model, return_one_element: bool=True):
    constants = GreaterThanConstants.get(model)

    probs = F.softmax(logits[:, -1], dim=-1)
    csum = torch.cumsum(probs[:, constants.TOKENS_TENSOR], dim=-1)
    yearend = constants.INV_TOKENS_TENSOR[tokens[:, 7]].to(logits.device)

    # At or after: positive term
    range = torch.arange(len(yearend))
    positive = csum[:, -1]
    # Before: negative term
    negative = torch.where(yearend == 0, torch.zeros((), device=csum.device), csum[range, yearend])
    if return_one_element:
        return - (positive - 2*negative).mean()
    else:
        return - (positive - 2*negative)


def get_year_data(num_examples, tokenizer, device):
    constants = GreaterThanConstants.get(tokenizer, device)
    template = "The {noun} lasted from the year {year1} to "

    # set some random seed
    torch.random.manual_seed(54)
    nouns_perm = torch.randint(0, len(NOUNS), (num_examples,))
    years_perm = torch.randint(0, len(constants.YEARS), (num_examples,))

    prompts = []
    prompts_tokenized = []
    end_idxs = []
    for i in range(num_examples):
        year = constants.YEARS[years_perm[i]]
        prompts.append(
            template.format(
                noun=NOUNS[nouns_perm[i]],
                year1=year,
            ) + year[:2]
        )
        prompts_tokenized.append(tokenizer.encode(prompts[-1], return_tensors="pt").to(device))
        end_idxs.append(len(prompts_tokenized[-1][0])-1)
        assert prompts_tokenized[-1].shape == prompts_tokenized[0].shape, (prompts_tokenized[-1].shape, prompts_tokenized[0].shape)
    prompts_tokenized = torch.cat(prompts_tokenized, dim=0)
    assert len(prompts_tokenized.shape) == 2, prompts_tokenized.shape
    return prompts_tokenized, end_idxs, np.max(end_idxs)+1

def get_greaterthan_data(num_examples, model):
    data, prompts = get_year_data(num_examples, model)
    patch_data = data.clone()
    patch_data[:, 7] = 486
    return data, patch_data