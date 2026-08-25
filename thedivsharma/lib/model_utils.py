"""Model loading, LoRA fine-tune step, and batched closed-book QA generation."""
from __future__ import annotations

import os

# Must be set before torch touches the MPS backend. Without a ceiling, MPS's
# unified-memory allocator will keep growing until the whole machine (not
# just this process) stops responding -- that's what happened during the
# first smoke test with a fp32 model. This forces a clean OOM in this one
# subprocess instead, well below the point where the OS itself would choke.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import gc
import time
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lora import LoRALinear, capture_activations, inject_lora, trainable_parameters

TARGET_SUFFIXES = ("self_attn.q_proj", "self_attn.v_proj")

# Matches general-knowledge/src/utils.py::SQUAD_ANSWER_TEMPLATE_BASE exactly.
# Using a base (non-instruct) model with this raw completion-style prompt --
# not an -Instruct model with a chat template -- because training self-edits
# happen as raw "Title\nFact." continuations; an instruct model's LoRA update
# optimized for that format corrupted its separate chat-formatted behavior in
# testing (fluent training loss, gibberish at chat-prompted generation). A
# base model has no chat behavior to protect, so train/eval distributions
# match, which is also what the paper's own general-knowledge setup does.
ANSWER_TEMPLATE_BASE = "Let's answer a question directly and concisely.\nQuestion: {question}\nAnswer:\n"


def free_memory(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # NOTE: padding_side is set per-call below (right for training/activation
    # capture, left for generation) -- it is deliberately NOT fixed here.
    # Training with left-padding shifted real content to high position
    # indices, corrupting Qwen's rotary embeddings during backprop (this
    # produced fluent-looking gibberish after training in testing); HF's
    # generate() is the one path that correctly derives position_ids from
    # attention_mask for left-padded batches, so left-padding belongs there
    # only.
    tokenizer.padding_side = "right"
    # bfloat16, not float32: this machine has 16GB total RAM shared with the
    # OS/VS Code, and a 1.5B model in fp32 (~6GB weights alone) was enough to
    # make the whole machine stop responding during the first smoke test.
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    # LoRA only trains a handful of small matrices, but backward still needs to
    # route gradients through all 28 frozen layers to reach them -- without
    # checkpointing, activation memory for that is comparable to full
    # fine-tuning and is what actually exhausted MPS memory in testing (not
    # batch size). Checkpointing recomputes activations during backward
    # instead of storing them, trading ~30% more compute for the memory this
    # machine doesn't have to spare.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model, tokenizer


def setup_lora(model, r: int, alpha: float) -> Dict[str, LoRALinear]:
    return inject_lora(model, TARGET_SUFFIXES, r=r, alpha=alpha)


@torch.no_grad()
def _reset_all(adapters: Dict[str, LoRALinear]) -> None:
    for ad in adapters.values():
        ad.reset()


def train_lora_step(
    model,
    tokenizer,
    adapters: Dict[str, LoRALinear],
    device: torch.device,
    texts: List[str],
    epochs: int,
    lr: float,
    max_length: int = 256,
) -> float:
    """Reset LoRA to fresh A/B, then SFT (causal LM loss) on `texts`.

    One example per forward/backward pass (batch_size=1), matching
    general-knowledge/src/inner/TTT_server.py's own default -- Qwen2.5's
    ~152k-token vocab makes a batched forward's [batch, seq, vocab] logits
    tensor the dominant memory cost, not the model weights. Batching
    several sequences together is what triggered the MPS OOM in testing;
    per-example steps keep peak memory bounded regardless of how many
    sequences a self-edit expands into.
    """
    tokenizer.padding_side = "right"  # real content must start at position 0 for training
    _reset_all(adapters)
    params = trainable_parameters(adapters)
    for p in params:
        p.requires_grad_(True)

    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    last_loss = 0.0
    for _ in range(epochs):
        for text in texts:
            # Fixed-length padding (not just truncation): per-example steps with
            # *varying* shapes fragmented the MPS allocator badly in testing --
            # each differently-sized example forced a new block, and freed
            # blocks of the "wrong" size couldn't be reused, so allocated
            # memory crept up over a training call even with explicit
            # del/gc/empty_cache each iteration. Identical shapes every
            # iteration let the allocator actually reuse the same blocks.
            enc = tokenizer(
                text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt"
            ).to(device)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            opt.zero_grad()
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
            out.loss.backward()
            # HF's Trainer (used by the real TTT_server.py) clips grad norm to
            # 1.0 by default; this hand-rolled loop didn't, and batch_size=1
            # training on a 0.5B model produced fluent-looking gibberish after
            # only ~28 steps without it -- an occasional large per-example
            # gradient with no averaging to smooth it out.
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
            last_loss = out.loss.item()
            del out, enc, labels
            # Per-example steps mean many small alloc/free cycles; observed in
            # testing that MPS memory climbed monotonically across iterations
            # of a single train_lora_step call even with del + empty_cache()
            # alone, which points at an autograd reference cycle that plain
            # refcounting doesn't clear -- gc.collect() breaks it.
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    model.eval()
    for p in params:
        p.requires_grad_(False)
    free_memory()
    return last_loss


@torch.no_grad()
def batched_generate(
    model, tokenizer, device: torch.device, prompts: List[str], max_new_tokens: int = 32, batch_size: int = 8
) -> List[str]:
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # required for correct batched causal generation
    outs: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding="max_length", truncation=True, max_length=128
        ).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        del enc, gen, new_tokens
        free_memory()
        outs.extend(t.strip() for t in texts)
    tokenizer.padding_side = original_padding_side
    return outs


def answer_questions(model, tokenizer, device, questions: List[Dict[str, Any]], batch_size: int = 8) -> List[str]:
    prompts = [ANSWER_TEMPLATE_BASE.format(question=q["question"]) for q in questions]
    return batched_generate(model, tokenizer, device, prompts, max_new_tokens=24, batch_size=batch_size)


@torch.no_grad()
def collect_context_activations(
    model, tokenizer, device, adapters: Dict[str, LoRALinear], title: str, context: str, max_length: int = 512
) -> Dict[str, torch.Tensor]:
    tokenizer.padding_side = "right"  # keep content at position 0, consistent with training
    text = f"{title}\n{context}"
    enc = tokenizer(
        [text], truncation=True, max_length=max_length, padding="max_length", return_tensors="pt"
    ).to(device)
    return capture_activations(adapters, model, enc["input_ids"], enc["attention_mask"])
