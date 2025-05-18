#!/usr/bin/env python
"""
Cleaned sentiment‑classification script converted from a Jupyter notebook.
Functionality is unchanged; imports are consolidated, helpers are extracted,
and the file is now runnable with ``python sentiment_classifier.py``.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# prefix = (
#         "You are an expert sentiment analyst for a business owner. "
#         "Classify the sentiment of the customer review below. "
#         "Choose exactly one of these labels: Positive, Negative, Neutral. "
#         "Respond with only the single chosen label—no punctuation, no extra words."
#     )

# prefix = (
#         "Task: Identify the sentiment expressed toward the business in the given review. "
#         "Possible outputs (lower-case, one word): positive, negative, neutral.\n"
#         "Example → Review: \"The service was okay.\" Output: neutral\n"
#         "Now analyze the next review and output just the single sentiment word."
#     )

# prefix = (
#         "You are a reasoning model. Silently analyze the tone and intent of the review, "
#         "decide whether it is positive, negative, or neutral, and then output ONLY that "
#         "single word on its own line. Do not reveal your reasoning. If both positive and negative are equally likely, predict neutral."
#     )


# CUDA_VISIBLE_DEVICES=4,5 ./run_overnight.py

import torch,time

def generate_prompt(num_few_shot, train_df, sentence):
    g = torch.Generator().manual_seed(time.time_ns() & ((1<<63)-1))
    indices = torch.randperm(len(train_df), generator=g)[:num_few_shot]
    examples = zip(
        train_df["sentence"].iloc[indices].tolist(),
        train_df["label"].iloc[indices].tolist()
    )
    # prefix = (
    #     "You are a highly accurate sentiment classifier.\n\n"
    #     "Your task is to classify the review as 'positive', 'negative', or 'neutral'. "
    #     "The reviews were collected from the internet, and were written by real persons."
    #     "The input format is the following: After 'Input:' the sentence to classify follows. "
    #     "Then, after 'Output:' you should respond with a single word. Either 'positive', 'negative', or 'neutral', "
    #     + "\n".join(f"Input: {ex} Output: {lbl}" for ex, lbl in examples)
    # )
    # return f"{prefix}\nInput: {sentence}\nOutput:"
    # prefix = (
    #     "You own a business and received reviews online. "
    #     "Your task is to classify the following review as 'positive', 'negative', or 'neutral'."
    #     "Your final output should be a single word describing the sentiment. If you don't know, predict neutral."
    # )
    # return f"{prefix}\n\nReview: \"{sentence}\"\n"

 
    prefix = (
        "You own a business and received reviews online. "
        "Your task is to classify the following review as 'positive', 'negative', or 'neutral'."
        "Your final output should be a single word describing the sentiment. If you don't know, predict neutral."
        "10 examples:"
        + "\n".join(f"Review: {ex} Output: {lbl}" for ex, lbl in examples)
    )
    
    return f"{prefix}\n\nReview: \"{sentence}\"\n"





import torch.nn.functional as F

import torch, pandas as pd, torch.nn.functional as F


@torch.no_grad()
def infer_sentiment_probs(
    train_df,
    test_df,
    model,
    tokenizer,
    num_few_shot: int,
    batch_size: int = 64,
):
    labels = ["positive", "negative", "neutral"]
    label_ids = {lbl: tokenizer(lbl, add_special_tokens=False).input_ids for lbl in labels}
    assert all(len(ids) == 1 for ids in label_ids.values()), "Each label must be one token"

    results = []

    for offset in range(0, len(test_df), batch_size):
        print(offset)
        batch = test_df.iloc[offset : offset + batch_size]

        for i, sentence in enumerate(batch["sentence"]):
            prompt = generate_prompt(num_few_shot, train_df, sentence)

            if any(k in model.model_name.lower() for k in ("it", "qwen")):
                prompt = tokenizer.apply_chat_template(
                    [[{"role": "user", "content": prompt}]],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

            prompt_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            logits = model(**prompt_inputs).logits[0, -1]
            log_probs = F.log_softmax(logits, dim=-1)

            scores = [log_probs[label_ids[lbl][0]].item() for lbl in labels]
            probs = F.softmax(torch.tensor(scores), dim=0).tolist()

            results.append(
                dict(
                    id = i + offset,
                    positive=probs[0],
                    negative=probs[1],
                    neutral=probs[2],
                    label=labels[int(torch.tensor(probs).argmax())],
                    sentence=sentence
                )
            )

    return pd.DataFrame(results)

import re, torch, pandas as pd

@torch.no_grad()
def infer_sentiment(train_df, test_df, model, tokenizer, num_few_shot, batch_size=8, max_new_tokens=1000):
    pat = re.compile(r"\b(positive|negative|neutral)\b", re.I)
    res = []
    from tqdm import tqdm

    for off in tqdm(range(0, len(test_df), batch_size)):
        batch = test_df.iloc[off : off + batch_size]
        prompts = []
        for s in batch["sentence"]:
            p = generate_prompt(num_few_shot, train_df, s)
            if any(k in model.model_name.lower() for k in ("it", "qwen")):
                p = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            prompts.append(p)
        tok = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(
            **tok, max_new_tokens=max_new_tokens, temperature=0.6, pad_token_id=tokenizer.eos_token_id
        )
        for i, (inp, out_ids, sent) in enumerate(zip(tok["input_ids"], gen, batch["sentence"])):
            txt = tokenizer.decode(out_ids[len(inp) :], skip_special_tokens=True).strip()
            matches = pat.findall(txt)
            label = matches[-1].lower() if matches else "neutral"
            res.append(dict(id=off + i, label=label, raw_output=txt, sentence=sent))
    return pd.DataFrame(res)




def main() -> None:
    
    SEED = 42
    np.random.seed(SEED)

    ####
    GEN_TEST = True
    #model_name = "Qwen/Qwen3-32B"  
    #model_name = "google/gemma-3-27b-it"
    #model_name = "unsloth/Llama-3.3-70B-Instruct"  

    import time, random, re
    model_names=["Qwen/Qwen3-8B","meta-llama/Llama-3.1-8B"]
    model_names = ["google/gemma-3-12b-it", "microsoft/phi-4", "Qwen/Qwen3-14B"] # , "Qwen/Qwen3-8B","meta-llama/Llama-3.1-8B"

    model_names = ["google/gemma-3-27b-it","Qwen/Qwen3-32B"] #, "google/gemma-3-27b-it"]
    shots=[10,15,20,25]
    model_name=random.choice(model_names)
    FEW_SHOT=random.choice(shots)
    # model_name="meta-llama/Llama-3.1-70B-Instruct"
    FEW_SHOT=10
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    if GEN_TEST:
        output_path=f"test/output_{re.sub(r'[^0-9A-Za-z_-]','_',model_name)}_{FEW_SHOT}shot_{int(time.time()*1000)}.csv"
    else:
        output_path=f"val/output_{re.sub(r'[^0-9A-Za-z_-]','_',model_name)}_{FEW_SHOT}shot_{int(time.time()*1000)}.csv"
    if model_name in ["Qwen/Qwen3-8B","meta-llama/Llama-3.1-8B"]: #doens't matter
        BATCH_SIZE=8
    else:
        BATCH_SIZE=16
    print(model_name, FEW_SHOT)

    ####

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
  
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    

 
    model.model_name = model_name
 
    #model = torch.compile(model) 
    
    train_df_raw = pd.read_csv("data/training.csv").sample(frac=1, random_state=42)
    train_df = train_df_raw.head(200).reset_index(drop=True)
    if GEN_TEST:
        test_df = pd.read_csv("data/test.csv")
    else:
        test_df = train_df_raw.tail(50).reset_index(drop=True)
    print(f"Loaded {len(test_df)} rows from test.csv")

    #print(prefix)
    # 48.14 neutr, 30.40 pos, 21.46 neg
    
    results_df = infer_sentiment(
    train_df,
    test_df,
    model,
    tokenizer,
    num_few_shot=FEW_SHOT,
    batch_size=BATCH_SIZE,
)
    
    #results_df = infer_sentiment_probs(train_df, test_df, model, tokenizer, num_few_shot=FEW_SHOT, batch_size=BATCH_SIZE)

    results_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
