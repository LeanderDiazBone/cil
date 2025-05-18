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



import torch,time

def generate_prompt(num_few_shot, train_df, sentence):
    g = torch.Generator().manual_seed(time.time_ns() & ((1<<63)-1))
    indices = torch.randperm(len(train_df), generator=g)[:num_few_shot]
    examples = zip(
        train_df["sentence"].iloc[indices].tolist(),
        train_df["label"].iloc[indices].tolist()
    )
    prefix = (
        "You are a highly accurate sentiment classifier.\n\n"
        "Your task is to classify the input sentence as 'positive', 'negative', or 'neutral'. "
        "The input format is the following: After 'Input:' the sentence to classify follows. "
        "Then, after 'Output:' you should respond with a single word. Either 'positive', 'negative', or 'neutral', "
        + "\n".join(f"Input: {ex} Output: {lbl}" for ex, lbl in examples)
    )
    return f"{prefix}\nInput: {sentence}\nOutput:"



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
    # FEW_SHOT=20
    if GEN_TEST:
        output_path=f"test/output_{re.sub(r'[^0-9A-Za-z_-]','_',model_name)}_{FEW_SHOT}shot_{int(time.time()*1000)}.csv"
    else:
        output_path=f"val/output_{re.sub(r'[^0-9A-Za-z_-]','_',model_name)}_{FEW_SHOT}shot_{int(time.time()*1000)}.csv"
    if model_name in ["Qwen/Qwen3-8B","meta-llama/Llama-3.1-8B"]: #doens't matter
        BATCH_SIZE=32
    else:
        BATCH_SIZE=32
    print(model_name, FEW_SHOT)

    ####

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
  
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    
   
    # from transformers import BitsAndBytesConfig

    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     llm_int8_threshold=6.0,
    #     llm_int8_skip_modules=None,
    #     llm_int8_enable_fp32_cpu_offload=True,
    # )

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     quantization_config=bnb_config,
    #     device_map="auto"
    # )
 
    model.model_name = model_name
 
    #model = torch.compile(model) 
    
    train_df_raw = pd.read_csv("data/training.csv").sample(frac=1, random_state=42)
    train_df = train_df_raw.head(200).reset_index(drop=True)
    if GEN_TEST:
        test_df = pd.read_csv("data/test.csv")
    else:
        test_df = train_df_raw.tail(500).reset_index(drop=True)
    print(f"Loaded {len(test_df)} rows from test.csv")

   
    # 48.14 neutr, 30.40 pos, 21.46 neg
 
    
    results_df = infer_sentiment_probs(train_df, test_df, model, tokenizer, num_few_shot=FEW_SHOT, batch_size=BATCH_SIZE)

    results_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
