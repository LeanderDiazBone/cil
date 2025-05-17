# Common imports
import os
import optax
import treescope
import functools
import json
import os
import pickle
import pandas as pd
import wandb
from utils import  create_parser

# Gemma imports
from kauldron import kd
from gemma import gm

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"
def main(args):
    training = pd.read_csv("data/training.csv")
    tokenizer = gm.text.Gemma3Tokenizer()

    #tokenizer.encode('This is an example sentence', add_bos=True)
    training["promts"] = "Determine if the following sentence has positive, neutral or negative sentiment: " + training["sentence"]
    input_tokens = tokenizer.encode(list(training["promts"][:100]), add_bos=True)
    #training["tokens"] = tokenizer.encode(training["sentences"])
    #training.to_csv("data/training_2.csvf")
    model = gm.nn.Gemma3_4B(
        tokens="batch.input",
    )
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)
    output_tokens = model.apply(params, input_tokens)
    print(output_tokens)
    #sampler = gm.text.ChatSampler(model=model,params=params,multi_turn=False)
    #prompt = "Tell me a story"
    #out0 = sampler.chat(prompt)
    #print(out0)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    print("Arguments:")
    print(
        json.dumps(
            vars(args), sort_keys=True, indent=4
        )
    )

    wandb.init(
        project=args.project_name,
        group=args.group_name,
        name=args.exp_name,
        config=vars(args),
        #mode="online" if args.log_wandb else "disabled",
    )

    #with Profiler(interval=0.1) as profiler:
    main(args)
    #profiler.print()
    #profiler.open_in_browser()