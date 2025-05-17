# Common imports
import os
import optax
import treescope
import functools
import json
import os
import pickle
import wandb
from utils import  create_parser

# Gemma imports
from kauldron import kd
from gemma import gm

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"
def main(args):
    tokenizer = gm.text.Gemma3Tokenizer()

    tokenizer.encode('This is an example sentence', add_bos=True)

    ds = kd.data.py.Tfds(
        name='mtnt/en-fr',
        split='train',
        shuffle=True,
        batch_size=8,
        transforms=[
            # Create the model inputs/targets/loss_mask.
            gm.data.Seq2SeqTask(
                # Select which field from the dataset to use.
                # https://www.tensorflow.org/datasets/catalog/mtnt
                in_prompt='src',
                in_response='dst',
                # Output batch is {'input': ..., 'target': ..., 'loss_mask': ...}
                out_input='input',
                out_target='target',
                out_target_mask='loss_mask',
                tokenizer=tokenizer,
                # Padding parameters
                max_length=200,
                truncate=True,
            ),
        ],
    )

    ex = ds[0]

    treescope.show(ex)

    model = gm.nn.Gemma3_4B(
        tokens="batch.input",
    )

    loss = kd.losses.SoftmaxCrossEntropyWithIntLabels(
        logits="preds.logits",
        labels="batch.target",
        mask="batch.loss_mask",
    )

    trainer = kd.train.Trainer(
        seed=42,  # The seed of enlightenment
        workdir='/tmp/ckpts',  # TODO(epot): Make the workdir optional by default
        # Dataset
        train_ds=ds,
        # Model
        model=model,
        init_transform=gm.ckpts.LoadCheckpoint(  # Load the weights from the pretrained checkpoint
            path=gm.ckpts.CheckpointPath.GEMMA3_4B_IT,
        ),
        # Training parameters
        num_train_steps=300,
        train_losses={"loss": loss},
        optimizer=optax.adafactor(learning_rate=1e-3),
    )

    state, aux = trainer.train()

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