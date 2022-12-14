from typing import List, Dict
from transformers import FlaxAutoModelForCausalLM, GPT2TokenizerFast
from datasets import load_dataset
import dcargs
import optax
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import time
import jax
from flax.training import train_state
from flax.training.common_utils import onehot
import jax.numpy as jnp

# Extend train state by keeping dropout rng in state
class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    # TODO: idk what this is for yet, multi-gpu?
#    def replicate(self):
#        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))

def main(
    batch_size: int = 16,
    max_length: int = 256,
):  
    seed_val = 0
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    rng = jax.random.PRNGKey(seed_val)
    rng, dropout_rng = jax.random.split(rng)

    model = FlaxAutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    def tokenize(batch):
        return tokenizer(
            batch["text"],
        )

    # Initialize language modeling dataset.
    dataset = load_dataset("wikitext", "wikitext-2-v1")
    dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=4,
        # Remove "text" column not required for training
        remove_columns=dataset["train"].column_names,
    )

    # Chunk the dataset, dropping remainders.
    def chunk_text(examples):
        keys = ["input_ids", "attention_mask"]
        concat_examples = {k: sum(examples[k], []) for k in keys}
        total_length = len(concat_examples["input_ids"])
        total_length = (total_length // max_length) * max_length
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concat_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    dataset = dataset.map(
        chunk_text,
        batched=True,
    )

    # Setup dataloaders.
    # Setting format only changes the __getitem__ return type, we also have to
    #  set the collate function that merges samples into a minibatch.
    dataset.set_format(type="numpy", columns=["input_ids", "attention_mask", "labels"])
    def numpy_collate_fn(samples: List[Dict[str, np.ndarray]]) -> np.ndarray:
        return {k: np.stack([s[k] for s in samples]) for k in samples[0].keys()}
        
    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=numpy_collate_fn,
    )

    val_dataloader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=numpy_collate_fn,
    )

    # Initiaize optimizer.
    optimizer = optax.adamw(
        learning_rate=1e-4,
    )
    optimizer.init(model.params)
    
    # Initialize train state.
    train_state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=optimizer,
        dropout_rng=dropout_rng,
    )
    
    # This could also be calculated within the HF forward pass (with a `labels` arg) but they don't implement it
    # TODO: labels is currently actually a np.ndarray, not a jnp.ndarray
    def loss_fn(logits: jnp.ndarray, labels: jnp.ndarray):
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        loss = optax.softmax_cross_entropy(shift_logits, onehot(shift_labels, shift_logits.shape[-1]))
        return loss.mean()

    # Functional style: define how to update state in a single step.
    # TODO: add types (what's torch / jnp.ndarray)
    @jax.jit
    def train_step(state, batch):
        # Note: you need to split the dropout rng for each step, otherwise we'll drop out the same units every time!
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)
        # Define a function that goes params -> loss that we'll differentiate.
        def compute_loss(params):
            # Remove labels since HF Flax model doesn't use them.
            labels = batch.pop("labels")
            # Call the model with the `apply_fn` kept by TrainState.
            outputs = state.apply_fn(**batch, params=params, train=True, dropout_rng=dropout_rng)
            return loss_fn(outputs.logits, labels)

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def eval_step(state, batch):
        labels = batch.pop("labels")
        outputs = state.apply_fn(**batch, params=state.params, train=False)
        loss = loss_fn(outputs.logits, labels)
        return loss

    # Train loop.
    total_loss = 0
    best_val = 100
    start = time.time()
    for i, batch in enumerate(train_dataloader):
        train_state, loss = train_step(train_state, batch)
        if i % 100 == 0:
            val_loss = 0
            for batch in val_dataloader:
                # TODO: should I be accumulating the loss here?
                val_loss += eval_step(train_state, batch).item()
            val_loss /= len(val_dataloader)
            if val_loss < best_val:
                best_val = val_loss
            if best_val < .25 * val_loss:
                print("Early stopping.")
                break
            print(f"step {i} | elapsed {(time.time() - start) / 60:.2f}m | loss: {loss.item():.2f} | val: {val_loss:.2f}")
            
    print("Done.")

if __name__ == "__main__":
    dcargs.cli(main)
