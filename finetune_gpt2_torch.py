from transformers import AutoModelForCausalLM, GPT2TokenizerFast, AdamW
from datasets import load_dataset
import dcargs
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import time

def main(
    batch_size: int = 8,
    max_length: int = 256,
):
    # Set the seed value all over the place to make this reproducible.
    seed_val = 0

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    def tokenize(batch):
        return tokenizer(
            batch["text"]
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
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
    )

    val_dataloader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
    )

    # Default params from HF TrainingArguments
    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,
    )
    
    # Train loop.
    model.cuda().train()
    total_loss = 0
    best_val = 100
    start = time.time()
    for i, batch in enumerate(train_dataloader):
        model.train()
        input_ids = batch["input_ids"].to("cuda")
        labels = batch["labels"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")

        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        model.zero_grad()

        if i % 100 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch["input_ids"].to("cuda")
                    labels = batch["labels"].to("cuda")
                    attention_mask = batch["attention_mask"].to("cuda")
                    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
                    val_loss += outputs.loss.item()
            val_loss /= len(val_dataloader)
            if val_loss < best_val:
                best_val = val_loss
            if best_val < .25 * val_loss:
                print("Early stopping.")
                break
            
            print(f"step {i} | elapsed {(time.time() - start) / 60:.2f}m | loss: {loss.item():.2f} | val: {val_loss:.2f}")

if __name__ == "__main__":
    dcargs.cli(main)
