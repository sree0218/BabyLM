
#!pip install git+https://github.com/huggingface/transformers.git
#!pip install datasets
#!pip install transformers torch
#!pip install accelerate
#!apt install git-lfs
# from huggingface_hub import notebook_login

# notebook_login()
# scp vchilaka@nlp-gpu-01.soe.ucsc.edu:/soe/vchilaka
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.cuda.empty_cache()

#from typing import Dict, Tuple
from datasets import load_dataset, DatasetDict
from collections import Counter
from typing import List, Dict, Union, Callable, Any
#import matplotlib.pyplot as plt
#import pandas as pd
#import seaborn as sns
from pprint import pprint
import torch
import torch.nn as nn
from src.models.ssm_seq import SSMLMHeadModel
#from src.models.sequence.long_conv_lm import ConvLMHeadModel
import transformers.modeling_outputs as CausalLMOutput

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# dataset: DatasetDict = load_dataset("Sree1994/babylm_childstories")

ds_train = load_dataset("Sree1994/blm_strict_small", split="train")
ds_valid = load_dataset("Sree1994/blm_strict_small", split="valid")

raw_datasets = DatasetDict(
    {
        "train": ds_train,
        "valid": ds_valid
    }
)

raw_datasets

from transformers import RobertaTokenizer

context_length = 128
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
vocab_size = tokenizer.vocab_size

outputs = tokenizer(
    raw_datasets["train"]["text"],
    truncation=True,
    max_length=context_length,
    return_overflowing_tokens=True,
    return_length=True,
    pad_to_max_length=True,
)

print(f"Input IDs length: {len(outputs['input_ids'])}")
print(f"Input chunk lengths: {(outputs['length'])}")
print(f"Chunk mapping: {outputs['attention_mask']}")

def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length <= context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)
tokenized_datasets

from transformers import AutoTokenizer, RobertaForCausalLM, AutoConfig
import torch

#_____________________________________________________________________________________________________________________________
class MyTrainer(Trainer):  
    def compute_loss(self, model, inputs, return_outputs=False):
        model_outputs = super().compute_loss(model, inputs, return_outputs)
        if type(model_outputs) == tuple:
            model_outputs = model_outputs[0]
        return (model_outputs['loss'],model_outputs) if return_outputs else model_outputs['loss']


class MySSMLMHeadModel(SSMLMHeadModel):
    def forward(self, input_ids, position_ids=None,inferece_params = None,state =None, labels = None, **kwargs):
        output,_ = super().forward(input_ids, position_ids,inferece_params,state)
        lm_logits = output.logits
        loss = None
        if labels is not None:
            shifted_prediction_scores = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            model_implied_vocab_size : int = shifted_prediction_scores.shape[-1]
            loss = loss_fct(shifted_prediction_scores.view(-1, model_implied_vocab_size), labels.view(-1))
        return CausalLMOutput(loss=loss, logits=lm_logits),None


# Instantiate and initialize weights
ssm_cfg = dict(mode='diag', measure='diag-lin')
attn_cfg = dict(num_heads=12, use_flash_attn=True, fused_bias_fc=True, dropout=0.1)
d_model: int = 768  # hidden state size
model: MySSMLMHeadModel = MySSMLMHeadModel(
        d_model=d_model,
        n_layer=12,
        d_inner=4 * d_model,
        vocab_size=len(tokenizer),
        ssm_cfg=ssm_cfg,
        attn_layer_idx=[1, 8],
        attn_cfg=attn_cfg,
        pad_vocab_size_multiple=8,
        layer={"_name_": "h3"}
)
model_size: int = sum(t.numel() for t in model.parameters())
print(f"H3 size: {model_size / 1000 ** 2:.1f}M parameters")


#_____________________________________________________________________________________________________________________________


from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")


from transformers import Trainer, TrainingArguments


args = TrainingArguments(
    output_dir="./h3",
    overwrite_output_dir=True,
    evaluation_strategy = 'epoch',    
    do_train=True,
    do_eval=True,
    do_predict=True,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    # evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=20,
    weight_decay=0.01,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=1000,
    fp16=True,
    push_to_hub=False,
    save_total_limit=1,
)

trainer = MyTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    # compute_metrics=my_compute_metrics,
)

trainer.train()


val = trainer.evaluate(metric_key_prefix="test", eval_dataset=tokenized_datasets["valid"])
valid_loss = val.get("test_loss")
# print(f"Training Loss: {trn.training_loss}")
print(f"Validation Loss: {valid_loss}")
print(f"Validation Perplexity: {torch.exp(torch.tensor(valid_loss))}")

print(f"Best Validation Perplexity: {torch.exp(torch.tensor(5.40))}")





