"""
FineWeb-Edu dataset that outputs in the same format as OpenWebText
"""

import os
import numpy as np
import tiktoken
from tqdm import tqdm
from datasets import load_dataset # huggingface datasets
import random

num_proc = 8
num_proc_load_dataset = num_proc

# Get the tokenizer
enc = tiktoken.get_encoding("gpt2")

# Create output directory
DATA_DIR = os.path.dirname(__file__)

# export HF_HOME="/data_hdd/users/sichongjie_sub3/cache/hf"
# export HF_ENDPOINT=https://hf-mirror.com 
train_dataset = load_dataset('HuggingFaceFW/fineweb', split='train')
# For validation, either load with split='validation' if available, or create a subset:
validation_size = int(len(train_dataset) * 0.0005)
indices = list(range(len(train_dataset)))
random.seed(2357)
random.shuffle(indices)
train_indices = indices[validation_size:]
val_indices = indices[:validation_size]

train_subset = train_dataset.select(train_indices)
val_subset = train_dataset.select(val_indices)
split_dataset = {'train': train_subset, 'val': val_subset}

# Define the tokenization function (matching OpenWebText EOT appending)
def process(example):
    ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores special tokens
    ids.append(enc.eot_token)  # append EOT token like OpenWebText
    out = {'ids': ids, 'len': len(ids)}
    return out

# Tokenize the dataset
tokenized = {}
for split_name, split_data in split_dataset.items():
    processed_data = {'ids': [], 'len': []}
    for example in tqdm(split_data, desc=f"tokenizing {split_name}"):
        result = process(example)
        processed_data['ids'].append(result['ids'])
        processed_data['len'].append(result['len'])
    tokenized[split_name] = processed_data

# Save as binary files like OpenWebText
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(DATA_DIR, f'{split}.bin')
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    
    total_batches = 1024
    idx = 0
    batch_size = len(dset['ids']) // total_batches + 1
    
    for batch_start in tqdm(range(0, len(dset['ids']), batch_size), desc=f'writing {filename}'):
        batch_end = min(batch_start + batch_size, len(dset['ids']))
        batch_ids = dset['ids'][batch_start:batch_end]
        # Concatenate all token IDs in this batch
        arr_batch = np.concatenate(batch_ids)
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()

print("Done! The data can be loaded with:")
print("train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')")
print("val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')")