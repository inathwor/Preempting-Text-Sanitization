from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, AutoModelForCausalLM
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import os
from os.path import join
import pandas as pd
import re
import numpy as np
from secrets import randbits
from cupyx.scipy.spatial import distance
import sys
# Add the main directory to sys.path to be able to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR, HUGGINGFACE_TOKEN

# PARAMS
default_save_path = join(ROOT_DIR, "datasets/multi_news/llama_generated_summaries/OnOriginalTexts/")
texts_per_slice = 10 #Process X texts at the same time
# END PARAMS
if not HUGGINGFACE_TOKEN:
    print("Please request access to llama3 model on HuggingFace and paste your account's access token in the config.py file.")
    sys.exit()

cuda_device = "cuda"
torch.set_default_device(cuda_device)

if not os.path.exists(default_save_path):
    os.makedirs(default_save_path)

texts = load_from_disk(join(ROOT_DIR, "datasets/multi_news/concatenated_clean_1024_tokens"))['document']

llama3_hf_name = "meta-llama/Meta-Llama-3-8B-Instruct"
llama_tokenizer = AutoTokenizer.from_pretrained(
    llama3_hf_name,
    padding_side="left",
    revision = "e5e23bbe8e749ef0efcf16cad411a7d23bd23298",
    token=HUGGINGFACE_TOKEN)
llama_model = AutoModelForCausalLM.from_pretrained(
    llama3_hf_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    revision = "e5e23bbe8e749ef0efcf16cad411a7d23bd23298",
    token=HUGGINGFACE_TOKEN
)

# Config tokenizer according to Llama example
llama_tokenizer.pad_token = llama_tokenizer.eos_token
terminators = [
    llama_tokenizer.eos_token_id,
    llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

i_max = len(texts)
filenumber = 0
for i in range(0, i_max, texts_per_slice):
    torch.cuda.empty_cache()
    if i%500 == 0:
        print(str(i), " processed.")

    j = min(i+texts_per_slice, i_max)
    texts_slice = texts[i:j]
    
    # Tokenize
    messages = [
        [
            {"role": "system", "content": "Your task is to summarize your input text."},
            {"role": "user", "content": text}
        ]
        for text in texts_slice
    ]
    
    input_ids = llama_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        return_tensors="pt",
        return_dict=True,
    ).to(cuda_device)

    with torch.no_grad():
        outputs = llama_model.generate(
            **input_ids,
            max_new_tokens=142,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        outputs = outputs[:, input_ids["input_ids"].shape[-1]:]
        generated_summaries = llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Save summaries to disk
    filepath = default_save_path+f"file_{filenumber:04d}"+".pickle"
    with open(filepath, 'wb') as f:
        pickle.dump(generated_summaries, f)
    filenumber += 1

generated_summaries = []
for file in sorted(os.listdir(default_save_path)):
    with open(default_save_path+file, 'rb') as f:
        generated_summaries += pickle.load(f)

with open(default_save_path+"full.pickle", 'wb') as f:
    pickle.dump(generated_summaries, f)