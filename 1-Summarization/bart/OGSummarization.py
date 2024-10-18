from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datetime import datetime
import pickle
import torch
from os.path import join
import os
import sys
# Add the main directory to sys.path to be able to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR

# PARAMS
cuda_device = "cuda"
texts_per_slice = 25 #Process X texts at the same time
# END PARAMS

default_save_path = join(ROOT_DIR, "datasets/multi_news/bart_generated_summaries/OnOriginalTexts/")
torch.set_default_device(cuda_device)

if not os.path.exists(default_save_path):
    os.makedirs(default_save_path)

mnews = (load_from_disk(join(ROOT_DIR, "datasets/multi_news/concatenated_clean_1024_tokens")))['document']

bart_tokenizer = AutoTokenizer.from_pretrained(
    "facebook/bart-large-cnn",
    device=cuda_device,
    torch_dtype="auto",
    use_fast=False,
    revision= "37f520fa929c961707657b28798b30c003dd100b"
)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/bart-large-cnn",
    torch_dtype="auto",
    revision= "37f520fa929c961707657b28798b30c003dd100b"
).to(cuda_device)
bart_model.eval(); #Ensure the model is in eval mode

n = len(mnews)
filenumber = 0

for i in range(0, n, texts_per_slice):
    if i%500 == 0:
        print(str(i), " processed.")
    j = min(i+texts_per_slice, n)
    texts = mnews[i:j]
    
    # Tokenize
    encoded_input = bart_tokenizer(texts, padding=True, truncation=False, return_tensors="pt").to(cuda_device)
    with torch.no_grad():
        encoded_outputs = bart_model.generate(**encoded_input)
        generated_summaries = bart_tokenizer.batch_decode(encoded_outputs, skip_special_tokens=True)
    
    # Save summaries to disk
    filepath = default_save_path+f"file_{filenumber:04d}"+".pickle"
    with open(filepath, 'wb') as f:
        pickle.dump(generated_summaries, f)
    filenumber += 1

# Load all summaries' files and merge into one file
generated_summaries = []
for file in sorted(os.listdir(default_save_path)):
    with open(default_save_path+file, 'rb') as f:
        generated_summaries += pickle.load(f)

with open(default_save_path+"full.pickle", 'wb') as f:
    pickle.dump(generated_summaries, f)