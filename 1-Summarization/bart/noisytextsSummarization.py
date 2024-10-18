from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datetime import datetime
import pickle
import torch
import re
import os
from os.path import join
import sys
# Add the main directory to sys.path to be able to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR

# PARAMS
load_folderpath =  join(ROOT_DIR, "datasets/multi_news/noisy_texts/AsTensors/")
save_folder_path = join(ROOT_DIR, "datasets/multi_news/bart_generated_summaries/OnSanitizedTexts/")
epsilons = [1] + [i for i in range(5, 101, 5)]
texts_per_slice = 25 #Process X texts at the same time
cuda_device = "cuda"
# END PARAMS

torch.set_default_device(cuda_device)

with open(os.path.join(load_folderpath, "attention_mask.pickle"), 'rb') as f:
    attention_mask = pickle.load(f).to(cuda_device)

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

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

for epsilon in epsilons:
    print(datetime.now().strftime('%Hh%Mm%Ss'), "Epsilon=", str(epsilon))

    with open(os.path.join(load_folderpath, f"epsi{epsilon}full.pickle"), 'rb') as f:
        noisy_texts_ids = pickle.load(f).to(cuda_device)

    n = noisy_texts_ids.shape[0]

    filenames_prefix = "epsi"+str(epsilon)
    filenumber = 0
    for i in range(0, n, texts_per_slice):
        if i%500 == 0:
            print(str(i), " processed.")
        if i+texts_per_slice > n:
            j = n
        else:
            j = i+texts_per_slice
        noisy_texts_ids_slice = noisy_texts_ids[i:j]
        attention_mask_slice = attention_mask[i:j]
        
        with torch.no_grad():
            encoded_outputs = bart_model.generate(input_ids=noisy_texts_ids_slice, attention_mask=attention_mask_slice)
            generated_summaries = bart_tokenizer.batch_decode(encoded_outputs, skip_special_tokens=True)
        
        # Save summaries to disk
        filepath = save_folder_path+filenames_prefix+f"part_{filenumber:04d}"+".pickle"
        with open(filepath, 'wb') as f:
            pickle.dump(generated_summaries, f)
        filenumber += 1

    # Load all summaries' files and merge into one file
    generated_summaries = []
    for file in sorted(os.listdir(save_folder_path)):
        if re.fullmatch("^"+filenames_prefix+"part.*", file):
            with open(save_folder_path+file, 'rb') as f:
                generated_summaries += pickle.load(f)

    with open(save_folder_path+filenames_prefix+"full.pickle", 'wb') as f:
        pickle.dump(generated_summaries, f)