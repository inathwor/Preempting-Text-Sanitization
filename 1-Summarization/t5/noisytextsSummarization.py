from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from datetime import datetime
import pickle
import torch
import os
from os.path import join
import re
import sys
# Add the main directory to sys.path to be able to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR

# PARAMS
load_folderpath = join(ROOT_DIR, "datasets/multi_news/noisy_texts/AsStrings/")
save_folderpath = join(ROOT_DIR, "datasets/multi_news/t5_generated_summaries/OnSanitizedTexts/")
epsilons = [1] + [i for i in range(5, 101, 5)]
cuda_device = "cuda"
# END PARAMS
torch.set_default_device(cuda_device)

if not os.path.exists(save_folderpath):
    os.makedirs(save_folderpath)

summarizer = pipeline(
    "summarization",
    model="Falconsai/text_summarization",
    revision="6e505f907968c4a9360773ff57885cdc6dca4bfd",
    device=cuda_device,
    batch_size=8
)

for epsilon in epsilons:
    torch.cuda.empty_cache()
    print(datetime.now().strftime('%Hh%Mm%Ss'), "Epsilon=", str(epsilon))

    with open(join(load_folderpath, f"epsi{epsilon}full.pickle"), 'rb') as f:
        noisy_texts = pickle.load(f)

    results = summarizer(noisy_texts, max_new_tokens=142, do_sample=True, truncation=True)
    generated_summaries = [result['summary_text'] for result in results]
        
    # Save summaries to disk
    filepath = join(save_folderpath, f"epsi{epsilon}full.pickle")
    with open(filepath, 'wb') as f:
        pickle.dump(generated_summaries, f)