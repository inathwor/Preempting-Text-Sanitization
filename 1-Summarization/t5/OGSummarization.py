from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from datetime import datetime
import pickle
import torch
import os
from os.path import join
import sys
# Add the main directory to sys.path to be able to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR

# PARAMETERS
default_save_path = join(ROOT_DIR, "datasets/multi_news/t5_generated_summaries/OnOriginalTexts/")
cuda_device = "cuda"
# END PARAMETERS

torch.set_default_device(cuda_device)
if not os.path.exists(default_save_path):
    os.makedirs(default_save_path)

texts = (load_from_disk(join(ROOT_DIR, "datasets/multi_news/concatenated_clean_1024_tokens")))['document']

summarizer = pipeline(
    "summarization",
    model="Falconsai/text_summarization",
    revision="6e505f907968c4a9360773ff57885cdc6dca4bfd",
    device=cuda_device,
    batch_size=8
)

results = summarizer(texts, max_new_tokens=142, do_sample=True, truncation=True)
generated_summaries = [result['summary_text'] for result in results]

# Save summaries to disk
filepath = default_save_path+"full.pickle"
with open(filepath, 'wb') as f:
    pickle.dump(generated_summaries, f)