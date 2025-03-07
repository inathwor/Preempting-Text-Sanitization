from transformers import pipeline
from datetime import datetime
import torch
import os
import re
from os.path import join
from dotenv import load_dotenv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path
from utils.tools import save_pickle, load_pickle, print_timed

load_dotenv()

# BEGIN PARAMETERS
embedding_model = "bart"  # "bart" or "llama", depending on the embedding model that was used to sanitize texts
batch_size = 128  # Modify according to your VRAM constraints
cuda_device = "cuda"
# END PARAMETERS

load_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    f"datasets/multi_news/noisy_texts/{embedding_model}_embedding_model",
)
save_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    f"datasets/multi_news/t5_generated_summaries/On_noisy_texts_with_{embedding_model}_embedding_model",
)
torch.set_default_device(cuda_device)
if not os.path.exists(save_folderpath):
    os.makedirs(save_folderpath)

summarizer = pipeline(
    "summarization",
    model="Falconsai/text_summarization",
    device=cuda_device,  # faster than device_map="auto" for this model
    batch_size=batch_size,
    revision="6e505f907968c4a9360773ff57885cdc6dca4bfd",
)

for file in sorted(os.listdir(load_folderpath)):
    regexp_match = re.fullmatch(r"^epsi(\d*)full.pickle$", file)
    if not regexp_match:
        continue
    epsilon = int(regexp_match.group(1))
    print_timed(f"Epsilon={epsilon}")

    noisy_texts = load_pickle(load_folderpath, f"epsi{epsilon}full.pickle")

    results = summarizer(
        noisy_texts,
        max_new_tokens=142,
        do_sample=True,
        truncation=True,
        batch_size=batch_size,
    )
    generated_summaries = [result["summary_text"] for result in results]

    # Save summaries to disk
    save_pickle(
        save_folderpath, f"epsi{epsilon}full.pickle", generated_summaries, False
    )
