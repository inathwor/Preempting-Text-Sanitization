from datasets import load_from_disk
from transformers import pipeline
from datetime import datetime
import numpy as np
import torch
import os
from os.path import join
from dotenv import load_dotenv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path
from utils.tools import save_pickle

load_dotenv()

# BEGIN PARAMETERS
cuda_device = "cuda"
batch_size = 128  # Modify according to your VRAM constraints
# END PARAMETERS

save_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    "datasets/multi_news/t5_generated_summaries/On_original_texts",
)
torch.set_default_device(cuda_device)
if not os.path.exists(save_folderpath):
    os.makedirs(save_folderpath)

texts = (
    load_from_disk(
        join(
            os.environ["ROOT_SAVE_FOLDER"],
            "datasets/multi_news/concatenated_clean_1024_tokens",
        )
    )
)["document"]

# Select the 1500 randomly sampled texts
texts_indexes_to_compute = np.load(
    join(
        os.environ["ROOT_SAVE_FOLDER"],
        "datasets/multi_news/texts_indexes_to_compute.npy",
    )
)
texts = (np.array(texts)[texts_indexes_to_compute]).tolist()

summarizer = pipeline(
    "summarization",
    model="Falconsai/text_summarization",
    device=cuda_device,  # faster than device_map="auto" for this model
    batch_size=batch_size,
    revision="6e505f907968c4a9360773ff57885cdc6dca4bfd",
)

results = summarizer(
    texts, max_new_tokens=142, do_sample=True, truncation=True, batch_size=batch_size
)
generated_summaries = [result["summary_text"] for result in results]

# Save summaries to disk
save_pickle(save_folderpath, "full.pickle", generated_summaries, False)
