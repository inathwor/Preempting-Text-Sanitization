from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import torch
import re
import os
from os.path import join
from dotenv import load_dotenv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path
from utils.tools import save_pickle, load_pickle, print_timed

load_dotenv()

# BEGIN PARAMETERS
cuda_device = "cuda"
batch_size = 25  # Modify according to your VRAM constraints
# END PARAMETERS

save_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    "datasets/multi_news/bart_generated_summaries/On_original_texts",
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

bart_tokenizer = AutoTokenizer.from_pretrained(
    "facebook/bart-large-cnn",
    device=cuda_device,
    torch_dtype="auto",
    use_fast=False,
    revision="37f520fa929c961707657b28798b30c003dd100b",
)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/bart-large-cnn",
    torch_dtype="auto",
    revision="37f520fa929c961707657b28798b30c003dd100b",
).to(cuda_device)
bart_model.eval()
# Ensure the model is in eval mode

n = len(texts)
filenumber = 0
for i in range(0, n, batch_size):
    if i % (batch_size * 10) == 0:
        print_timed(f"{i} processed.")
    j = min(i + batch_size, n)

    texts_slice = texts[i:j]

    # Tokenize
    encoded_input = bart_tokenizer(
        texts_slice, padding=True, truncation=False, return_tensors="pt"
    ).to(cuda_device)
    with torch.no_grad():
        encoded_outputs = bart_model.generate(**encoded_input)
        generated_summaries = bart_tokenizer.batch_decode(
            encoded_outputs, skip_special_tokens=True
        )

    # Save summaries to disk
    save_pickle(
        save_folderpath, f"filepart_{filenumber:04d}.pickle", generated_summaries, False
    )
    filenumber += 1

generated_summaries = []
for file in sorted(os.listdir(save_folderpath)):
    if re.fullmatch(f"^filepart_.*.pickle", file):
        generated_summaries += load_pickle(save_folderpath, file)
        os.remove(join(save_folderpath, file))

save_pickle(save_folderpath, "full.pickle", generated_summaries)
