from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
from os.path import join
from dotenv import load_dotenv
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path
from utils.tools import print_timed, save_pickle, load_pickle

load_dotenv()

# BEGIN PARAMETERS
embedding_model = "bart"  # "bart" or "llama", depending on the embedding model that was used to sanitize texts
batch_size = 50  # Modify according to your VRAM constraints
cuda_device = "cuda"
# END PARAMETERS

load_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    f"datasets/multi_news/noisy_texts/{embedding_model}_embedding_model",
)
save_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    f"datasets/multi_news/bart_generated_summaries/On_noisy_texts_with_{embedding_model}_embedding_model",
)

torch.set_default_device(cuda_device)
if not os.path.exists(save_folderpath):
    os.makedirs(save_folderpath)

bart_tokenizer = AutoTokenizer.from_pretrained(
    "facebook/bart-large-cnn",
    device=cuda_device,
    torch_dtype="auto",
    use_fast=False,
    revision="37f520fa929c961707657b28798b30c003dd100b",
)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/bart-large-cnn",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    revision="37f520fa929c961707657b28798b30c003dd100b",
).to(cuda_device)
bart_model.eval()  # Ensure the model is in eval mode

for file in sorted(os.listdir(load_folderpath)):
    regexp_match = re.fullmatch(r"^epsi(\d*)full.pickle$", file)
    if not regexp_match:
        continue
    epsilon = int(regexp_match.group(1))

    texts: list[str] = load_pickle(load_folderpath, file)

    print_timed(f"Epsilon={epsilon}")

    number_of_texts = len(texts)

    filenames_prefix = f"epsi{epsilon}"
    filenumber = 0
    for i in range(0, number_of_texts, batch_size):
        if i % (batch_size * 10) == 0:
            print_timed(f"{i} processed.")
        j = min(i + batch_size, number_of_texts)

        texts_slice = texts[i:j]

        encoded_input = bart_tokenizer(
            texts_slice,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(cuda_device)

        with torch.no_grad():
            encoded_output = bart_model.generate(**encoded_input)
            generated_summaries = bart_tokenizer.batch_decode(
                encoded_output, skip_special_tokens=True
            )

        # Save summaries to disk
        save_pickle(
            save_folderpath,
            f"{filenames_prefix}part_{filenumber:04d}.pickle",
            generated_summaries,
            False,
        )
        filenumber += 1

    generated_summaries = []
    for file in sorted(os.listdir(save_folderpath)):
        if re.fullmatch(f"^{filenames_prefix}part.*", file):
            generated_summaries += load_pickle(save_folderpath, file)
            os.remove(join(save_folderpath, file))

    save_pickle(
        save_folderpath, f"{filenames_prefix}full.pickle", generated_summaries, False
    )
