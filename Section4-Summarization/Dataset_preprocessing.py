import os
from dotenv import load_dotenv
from os.path import join
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
import re

load_dotenv()

# BEGIN PARAMETERS
cuda_device = "cuda"
# END PARAMETERS

save_folderpath = join(os.environ["ROOT_SAVE_FOLDER"], "datasets/multi_news/")
if not os.path.exists(save_folderpath):
    os.makedirs(save_folderpath)

# We are leveraging the Multi-News dataset ([Link](https://huggingface.co/datasets/multi_news)).
mnews = load_dataset(
    "multi_news",
    revision="38cb206959a2cca3aa49858914fba76258a3dcaf",
    cache_dir=join(os.environ["ROOT_SAVE_FOLDER"], "datasets/multi_news/cache"),
)

# ## Concatenate the train, validation and test sets

mnews = concatenate_datasets([mnews["train"], mnews["validation"], mnews["test"]])

# ## Clean the dataset
# Many texts have problems:
# - Some have a big header -before the actual text- from internet archive "Web wide crawl with initial"
# - Some only contains "When you have eliminated the JavaScript, whatever remains must be an empty page."
# - Others have been truncated

# ### Cleaning Part 1: Truncate
# - Texts that start with "Web wide crawl with initial" should be truncated from start to the "|||||" character. See for instance indexes 690, 2931, 8801, 8841, 10817, 14995, 15483, 16714, 16810, 18014, 21246, 22587, 27140, 27297, 30521, 30573, 33827, 35102, 46732, 47022, 47805, 48164, 54158, 54906, 55679
#
# - Same for texts starting with "The seed for this crawl was a list" (e.g., indexes 590, 923, 937, 959, 1938, 55095, 55385, 55486, 55685, 55906)
#
# - Same for "These crawls are part of an effort to archive" (e.g., indexes 40, 323, 363, 438, 542, 55733, 55843, 55951, 55998, 56206)


def my_truncate(row, index):
    if (
        re.fullmatch("^Web wide crawl with initial.*", row["document"], re.DOTALL)
        or re.fullmatch(
            "^The seed for this crawl was a list.*", row["document"], re.DOTALL
        )
        or re.fullmatch(
            "^These crawls are part of an effort to archive.*",
            row["document"],
            re.DOTALL,
        )
        or re.fullmatch(
            "^Starting in 1996, Alexa Internet has been donating.*",
            row["document"],
            re.DOTALL,
        )
        or re.fullmatch(
            "^Crawl of outlinks from wikipedia.org.*", row["document"], re.DOTALL
        )
        or re.fullmatch(
            "^Sorry, the page you requested was not found.*", row["document"], re.DOTALL
        )
    ):
        if "|||||" not in row["document"]:
            print("Cannot remove header in index ", index)
        else:
            split_str = row["document"].split("||||| ", 1)
            row["document"] = split_str[1]
    return row


mnews = mnews.map(my_truncate, with_indices=True, keep_in_memory=True)

# ### Cleaning part 2: Remove

# index manually seen to be bad (several news in the same texts, summary longer than original text because the latter was truncated) etc ...
indices_to_remove = [
    38,
    73,
    248,
    424,
    1378,
    3273,
    3386,
    3486,
    3794,
    3998,
    4722,
    5011,
    7237,
    7398,
    9347,
    11783,
    12003,
    13404,
    14411,
    18658,
    19070,
    22681,
    22691,
    23120,
    30395,
    31659,
    32152,
    32789,
    33978,
    37881,
    40280,
    40798,
    44177,
    45346,
    46082,
    51151,
    55476,
    55685,
    55713,
]
mnews = mnews.filter(
    lambda example, idx: idx not in indices_to_remove, with_indices=True
)

print("Removing dissimilar indexes ...")
# Remove all samples where the text and its associated summary have a cosine similarity inferior to 0.5
embedding_model = SentenceTransformer(
    "all-mpnet-base-v2",
    device=cuda_device,
    revision="9a3225965996d404b775526de6dbfe85d3368642",
)
texts_embedding = embedding_model.encode(mnews["document"])
summaries_embedding = embedding_model.encode(mnews["summary"])

dissimilar_indexes = []
for i in range(texts_embedding.shape[0]):
    similarity = (util.cos_sim(texts_embedding[i], summaries_embedding[i])).item()
    if similarity < 0.5:
        dissimilar_indexes.append(i)

print(len(dissimilar_indexes))
mnews = mnews.filter(
    lambda example, idx: idx not in dissimilar_indexes, with_indices=True
)

print("Prepare dataset for BART (~20min)")
# ## Prepare dataset for BART
# Get the indexes of texts that are maximum 1024 tokens long for BART
texts = mnews["document"]

# Load BART
torch.set_default_device(cuda_device)

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/bart-large-cnn",
    device=cuda_device,
    torch_dtype="auto",
    use_fast=False,
    revision="37f520fa929c961707657b28798b30c003dd100b",
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/bart-large-cnn",
    torch_dtype="auto",
    revision="37f520fa929c961707657b28798b30c003dd100b",
)

# Tokenize without padding
tokenizer_ouptut = tokenizer(texts, padding=False)
texts_tokens = tokenizer_ouptut["input_ids"]

index_list = []
for index, tokens in enumerate(texts_tokens):
    if len(tokens) <= 1024:
        index_list.append(index)

# Restrict the dataset to selected text and save it for fast loading

filtered_dataset = mnews.select(index_list)

filtered_dataset.save_to_disk(
    join(
        save_folderpath,
        "concatenated_clean_1024_tokens",
    )
)

# Select 1500 random samples to be computed in the experiments
texts_indexes_to_compute = np.random.choice(
    len(filtered_dataset), size=1500, replace=False
)
np.save(join(save_folderpath, "texts_indexes_to_compute.npy"), texts_indexes_to_compute)
