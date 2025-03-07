from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from os.path import join
from dotenv import load_dotenv
import numpy as np
import pickle
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path
from utils.dx import sample_noise_vectors_np
from utils.text_lm import (
    get_model_vocabulary,
    text_to_tokens_ids,
    texts_ids_to_embeddings,
    nearest_neighbor_search_on_textsV2,
    apply_post_processing_on_textsV2,
    ids_to_texts,
)
from utils.tools import print_timed, save_pickle, load_pickle

load_dotenv()

# BEGIN PARAMETERS
epsilons = [1] + [i for i in range(50, 2001, 50)]
dx_constant = 0.001
distance_metric = "euclidean"
cuda_device = "cpu"
batch_size = 250  # Modify according to your VRAM constraints
# END PARAMETERS

save_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    "datasets/multi_news/noisy_texts/llama_embedding_model",
)

torch.set_default_device(cuda_device)
if not os.path.exists(save_folderpath):
    os.makedirs(save_folderpath)

mnews = load_from_disk(
    join(
        os.environ["ROOT_SAVE_FOLDER"],
        "datasets/multi_news/concatenated_clean_1024_tokens",
    )
)

texts = mnews["document"]

# Select the 1500 randomly sampled texts
texts_indexes_to_compute = np.load(
    join(
        os.environ["ROOT_SAVE_FOLDER"],
        "datasets/multi_news/texts_indexes_to_compute.npy",
    )
)
texts = (np.array(texts)[texts_indexes_to_compute]).tolist()


def load_embedding_model() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        padding_side="left",
        revision="5f0b02c75b57c5855da9ae460ce51323ea669d8a",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        torch_dtype=torch.bfloat16,
        revision="5f0b02c75b57c5855da9ae460ce51323ea669d8a",
    ).to("cpu")
    model.eval()
    # Ensure the model is in eval mode
    # Config tokenizer according to Llama example
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


tokenizer, model = load_embedding_model()
vocab_embs = get_model_vocabulary(model)
# Explicitely convert to a float32
vocab_embs = vocab_embs.to(torch.float32).numpy()
vocab_size = vocab_embs.shape[0]
hidden_size = vocab_embs.shape[1]
del model  # Save RAM

# Transform texts to token ids
# Do NOT move this into the loop. By tokenizing here,
# we are sure all texts have the same number of tokens
# which is simpler when storing token_ids.
texts_ids, attention_mask, texts_tokens = text_to_tokens_ids(
    tokenizer, texts, return_tokens=False
)

attention_mask = attention_mask.numpy()
# Save attention mask to disk
save_pickle(save_folderpath, "attention_mask.pickle", attention_mask)

n = len(texts)
for epsilon in epsilons:
    print_timed(f"Epsilon = {epsilon}")
    part = 1
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)

        print_timed(f"Epsi{epsilon}: Processing slice {i}:{j}")

        texts_embeddings = texts_ids_to_embeddings(vocab_embs, texts_ids[i:j])

        print_timed("Sampling noise")
        # We need one noise vector per token
        noises = sample_noise_vectors_np(
            dimension=texts_embeddings.shape[2],
            shape1=texts_embeddings.shape[0],
            shape2=texts_embeddings.shape[1],
            epsilon=epsilon,
        )

        # Use attention_mask to avoid adding noise to special tokens like <PAD>
        # The following line multiplies the noise by zero for special tokens
        noises *= (attention_mask[i:j])[..., np.newaxis]

        texts_embeddings += noises

        print_timed("nearest_neighbor_search_on_textsV2")
        pivot_texts_ids = nearest_neighbor_search_on_textsV2(
            texts_embeddings,
            vocab_embs,
            attention_mask,
            tokenizer.pad_token_id,
            distance_metric,
        )

        noisy_texts_embeddings = vocab_embs[pivot_texts_ids]

        print_timed("Post-processing fix")
        noisy_texts_ids = apply_post_processing_on_textsV2(
            noisy_texts_embeddings,
            vocab_embs,
            attention_mask,
            tokenizer.pad_token_id,
            dx_constant,
            epsilon,
            distance_metric,
        )

        print_timed("ids_to_texts")
        noisy_texts = ids_to_texts(noisy_texts_ids, tokenizer)

        print_timed("Saving")
        filename = f"epsi{epsilon}partfile_{part:04d}"
        save_pickle(save_folderpath, f"{filename}.pickle", noisy_texts, False)

        # Also save noisy_texts_ids
        np.save(join(save_folderpath, f"{filename}.npy"), noisy_texts_ids)
        part += 1

    # Load all parts file for the current epsilon and save into one file
    slice_noisy_ids = []
    slice_noisy_texts_ids: list[np.ndarray] = []
    for file in sorted(os.listdir(save_folderpath)):
        if re.fullmatch(f"^epsi{epsilon}partfile.*.pickle", file):
            slice_noisy_ids += load_pickle(save_folderpath, file)
            os.remove(join(save_folderpath, file))
        elif re.fullmatch(f"^epsi{epsilon}partfile.*.npy", file):
            slice_noisy_texts_ids.append(np.load(join(save_folderpath, file)))
            os.remove(join(save_folderpath, file))

    save_pickle(save_folderpath, f"epsi{epsilon}full.pickle", slice_noisy_ids, False)

    slice_noisy_texts_ids_merged = np.concat(slice_noisy_texts_ids)
    np.save(
        join(save_folderpath, f"epsi{epsilon}full.npy"), slice_noisy_texts_ids_merged
    )
