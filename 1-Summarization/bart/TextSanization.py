from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datetime import datetime
import torch
import os
from os.path import join
import numpy as np
from secrets import randbits
import pickle
import re
import pandas as pd
from cupyx.scipy.spatial import distance
import cupy as cp
import sys
# Add the main directory to sys.path to be able to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR

# PARAMS
epsilons = [1] + [i for i in range(5, 101, 5)]
directory_path = join(ROOT_DIR, "datasets/multi_news/noisy_texts/AsTensors/")
cuda_device = "cpu" # Sanitization does not need GPU
texts_per_slice = 1000 #Process X texts at the same time
# END PARAMS
torch.set_default_device(cuda_device)

if not os.path.exists(directory_path):
    os.makedirs(directory_path)

def load_embedding_model() -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    '''Load the language model and its associated Tokenizer'''
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/bart-large-cnn",
        device=cuda_device,
        torch_dtype="auto",
        use_fast=False,
        revision= "37f520fa929c961707657b28798b30c003dd100b"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/bart-large-cnn",
        torch_dtype="auto",
        revision= "37f520fa929c961707657b28798b30c003dd100b"
    )
    model.eval(); #Ensure the model is in eval mode
    return tokenizer, model

def text_to_tokens_ids(
    tokenizer: AutoTokenizer,
    texts: list[list[str]],
    return_tokens: bool = False # If the function should returns the tokens' strings, useful for debug but requires an additional operation.
)   -> tuple[torch.tensor, torch.tensor, list[list[str]]]:
    '''Tokenize text into token ids.'''
    encoded_input = tokenizer(
        texts, padding=True, truncation=False, add_special_tokens=False, return_tensors="pt"
    )

    texts_ids = encoded_input["input_ids"]
    texts_tokens = []
    if return_tokens:
        for text_ids in texts_ids:
            texts_tokens.append(tokenizer.convert_ids_to_tokens(text_ids))
    
    return texts_ids, encoded_input["attention_mask"], texts_tokens

def get_model_vocabulary(
    model
) -> torch.tensor:
    return model.get_input_embeddings().weight.detach()

def texts_ids_to_embeddings(
    vocabulary: torch.tensor,
    texts_ids: torch.tensor,
)   -> torch.tensor:

    return vocabulary[texts_ids]

def sample_noise_vectors(dimension: int, shape1: int, shape2: int, epsilon: float, dtype: np.dtype = np.float32) -> torch.tensor:
    '''Sample dx-private noise vectors.'''
    rng = np.random.default_rng(randbits(128))

    # Generate an array of noise vectors sampled from the multivariate normal distribution
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multivariate_normal.html
    # mean: Mean of the N-dimensional distribution. Chosen as the origin following (Feyisetan et al., 2020, Sec. 2.6)
    # cov: The covariance matrix of the distribution. Chosen as the identity matrix following (Feyisetan et al., 2020, Sec. 2.6)
    # size: Shape of the ouput. Set to the number of noise vectors we need.
    # check_valid: raise error if the covariance matrix is not positive semidefinite.
    # tol: Tolerance when checking the singular values in covariance matrix. Unset, default 1e-8.
    # method: Method for computing an intermediate matrix. Only impacts performances. "cholesky" is the fastest.
    origin = np.full(dimension, 0)
    cov_matrix = np.identity(dimension)
    noises = rng.multivariate_normal(mean=origin, cov=cov_matrix, size=(shape1, shape2), check_valid="raise", method="cholesky").astype(dtype)

    # Normalize each noise
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
    # x: The vector to be normalized
    # ord: Order of the norm. None uses the Frobenius matrix norm, which, applied on vectors, results in the Euclidean/L2 norm.
    # axis: Specifies the axis of x along which to compute the vector norms. We want each single vector to be normarlized so the last axis i.e. -1
    # keepdims: The normed axis are left in the result as dimensions with size one.
    noises /= np.linalg.norm(noises, ord=None, axis=-1, keepdims=True).astype(dtype)

    # Generate an array of magnitude scalars.
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.gamma.html
    # shape: Shape of the gamma distribution, often noted "k". Set to the embeddings' dimension following (Feyisetan et al., 2020, Sec. 2.6) and (Qu et al., 2021, Sec. 3.2.3)
    # scale: Scale of the distribution, often noted theta. Set to 1/epsilon following (Feyisetan et al., 2020, Sec. 2.6) and (Qu et al., 2021, Sec. 3.2.3)
    # size: Shape of the ouput. Set to the number of magnitude scalars we need.
    magnitudes = rng.gamma(shape=dimension, scale=1.0/epsilon, size=(shape1, shape2)).astype(dtype) 

    noises *= magnitudes[..., np.newaxis]
    
    return torch.from_numpy(noises).to(cuda_device)


def noisy_embeddings_to_ids(texts_embeddings: torch.tensor, vocabulary: torch.tensor) -> list[list[int]]:
    '''Considering the texts_embeddings, perform a nearest neighbor search in the vocabulary and return
    the associated token_id.'''
    number_of_texts = texts_embeddings.shape[0]
    padded_number_of_tokens = texts_embeddings.shape[1]
    noisy_texts_ids = cp.empty((number_of_texts, padded_number_of_tokens), dtype=np.uint32)
    for i in range(number_of_texts):
        distances = distance.cdist(texts_embeddings[i], vocabulary, 'euclidean')
        noisy_texts_ids[i] = distances.argmin(axis=-1)
    return noisy_texts_ids.tolist()

def ids_to_texts(texts_ids: list[list[int]], tokenizer) -> list[str]:
    '''Leverage the tokenizer to transform token ids into texts.'''
    return [tokenizer.decode(e, skip_special_tokens=True) for e in texts_ids]

mnews = load_from_disk(join(ROOT_DIR, "datasets/multi_news/concatenated_clean_1024_tokens"))

texts = mnews['document']

tokenizer, model = load_embedding_model()
vocabulary = get_model_vocabulary(model)

del model
torch.cuda.empty_cache()

# Transform texts to token ids
texts_ids, attention_mask, texts_tokens = text_to_tokens_ids(tokenizer, texts, return_tokens=False)

# Save attention mask to disk
with open(join(directory_path, "attention_mask.pickle"), "wb") as f:
    pickle.dump(attention_mask, f)

n = len(texts)
for epsilon in epsilons:
    print("Epsilon = ", epsilon)
    part = 1
    for i in range(0, n, texts_per_slice):
        torch.cuda.empty_cache()
        j = min(i+texts_per_slice, n)

        print(datetime.now().strftime('%Hh%Mm%Ss'), "Processing slice ", str(i)+":"+str(j))

        texts_embeddings = texts_ids_to_embeddings(vocabulary, texts_ids[i:j])

        print(datetime.now().strftime('%Hh%Mm%Ss'), "Sampling noise")
        # We need one noise vector per token
        noises = sample_noise_vectors(dimension=texts_embeddings.shape[2], 
                                    shape1=texts_embeddings.shape[0], 
                                    shape2=texts_embeddings.shape[1], 
                                    epsilon=epsilon)

        # Use attention_mask to avoid adding noise to special tokens like <PAD>
        # The following line multiplies the noise by zero for special tokens
        noises *= (attention_mask[i:j])[..., np.newaxis]

        print(datetime.now().strftime('%Hh%Mm%Ss'), "Adding noise")
        texts_embeddings += noises

        print(datetime.now().strftime('%Hh%Mm%Ss'), "noisy_embeddings_to_ids")
        noisy_texts_ids = noisy_embeddings_to_ids(texts_embeddings, vocabulary)

        del texts_embeddings; del noises# Force freeing

        print(datetime.now().strftime('%Hh%Mm%Ss'), "Saving")
        filepath = directory_path+"epsi"+str(epsilon)+"part"+f"file_{part:04d}"+".pickle"
        with open(filepath, 'wb') as f:
            pickle.dump(noisy_texts_ids, f)
        del noisy_texts_ids # Force freeing
        part += 1

    # Load all slices' files and merge into one file
    slice_noisy_ids = []
    for file in sorted(os.listdir(directory_path)):
        if re.fullmatch(f"^epsi{epsilon}part.*", file):
            with open(join(directory_path, file), 'rb') as f:
                slice_noisy_ids += pickle.load(f)
    
    # Save as a tensor
    slice_noisy_ids_pt = torch.tensor(slice_noisy_ids, dtype=torch.int64).to(cuda_device)
    with open(join(directory_path, f"epsi{epsilon}full.pickle"), 'wb') as f:
        pickle.dump(slice_noisy_ids_pt, f)

    # Delete part files
    for file in sorted(os.listdir(directory_path)):
        if re.fullmatch(f"^epsi{epsilon}part.*", file):
            os.remove(join(directory_path, file))

    # Convert ids to text and save
    texts_strs = ids_to_texts(slice_noisy_ids, tokenizer)
    texts_strs_save_folderpath = join(directory_path, "../", "AsStrings/")
    if not os.path.exists(texts_strs_save_folderpath):
        os.makedirs(texts_strs_save_folderpath)
    with open(join(texts_strs_save_folderpath, f"epsi{epsilon}full.pickle"), "wb") as f:
        pickle.dump(texts_strs, f)