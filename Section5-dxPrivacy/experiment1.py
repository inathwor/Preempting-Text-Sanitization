from annoy import AnnoyIndex
import numpy as np
from numpy.random import normal
from os.path import join
from collections import Counter
from scipy.spatial import distance
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# Use utils package from other folder
sys.path.append(join(str(Path(__file__).parent.parent), "Section4-Summarization"))
from utils.tools import load_pickle, print_timed

load_dotenv()

# BEGIN PARAMETERS
words = ["encryption", "hockey", "spacecraft"]
epsilons = [19, 25, 35, 43]
repeat = 1000
# END PARAMETERS

print_timed("Loading GloVe and FastText")
glove_data_folderpath = os.environ["ROOT_SAVE_FOLDER"]
fasttext_data_folderpath = os.environ["ROOT_SAVE_FOLDER"]

# Import GloVe and Fasttext and keep GloVe embeddings of words in common

glove = load_pickle(glove_data_folderpath, "glove.6B.300d.pkl")
fasttext = load_pickle(fasttext_data_folderpath, "wiki.en.pkl")

common_words = set(fasttext.keys()) & set(glove.keys())
glove_commons = {key: glove[key] for key in common_words}

# Saving RAM
del glove
del fasttext

# Put the data into appropriate structures to be able to transform a word into its ID or its embedding easily.

vocab_embs = np.array(list(glove_commons.values()))
words_to_id = {word: index for index, word in enumerate(glove_commons.keys())}
id_to_words = list(glove_commons.keys())
vocab_size = vocab_embs.shape[0]
hidden_size = vocab_embs.shape[1]

del glove_commons  # Saving RAM
print_timed(f"vocab_size={vocab_size}, hidden_size={hidden_size}")

# Define the nearest neighbor search function.
# We define two version, replace_word using Approximate Nearest Neighbor and replace_word_exact using Exact Nearest Neighbor


# Code taken from https://github.com/awslabs/sagemaker-privacy-for-nlp/blob/master/source/sagemaker/src/package/data_privatization/data_privatization.py
def replace_word(sensitive_word, epsilon, ann_index, embedding_dims, sensitivity=1.0):
    """
    Given a word will inject noise according to the provided epsilon value and return a perturbed word.
    """
    # Generate a noise vector
    noise = generate_laplacian_noise_vector(embedding_dims, sensitivity, epsilon)
    # Get vector of sensitive word
    original_vec = vocab_embs[words_to_id[sensitive_word]]
    # Get perturbed vector
    noisy_vector = original_vec + noise
    # Get item closest to noisy vector
    closest_item = ann_index.get_nns_by_vector(noisy_vector, 1)[0]
    # Get word from item
    privatized_word = id_to_words[closest_item]
    return privatized_word


def generate_laplacian_noise_vector(dimension, sensitivity, epsilon):
    rand_vec = normal(size=dimension)
    normalized_vec = rand_vec / np.linalg.norm(rand_vec)
    magnitude = np.random.gamma(shape=dimension, scale=sensitivity / epsilon)
    return normalized_vec * magnitude


# Index definition is the same as https://github.com/awslabs/sagemaker-privacy-for-nlp/blob/master/source/sagemaker/src/package/data_privatization/data_privatization.py
# Create approximate nearest neighbor index
num_trees = 50

ann_index = AnnoyIndex(hidden_size, "euclidean")

for vector_num, vector in enumerate(vocab_embs):
    ann_index.add_item(vector_num, vector)

print_timed("Building annoy index...")
assert ann_index.build(num_trees)
print_timed("Annoy index built")


# This function is almost exactly the same as above, except that the exact nearest neighbor
# is found instead of leveraging the annoy index.
def replace_word_exact(sensitive_word, epsilon, embedding_dims, sensitivity=1.0):
    """
    Given a word will inject noise according to the provided epsilon value and return a perturbed word.
    """
    # Generate a noise vector
    noise = generate_laplacian_noise_vector(embedding_dims, sensitivity, epsilon)
    # Get vector of sensitive word
    original_vec = vocab_embs[words_to_id[sensitive_word]]
    # Get perturbed vector
    noisy_vector = original_vec + noise
    # Get item closest to noisy vector
    closest_item = (
        distance.cdist(np.array([noisy_vector]), vocab_embs, "euclidean")
        .argmin(axis=-1)
        .item()
    )
    # Get word from item
    privatized_word = id_to_words[closest_item]
    return privatized_word


# ## Experiment 1

for epsilon in epsilons:
    print_timed(f"Epsilon = {epsilon}")
    for word in words:
        print_timed(f"Word = {word}")
        # With Approximate Nearest Neighbor
        privatized_word = [
            replace_word(word, epsilon, ann_index, hidden_size) for _ in range(repeat)
        ]
        print_timed("Approximate nearest neighbor:", Counter(privatized_word))

        # With Exact Nearest Neighbor
        privatized_word = [
            replace_word_exact(word, epsilon, hidden_size) for _ in range(repeat)
        ]
        print_timed("Exact nearest neighbor:", Counter(privatized_word))
