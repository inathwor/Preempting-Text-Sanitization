import numpy as np
import os
from os.path import join
from dotenv import load_dotenv
from dotenv import load_dotenv
import sys
from pathlib import Path

# Use utils package from other folder
sys.path.append(join(str(Path(__file__).parent.parent), "Section4-Summarization"))
from utils.dx import sample_noise_vectors_np, noisy_embeddings_to_ids_cp
from utils.tools import load_pickle, save_pickle, print_timed

load_dotenv()

# BEGIN PARAMETERS
epsilons = [i for i in range(5, 56, 5)]
repeat = 1000
max_chunk_size = 100  # Number of words to be processed at the same time. Modify according to your VRAM constraints.
# END PARAMETERS

save_folderpath = join(os.environ["ROOT_SAVE_FOLDER"], "emb_space_analysis")
if not os.path.exists(save_folderpath):
    os.makedirs(save_folderpath)

glove_data_folderpath = os.environ["ROOT_SAVE_FOLDER"]
fasttext_data_folderpath = os.environ["ROOT_SAVE_FOLDER"]

glove = load_pickle(glove_data_folderpath, "glove.6B.300d.pkl")
fasttext = load_pickle(fasttext_data_folderpath, "wiki.en.pkl")

common_words = set(fasttext.keys()) & set(glove.keys())
glove_commons = {key: glove[key] for key in common_words}

# Saving RAM
del glove
del fasttext

# Put the data into appropriate structures to be able to transform a word into its ID or its embedding easily.
vocab_embs = np.array(list(glove_commons.values()))
vocab_size = vocab_embs.shape[0]
hidden_size = vocab_embs.shape[1]

del glove_commons  # Saving RAM

for epsilon in epsilons:
    print_timed(f"epsilon={epsilon}")
    results = np.zeros((vocab_size), dtype=int)

    # Perform the exact nearest neighbor search on GPU
    # Take max_chunk_size words at the same time, each of them in _repeat_ number of occurences
    for i in range(0, vocab_size, max_chunk_size):
        j = min(i + max_chunk_size, vocab_size)
        current_chunk_size = j - i
        if i % 10000 == 0:
            print_timed(f"epsilon={epsilon}, i={i}")

        # Take the word embedding _repeat_ times
        indexes = np.repeat(np.arange(i, j), repeat)
        words_embeddings = vocab_embs[indexes]

        # Sample repeat*current_chunk_size noise vectors
        noises = sample_noise_vectors_np(
            dimension=hidden_size,
            shape1=1,
            shape2=repeat * current_chunk_size,
            epsilon=epsilon,
        )[0]

        # Adding noise to embeddings
        noisy_words_embeddings = words_embeddings + noises

        # Find the replacement word index for each vector
        noisy_words_ids = noisy_embeddings_to_ids_cp(noisy_words_embeddings, vocab_embs)

        noisy_words_ids = noisy_words_ids.reshape((current_chunk_size, repeat))
        for k in range(i, j):
            # Count the number of times the initial word was returned
            results[k] = np.count_nonzero(noisy_words_ids[k - i] == k)

    save_pickle(
        save_folderpath,
        f"CommonFastTextGlove-ENN-epsi{epsilon}-{repeat}repeat-statIdentical.pickle",
        results,
    )

# Load and print results
results = []
for epsilon in epsilons:
    results.append(
        load_pickle(
            save_folderpath,
            f"CommonFastTextGlove-ENN-epsi{epsilon}-{repeat}repeat-statIdentical.pickle",
        )
    )

results = np.array(results).mean(axis=-1)
print({epsilons[i]: results[i] for i in range(len(epsilons))})
