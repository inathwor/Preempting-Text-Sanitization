import numpy as np
import cupy as cp
from cupyx.scipy.spatial import distance
from datetime import datetime
import math
import pickle
from os.path import join
from typing import Any, Type


def best_uint_type(x: int) -> Type[np.unsignedinteger]:
    """Check the number of bits needed to represent x and
    return the appropriate numpy dtype."""
    if x < 2**16:
        return np.uint16
    elif x < 2**32:
        return np.uint32
    else:
        return np.uint64


def print_timed(*args, **kwargs) -> None:
    print(datetime.now().strftime("%Hh%Mm%Ss"), *args, **kwargs)


def save_pickle(
    folderpath: str, filename: str, toBeSaved, datetime_prefix: bool = False
) -> None:
    if datetime_prefix:
        filename = f"{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}_{filename}"
    filepath = join(folderpath, filename)

    with open(filepath, "wb") as f:
        pickle.dump(toBeSaved, f)


def load_pickle(folderpath: str, filename: str) -> Any:
    with open(join(folderpath, filename), "rb") as f:
        result = pickle.load(f)
    return result


def best_chunk_size(input_size: int, unitary_size: int) -> int:
    """Compute the number of elements which can be processed at the same time
    in a GPU computation. input_size is the amount of bytes which has to be
    fully loaded in memory during the entire computation. unitary_size is the
    amount of bytes needed when computing one element, both for storing
    this element and its associated result."""
    # Size of unallocated VRAM
    available_vram = cp.cuda.runtime.memGetInfo()[0]

    mempool = cp.get_default_memory_pool()
    # Size of VRAM allocated to CuPy but unused
    cp_mempool_available = mempool.total_bytes() - mempool.used_bytes()

    available_vram = available_vram + cp_mempool_available
    # Using 90% of the available VRAM
    available_vram = math.floor(available_vram * 0.90)

    # Memory available for computing elements
    memory_size = available_vram - input_size

    # Chunk size is at least 1
    return max(1, math.floor(memory_size / unitary_size))


def rank_neighbors(
    embeddings: np.ndarray,
    vocabulary: np.ndarray,
    distance_metric: str = "euclidean",
) -> np.ndarray:
    """For each embedding, ranks the elements in the vocabulary according to their distance
    with the embedding. Returns a numpy array of shape (embeddings.shape[0], vocabulary.shape[0])
    where array[i][j] contains the rank of the j-th vocabulary element in the list of neighbors of
    the i-th embedding. The function benefits from cupy for a faster computation of distances and
    sorting on GPU. Computes chunk-by-chunk to avoid overloading the VRAM."""
    # Calculate the ideal chunk_size for the computation. Note that cupyx.scipy.spatial.distance.cdist(x1, x2)
    # first allocates an array of shape (x1.shape[0], x2.shape[0]) with np.float64 precision (8 bytes).
    # Argsort returns an array of dtype int64 (8 bytes).
    chunk_size = best_chunk_size(
        vocabulary.shape[0] * vocabulary.shape[1] * 4,
        embeddings.shape[1] * 4 + vocabulary.shape[0] * 8 * 2,
    )
    # Argsort allocates more than expected. Choosing a smaller chunk_size for safety.
    chunk_size = math.floor(chunk_size / 4)

    number_of_words = embeddings.shape[0]
    # Copy the vocabulary to GPU. Casting to float32 as distance.cdist will do it anyway,
    # and we want to avoid a second copy.
    vocab_cp = cp.asarray(vocabulary, dtype="float32")

    # Ranks will be stored on RAM as a numpy array
    words_neighbors_ranked = np.empty(
        (embeddings.shape[0], vocab_cp.shape[0]),
        dtype=best_uint_type(vocab_cp.shape[0]),
    )

    for i in range(0, number_of_words, chunk_size):
        j = min(i + chunk_size, number_of_words)
        words_neighbors_ranked[i:j, :] = (
            distance.cdist(embeddings[i:j], vocab_cp, distance_metric)
            .argsort(axis=-1)
            .argsort(axis=-1)
            .get()
        )  # argsort on GPU before copying to RAM with .get()

    return words_neighbors_ranked
