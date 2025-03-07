import numpy as np
from secrets import randbits
import cupy as cp
from cupyx.scipy.spatial import distance
import torch
from typing import Type
from utils.tools import best_uint_type, best_chunk_size, rank_neighbors


def sample_noise_vectors_np(
    dimension: int,
    shape1: int,
    shape2: int,
    epsilon: float,
    dtype: Type[np.floating] = np.float32,
) -> np.ndarray:
    """Sample shape1*shape2 noise vectors of dimensions _dimension_ according to the
    definition by (Feyisetan et al., 2020) and (Qu et al., 2021).

    Args:
        dimension (int): The number of dimensions for the noise vectors. Also called hidden size.
        shape1 (int): The first shape of the array of noise vectors
        shape2 (int): The second shape of the array of noise vectors
        epsilon (float): The epsilon value in the dx-privacy formula.
        dtype (Type[np.floating], optional): The data type of the vectors. Defaults to np.float32.

    Returns:
        np.ndarray: The noise vector of shape (shape1, shape2, dimension)
    """
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
    noises = rng.multivariate_normal(
        mean=origin,
        cov=cov_matrix,
        size=(shape1, shape2),
        check_valid="raise",
        method="cholesky",
    ).astype(dtype)

    # Normalize each noise by dividing each vector by its norm.
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
    # x: The vector to be normalized
    # ord: Order of the norm. None uses the Frobenius matrix norm, which, applied on vectors, results in the Euclidean/L2 norm.
    # axis: Specifies the axis of x along which to compute the vector norms. We want each single vector to be normalized thus choosing the last axis i.e. -1
    # keepdims: The normed axis are left in the result as dimensions with size one.
    noises /= np.linalg.norm(noises, ord=None, axis=-1, keepdims=True).astype(dtype)

    # Generate an array of magnitude scalars.
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.gamma.html
    # shape: Shape of the gamma distribution, often noted "k". Set to the embeddings' dimension following (Feyisetan et al., 2020, Sec. 2.6) and (Qu et al., 2021, Sec. 3.2.3)
    # scale: Scale of the distribution, often noted theta. Set to 1/epsilon following (Feyisetan et al., 2020, Sec. 2.6) and (Qu et al., 2021, Sec. 3.2.3)
    # size: Shape of the ouput. Set to the number of magnitude scalars we need.
    magnitudes = rng.gamma(
        shape=dimension, scale=1.0 / epsilon, size=(shape1, shape2)
    ).astype(dtype)

    noises *= magnitudes[..., np.newaxis]

    return noises


def sample_noise_vectors(
    dimension: int,
    shape1: int,
    shape2: int,
    epsilon: float,
    dtype: Type[np.floating] = np.float32,
) -> torch.Tensor:
    """Sample shape1*shape2 noise vectors of dimensions _dimension_ according to the
    definition by (Feyisetan et al., 2020) and (Qu et al., 2021).

    Args:
        dimension (int): The number of dimensions for the noise vectors. Also called hidden size.
        shape1 (int): The first shape of the array of noise vectors
        shape2 (int): The second shape of the array of noise vectors
        epsilon (float): The epsilon value in the dx-privacy formula.
        dtype (Type[np.floating], optional): The data type of the vectors. Defaults to np.float32.

    Returns:
        torch.tensor: The noise vector of shape (shape1, shape2, dimension)
    """
    noises = sample_noise_vectors_np(dimension, shape1, shape2, epsilon, dtype)
    return torch.from_numpy(noises)


def noisy_embeddings_to_ids_cp_chunked(
    embeddings: np.ndarray,
    vocabulary: np.ndarray,
    distance_metric: str = "euclidean",
    chunk_size: int = -1,
) -> np.ndarray:
    """Performs a nearest neighbor search of the embeddings against the vocabulary. This second version performs the computation chunk-by-chunk to avoid VRAM overload.

    Args:
        embeddings (np.ndarray): A two-dimensional array containing the embeddings we want to process.
        vocabulary (np.ndarray): A two-dimensional array containing all the embeddings of the vocabulary
            to compute the nearest neighbor search against.
        distance_metric (str, optional): The distance metric to use for ranking elements of the vocabulary. Defaults to "euclidean".
        chunk_size (int, optional): The number of elements in embeddings to compute at a time. Defaults to -1, which will
            determine the best chunk size for considering the VRAM available.

    Returns:
        np.ndarray: A one-dimensional array of shape (embeddings.shape[0]) containing the vocabulary index of the
       nearest neighbor for each embedding.
    """
    if chunk_size == -1:
        # Calculate the ideal chunk_size for the computation. Note that
        # cupyx.scipy.spatial.distance.cdist(x1, x2) first allocates an array of shape
        # (x1.shape[0], x2.shape[0]) with float64 precision.
        chunk_size = best_chunk_size(
            vocabulary.shape[0] * vocabulary.shape[1] * 4,
            embeddings.shape[1] * 4 + vocabulary.shape[0] * 8,
        )

    input_size = embeddings.shape[0]
    # Copy the vocabulary to GPU. Casting to float32 as distance.cdist will do it anyway,
    # and we want to avoid a second copy.
    vocab_cp = cp.asarray(vocabulary, dtype="float32")

    noisy_ids = np.empty(shape=(input_size), dtype=best_uint_type(vocabulary.shape[0]))

    # Compute the distances and find the nearest neighbor of each embedding, chunk-by-chunk
    for i in range(0, input_size, chunk_size):
        j = min(i + chunk_size, input_size)
        distances = distance.cdist(embeddings[i:j], vocab_cp, distance_metric)
        noisy_ids[i:j] = distances.argmin(axis=-1).get()

    return noisy_ids


def noisy_embeddings_to_ids_cp(
    embeddings: np.ndarray,
    vocabulary: np.ndarray,
    distance_metric: str = "euclidean",
) -> np.ndarray:
    """Performs a nearest neighbor search of the embeddings against the vocabulary.

    Args:
        embeddings (np.ndarray): A two-dimensional array containing the embeddings we want to process.
        vocabulary (np.ndarray): A two-dimensional array containing all the embeddings of the vocabulary
            to compute the nearest neighbor search against.
        distance_metric (str, optional): The distance metric to use for ranking elements of the vocabulary. Defaults to "euclidean".

    Returns:
        np.ndarray: A one-dimensional array of shape (embeddings.shape[0]) containing the vocabulary index of the
       nearest neighbor for each embedding.
    """
    return noisy_embeddings_to_ids_cp_chunked(embeddings, vocabulary, distance_metric)


def dx_post_processing(
    embeddings: np.ndarray,
    vocabulary: np.ndarray,
    dx_constant: float,
    epsilon: int,
    distance_metric: str = "euclidean",
) -> np.ndarray:
    """Applies the post-processing fix proposed in (Asghar et al., 2024).

    Args:
        embeddings (np.ndarray): A two-dimensional array containing the embeddings we want to process.
        vocabulary (np.ndarray): A two-dimensional array containing all the embeddings of the vocabulary
            to compute the fix against.
        dx_constant (float): The c constant in the formula of (Asghar et al., 2024).
        epsilon (int): The epsilon value in the dx-privacy formula.
        distance_metric (str, optional): The distance metric to use for ranking elements of the vocabulary. Defaults to "euclidean".

    Returns:
        np.ndarray: A one-dimensional numpy array containing the ids of the sampled replacements.
    """
    input_size = embeddings.shape[0]
    vocab_size = vocabulary.shape[0]

    # Rank the elements of the vocabulary according to their distance with each of the embeddings
    embeddings_neighbors_ranked = cp.array(
        rank_neighbors(embeddings, vocabulary, distance_metric)
    )

    # Compute probabilities
    probabilities = cp.exp(-dx_constant * epsilon * embeddings_neighbors_ranked)

    # Normalize probabilities along the last axis
    probabilities /= probabilities.sum(axis=-1, keepdims=True)

    # Sample one element for each embedding and return their ids
    noisy_ids = np.empty((input_size), dtype=best_uint_type(vocab_size))
    for i in range(input_size):
        noisy_ids[i] = cp.random.choice(vocab_size, size=1, p=probabilities[i]).item()

    return noisy_ids
