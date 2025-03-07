import torch
import numpy as np
import cupy as cp
from transformers import AutoTokenizer, AutoModel
from .dx import noisy_embeddings_to_ids_cp, dx_post_processing
from .tools import best_uint_type


def text_to_tokens_ids(
    tokenizer: AutoTokenizer,
    texts: list[list[str]],
    return_tokens: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[list[str]]]:
    """Tokenize text into token ids.

    Args:
        tokenizer (AutoTokenizer): The tokenize to use.
        texts (list[list[str]]): The texts to tokenize.
        return_tokens (bool, optional): If the function should returns the tokens' strings, useful for debug but requires an additional operation. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, list[list[str]]]: A Tuple containing i) the texts_ids resulting from tokenization, ii) the attention mask marking the position of pad tokens and iii) An empty list if return_tokens if False, or the tokenized texts as lists of strings of tokens if return_tokens is True.
    """

    # Tokenize the texts, pad and truncate if needed
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=False,
        add_special_tokens=False,
        return_tensors="pt",
    )
    texts_ids = encoded_input["input_ids"]

    # If needed, also return the texts as strings of tokens.
    texts_tokens = []
    if return_tokens:
        for text_ids in texts_ids:
            texts_tokens.append(tokenizer.convert_ids_to_tokens(text_ids))

    return texts_ids, encoded_input["attention_mask"], texts_tokens


def get_model_vocabulary(model: AutoModel) -> torch.Tensor:
    """Get the vocabulary (i.e., token embedding model) of the language model.

    Args:
        model (_type_): The language model

    Returns:
        torch.Tensor: The vocabulary of the language model
    """
    return model.get_input_embeddings().weight.detach().clone()


def texts_ids_to_embeddings(
    vocabulary: torch.Tensor,
    texts_ids: torch.Tensor,
) -> torch.Tensor:
    return vocabulary[texts_ids]


def nearest_neighbor_search_on_texts(
    texts_embeddings: torch.Tensor,
    vocabulary: torch.Tensor,
    distance_metric: str = "euclidean",
) -> list[list[int]]:
    """Performs a nearest neighbor search on the texts_embeddings array against the vocabulary.

    Args:
        texts_embeddings (torch.Tensor): A three-dimensional array containing the embeddings we want to process.
        vocabulary (torch.Tensor): A two-dimensional array containing all the embeddings of the vocabulary
            to compute the nearest neighbor search against.
        distance_metric (str, optional): The distance metric to use for ranking elements of the vocabulary. Defaults to "euclidean".

    Returns:
        list[list[int]]: The nearest neighbor of each embeddings.
    """

    number_of_texts = texts_embeddings.shape[0]
    padded_number_of_tokens = texts_embeddings.shape[1]

    # Copy the vocabulary to GPU once to avoid copying each turn
    # of the loop below.
    vocab_cp = cp.asarray(vocabulary, dtype="float32")

    # Performs a nearest neighbor search for each text one-by-one.
    noisy_texts_ids = np.empty(
        (number_of_texts, padded_number_of_tokens),
        dtype=best_uint_type(vocabulary.shape[0]),
    )
    for i in range(number_of_texts):
        noisy_texts_ids[i] = noisy_embeddings_to_ids_cp(
            texts_embeddings[i], vocab_cp, distance_metric
        )
    return noisy_texts_ids.tolist()


def nearest_neighbor_search_on_textsV2(
    texts_embeddings: np.ndarray,
    vocabulary: np.ndarray,
    attention_mask: np.ndarray,
    pad_token_id: int,
    distance_metric: str = "euclidean",
) -> np.ndarray:
    """Performs a nearest neighbor search on the texts_embeddings array against the vocabulary.
    This second version does not process pad tokens marked as such by the attention_mask and directly
    puts pad_token_id as their associated result.

    Args:
        texts_embeddings (torch.Tensor): A three-dimensional array containing the embeddings we want to process.
        vocabulary (torch.Tensor): A two-dimensional array containing all the embeddings of the vocabulary to compute the nearest neighbor search against.
        attention_mask (np.ndarray): A two-dimensional array of the same shape as texts_embeddings, where a 0 marks the position of a pad token.
        pad_token_id (int): The token id of a pad token (depends on the tokenizer)
        distance_metric (str, optional): The distance metric to use for ranking elements of the vocabulary. Defaults to "euclidean".

    Returns:
        np.ndarray: The nearest neighbor of each embeddings.
    """
    number_of_texts = texts_embeddings.shape[0]
    padded_number_of_tokens = texts_embeddings.shape[1]

    # Copy the vocabulary to GPU once to avoid copying each turn
    # of the loop below.
    vocab_cp = cp.asarray(vocabulary, dtype="float32")

    # Declare the result as a two-dimensional array, consisting of only pad tokens for now.
    noisy_texts_ids = np.full(
        (number_of_texts, padded_number_of_tokens),
        pad_token_id,
        dtype=best_uint_type(vocabulary.shape[0]),
    )

    # Performs the nearest neighbor search text-by-text for all text embeddings, excluding pad tokens.
    for i in range(number_of_texts):
        # Gather the indexes not containing pad tokens.
        indexes_to_be_computed = np.where(attention_mask[i] == 1)[0]

        # Indexes to be computed follow each other, but they can be either left or right padded.
        first_index_to_be_computed = indexes_to_be_computed[0]
        last_index_to_be_computed = indexes_to_be_computed[-1] + 1

        noisy_texts_ids[i][first_index_to_be_computed:last_index_to_be_computed] = (
            noisy_embeddings_to_ids_cp(
                texts_embeddings[i][
                    first_index_to_be_computed:last_index_to_be_computed
                ],
                vocab_cp,
                distance_metric,
            )
        )
    return noisy_texts_ids


def apply_post_processing_on_texts(
    texts_embeddings: np.ndarray,
    vocabulary: np.ndarray,
    dx_constant: float,
    epsilon: int,
    distance_metric: str = "euclidean",
) -> np.ndarray:
    """Applies the post-processing fix proposed in (Asghar et al., 2024) on texts_embeddings, text-by-text.

    Args:
        texts_embeddings (np.ndarray): A three-dimensional array containing the embeddings we want to process.
        vocabulary (np.ndarray): A two-dimensional array containing all the embeddings of the vocabulary to compute the fix against.
        dx_constant (float): The c constant in the formula of (Asghar et al., 2024).
        epsilon (int): The epsilon value in the dx-privacy formula.
        distance_metric (str, optional): The distance metric to use for ranking elements of the vocabulary. Defaults to "euclidean".

    Returns:
        np.ndarray: A two-dimensional numpy array containing the ids of the sampled replacements.
    """
    number_of_texts = texts_embeddings.shape[0]
    padded_number_of_tokens = texts_embeddings.shape[1]

    # Apply the fix for each text one-by-one.
    noisy_texts_ids = np.empty(
        (number_of_texts, padded_number_of_tokens),
        dtype=best_uint_type(vocabulary.shape[0]),
    )
    for i in range(number_of_texts):
        noisy_texts_ids[i] = dx_post_processing(
            texts_embeddings[i], vocabulary, dx_constant, epsilon, distance_metric
        )

    return noisy_texts_ids


def apply_post_processing_on_textsV2(
    texts_embeddings: np.ndarray,
    vocabulary: np.ndarray,
    attention_mask: np.ndarray,
    pad_token_id: int,
    dx_constant: float,
    epsilon: int,
    distance_metric: str = "euclidean",
) -> np.ndarray:
    """Applies the post-processing fix proposed in (Asghar et al., 2024) on texts_embeddings, text-by-text. This second version does not process pad tokens marked as such by the attention_mask and directly puts pad_token_id as their associated result.

    Args:
        texts_embeddings (np.ndarray): A three-dimensional array containing the embeddings we want to process.
        vocabulary (np.ndarray): A two-dimensional array containing all the embeddings of the vocabulary to compute the fix against.
        attention_mask (np.ndarray): A two-dimensional array of the same shape as texts_embeddings, where a 0 marks the position of a pad token.
        pad_token_id (int): The token id of a pad token (depends on the tokenizer)
        dx_constant (float): The c constant in the formula of (Asghar et al., 2024).
        epsilon (int): The epsilon value in the dx-privacy formula.
        distance_metric (str, optional): The distance metric to use for ranking elements of the vocabulary. Defaults to "euclidean".

    Returns:
        np.ndarray: A two-dimensional numpy array containing the ids of the sampled replacements.
    """
    number_of_texts = texts_embeddings.shape[0]
    padded_number_of_tokens = texts_embeddings.shape[1]

    # Declare the result as a two-dimensional array, consisting of pad tokens for now.
    noisy_texts_ids = np.full(
        (number_of_texts, padded_number_of_tokens),
        pad_token_id,
        dtype=best_uint_type(vocabulary.shape[0]),
    )

    # Apply the fix text-by-text for all text embeddings, excluding pad tokens.
    for i in range(number_of_texts):
        # Gather the indexes not containing pad tokens.
        indexes_to_be_computed = np.where(attention_mask[i] == 1)[0]
        first_index_to_be_computed = indexes_to_be_computed[0]
        last_index_to_be_computed = indexes_to_be_computed[-1] + 1

        noisy_texts_ids[i][first_index_to_be_computed:last_index_to_be_computed] = (
            dx_post_processing(
                texts_embeddings[i][
                    first_index_to_be_computed:last_index_to_be_computed
                ],
                vocabulary,
                dx_constant,
                epsilon,
                distance_metric,
            )
        )
    return noisy_texts_ids


def ids_to_texts(texts_ids: list[list[int]], tokenizer: AutoTokenizer) -> list[str]:
    # batch_decode does not exist for some tokenizers
    try:
        return tokenizer.batch_decode(texts_ids, skip_special_tokens=True)
    except NameError:
        return [tokenizer.decode(e, skip_special_tokens=True) for e in texts_ids]
