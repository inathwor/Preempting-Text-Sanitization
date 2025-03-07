# # My Regression

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
from os.path import join
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    KFold,
)
from sklearn.ensemble import HistGradientBoostingRegressor
import sys
from pathlib import Path

load_dotenv()
sys.path.append(str(Path(__file__).parent))  # Add parent directory to path
from utils.tools import load_pickle

# BEGIN PARAMETERS
embedding_models = ["bart", "llama"]
large_language_models = ["llama", "gemini"]
small_language_models = ["bart", "llama3.2-1B", "t5"]
scoring = {
    "r2": "r2",
    "neg_rmse": "neg_root_mean_squared_error",
}  # Scikitlearn scoring methods
# END PARAMETERS

load_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"], "datasets/multi_news/similarities"
)


def evaluate_regression_quality(data, target):
    model = make_pipeline(StandardScaler(), HistGradientBoostingRegressor())
    cv_results = cross_validate(
        model, data, target, cv=KFold(n_splits=5, shuffle=True), scoring=scoring
    )

    print(f"Mean R2: {np.mean(cv_results['test_r2']):.2f}")
    print(f"Mean RMSE: {np.mean(-cv_results['test_neg_rmse']):.2f}")

    # Measuring wasted $, wasted epsilon and failed predictions
    data_train, data_test, target_train, target_test = train_test_split(
        data, target, test_size=0.20, shuffle=True
    )
    model.fit(data_train, target_train)

    # Results for all features
    threshold = 0.1  # All target higher than prediction+threshold are considered failed predictions
    predictions = model.predict(data_test)

    # Count wrong predictions
    # 1. too_optimistic = the number of text for which we will loose money, i.e. those where the target cosine distance is higher than $prediction+threshold$
    # 2. too_pessimistic = the number of text for which we could have had better privacy, i.e. those where the target cosine distance is lower than $prediction-threshold$
    # 3. bad_predictions = too_optimistic & too_pessimistic
    too_optimistic = np.sum(target_test > predictions + threshold).item()
    too_pessimistic = np.sum(target_test < predictions - threshold).item()
    bad_predictions = np.sum(abs(predictions - target_test) > threshold).item()
    print(
        "too_optimistic",
        too_optimistic,
        "/",
        predictions.shape[0],
        "which is ",
        f"{too_optimistic*100/predictions.shape[0]:.0f}",
        "%",
    )
    print(
        "too_pessimistic",
        too_pessimistic,
        "/",
        predictions.shape[0],
        "which is ",
        f"{too_pessimistic*100/predictions.shape[0]:.0f}",
        "%",
    )
    print(
        "bad_predictions",
        bad_predictions,
        "/",
        predictions.shape[0],
        "which is ",
        f"{bad_predictions*100/predictions.shape[0]:.0f}",
        "%",
    )


def perform_regression(
    embedding_model: str,
    large_language_model: str,
    small_language_model: str,
    corrected_texts: bool,
):
    print(
        f"Embedding_model={embedding_model}, SLM={small_language_model}, LLM={large_language_model}, corrected_texts={corrected_texts}"
    )
    # Load SLM Data
    ogtext_slmgensum_sim = np.array(
        load_pickle(
            load_folderpath, f"OGtextsVS{small_language_model}Gensummary.pickle"
        )
    )

    ogtext_slmnoisygensum_sim = load_pickle(
        load_folderpath,
        f"OGtextVS{small_language_model}noisygenSummaryFrom{embedding_model}_embedding_model.pickle",
    )
    if corrected_texts:
        ogtext_slmnoisygensum_sim = load_pickle(
            load_folderpath,
            f"OGtextVS{small_language_model}noisygenSummaryFrom{embedding_model}_embedding_model_corrected.pickle",
        )

    # Load LLM Data
    ogtext_llmnoisygensum_sim = load_pickle(
        load_folderpath,
        f"OGtextVS{large_language_model}noisygenSummaryFrom{embedding_model}_embedding_model.pickle",
    )
    if corrected_texts:
        ogtext_llmnoisygensum_sim = load_pickle(
            load_folderpath,
            f"OGtextVS{large_language_model}noisygenSummaryFrom{embedding_model}_embedding_model_corrected.pickle",
        )

    # LM-agnostic data
    ogtext_noisytext_sim = load_pickle(
        load_folderpath,
        f"OGtextsVSnoisytextsFrom{embedding_model}_embedding_model.pickle",
    )
    if corrected_texts:
        ogtext_noisytext_sim = load_pickle(
            load_folderpath,
            f"OGtextsVSnoisytextsFrom{embedding_model}_embedding_model_corrected.pickle",
        )

    # Sanity check: Each dictionary should have the same epsilon values in the same order
    assert list(ogtext_slmnoisygensum_sim.keys()) == list(
        ogtext_llmnoisygensum_sim.keys()
    ) and list(ogtext_slmnoisygensum_sim.keys()) == list(ogtext_noisytext_sim.keys())
    nb_texts = ogtext_slmgensum_sim.shape[0]
    epsilons = list(ogtext_slmnoisygensum_sim.keys())
    nb_epsilons = len(epsilons)

    # Merging data into one Dataframe
    # We convert cosine similarities to cosine distances $\in [0,2]$ to avoid negative values.
    df2 = pd.DataFrame(
        {
            # Features
            "epsilon": np.repeat(epsilons, nb_texts),
            "ogtext_noisytext_sim": 1
            - np.array(list(ogtext_noisytext_sim.values())).flatten(),
            "ogtext_slmgensum_sim": 1 - np.tile(ogtext_slmgensum_sim, nb_epsilons),
            "ogtext_slmnoisygensum_sim": 1
            - np.array(list(ogtext_slmnoisygensum_sim.values())).flatten(),
            # Target
            "ogtext_llmnoisygensum_sim": 1
            - np.array(list(ogtext_llmnoisygensum_sim.values())).flatten(),
        }
    )

    target = df2["ogtext_llmnoisygensum_sim"]
    data = df2.drop(columns=["ogtext_llmnoisygensum_sim"])

    # ALL FEATURES SCORE
    print("All features:")
    evaluate_regression_quality(data, target)

    # BASELINE SCORE
    data_baseline = df2.drop(
        columns=[
            "ogtext_noisytext_sim",
            "ogtext_slmgensum_sim",
            "ogtext_slmnoisygensum_sim",
        ]
    )

    print()
    print("Baseline:")
    evaluate_regression_quality(data_baseline, target)
    print("\n######################\n")


for embedding_model in embedding_models:
    for large_language_model in large_language_models:
        for small_language_model in small_language_models:
            perform_regression(
                embedding_model,
                large_language_model,
                small_language_model,
                corrected_texts=False,
            )


# Additionnal code to handle corrected texts
# In the paper, only the following language models can be tested against corrected texts
correction_enabled_small_language_models = ["llama3.2-1B"]
correction_enabled_large_language_models = ["llama", "gemini"]
# Keep language models that are both correction_enabled and in the selected language models
for embedding_model in embedding_models:
    for large_language_model in set(correction_enabled_large_language_models) & set(
        large_language_models
    ):
        for small_language_model in set(correction_enabled_small_language_models) & set(
            small_language_models
        ):
            perform_regression(
                embedding_model,
                large_language_model,
                small_language_model,
                corrected_texts=True,
            )
