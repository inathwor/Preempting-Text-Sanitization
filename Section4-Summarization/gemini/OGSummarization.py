from datasets import load_from_disk
from datetime import datetime
import numpy as np
import time
import re
import os
from os.path import join
from dotenv import load_dotenv
from google import genai
from google.genai import types
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path
from utils.tools import print_timed, load_pickle, save_pickle

load_dotenv()

# BEGIN PARAMETERS
batch_size = 150  # Number of texts per minute if rate_limited is true
rate_limited = False
summarize_prompt = "You are a highly skilled text summarizer. Your task is to generate concise and accurate summaries of input text. Provide only the summary itself, without any remarks, explanations, or formatting cues."
# END PARAMETERS

save_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    "datasets/multi_news/gemini_generated_summaries/On_original_texts",
)
if not os.path.exists(save_folderpath):
    os.makedirs(save_folderpath)

texts = (
    load_from_disk(
        join(
            os.environ["ROOT_SAVE_FOLDER"],
            "datasets/multi_news/concatenated_clean_1024_tokens",
        )
    )
)["document"]

# Select the 1500 randomly sampled texts
texts_indexes_to_compute = np.load(
    join(
        os.environ["ROOT_SAVE_FOLDER"],
        "datasets/multi_news/texts_indexes_to_compute.npy",
    )
)
texts = (np.array(texts)[texts_indexes_to_compute]).tolist()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

n = len(texts)
filenumber = 0
print_milestone = 10  # The next percentage milestone to report
for i in range(0, n, batch_size):
    j = min(i + batch_size, n)

    texts_slice = texts[i:j]

    generated_summaries = [
        client.models.generate_content(
            model="gemini-2.0-flash",
            contents=texts_slice[k],
            config=types.GenerateContentConfig(
                max_output_tokens=142, system_instruction=summarize_prompt
            ),
        ).text
        for k in range(len(texts_slice))
    ]

    # Save summaries to disk
    save_pickle(
        save_folderpath, f"filepart_{filenumber:04d}.pickle", generated_summaries, False
    )
    filenumber += 1

    # Print progression
    percent_done = (j / n) * 100
    if percent_done >= print_milestone:  # If we passed a milestone
        print_timed(f"Processed {int(percent_done)}% ({j}/{n})")
        print_milestone += 10  # Move to the next milestone

    if rate_limited:
        # 60s delay for Google API
        time.sleep(60)

generated_summaries = []
for file in sorted(os.listdir(save_folderpath)):
    if re.fullmatch(f"^filepart_.*.pickle", file):
        generated_summaries += load_pickle(save_folderpath, file)
        os.remove(join(save_folderpath, file))

save_pickle(save_folderpath, "full.pickle", generated_summaries, False)
