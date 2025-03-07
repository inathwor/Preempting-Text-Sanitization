from datetime import datetime
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
embedding_model = "bart"  # "bart" or "llama", depending on the embedding model that was used to sanitize texts
corrected_texts = False  # Compute on correct texts or not
batch_size = 150  # Number of texts per minute if rate_limited is true
rate_limited = False
summarize_prompt = "You are a highly skilled text summarizer. Your task is to generate concise and accurate summaries of input text. Provide only the summary itself, without any remarks, explanations, or formatting cues."
# END PARAMETERS

load_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    f"datasets/multi_news/noisy_texts/{embedding_model}_embedding_model",
)
save_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    f"datasets/multi_news/gemini_generated_summaries/On_noisy_texts_with_{embedding_model}_embedding_model",
)

if corrected_texts:
    load_folderpath = join(
        os.environ["ROOT_SAVE_FOLDER"],
        f"datasets/multi_news/noisy_texts/{embedding_model}_embedding_model_corrected",
    )
    save_folderpath = join(
        os.environ["ROOT_SAVE_FOLDER"],
        f"datasets/multi_news/gemini_generated_summaries/On_noisy_texts_with_{embedding_model}_embedding_model_corrected",
    )

if not os.path.exists(save_folderpath):
    os.makedirs(save_folderpath)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

for file in sorted(os.listdir(load_folderpath)):
    regexp_match = re.fullmatch(r"^epsi(\d*)full.pickle$", file)
    if not regexp_match:
        continue
    epsilon = int(regexp_match.group(1))

    texts: list[str] = load_pickle(load_folderpath, f"epsi{epsilon}full.pickle")
    n = len(texts)

    filenames_prefix = f"epsi{epsilon}"
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
        filename = f"{filenames_prefix}part_{filenumber:04d}.pickle"
        save_pickle(save_folderpath, filename, generated_summaries, False)
        filenumber += 1

        percent_done = (j / n) * 100
        if percent_done >= print_milestone:  # If we passed a milestone
            print_timed(f"Epsilon={epsilon} ; Processed {int(percent_done)}% ({j}/{n})")
            print_milestone += 10  # Move to the next milestone

        if rate_limited:
            # 60s delay for Google API
            time.sleep(60)

    generated_summaries = []
    for file in sorted(os.listdir(save_folderpath)):
        if re.fullmatch(f"^{filenames_prefix}part.*", file):
            generated_summaries += load_pickle(save_folderpath, file)
            os.remove(join(save_folderpath, file))

    save_pickle(
        save_folderpath, f"{filenames_prefix}full.pickle", generated_summaries, False
    )

    if rate_limited:
        # 1 day delay for Google API
        print_timed("Sleeping for a day ...")
        time.sleep(86400)
