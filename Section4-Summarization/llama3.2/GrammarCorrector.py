from transformers import pipeline
import re
import torch
import os
from os.path import join
from dotenv import load_dotenv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path
from utils.tools import print_timed, save_pickle, load_pickle

load_dotenv()
# BEGIN PARAMETERS
embedding_model = "bart"  # "bart" or "llama", depending on the embedding model that was used to sanitize texts
cuda_device = "cuda"
batch_size = 48  # Modify according to your VRAM constraints
correction_prompt = "You are an expert in English text restoration. Your task is to correct any errors or inconsistencies in the provided text caused by token-level distortions. Ensure proper English grammar, coherence, and fidelity to the intended meaning. Make only the minimal necessary changes to restore the text to a natural and logical state. Respond with the fully restored text only, without any additional comments or explanations."
# END PARAMETERS

load_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    f"datasets/multi_news/noisy_texts/{embedding_model}_embedding_model/",
)
save_folderpath = join(
    os.environ["ROOT_SAVE_FOLDER"],
    f"datasets/multi_news/noisy_texts/{embedding_model}_embedding_model_corrected",
)

torch.set_default_device(cuda_device)
if not os.path.exists(save_folderpath):
    os.makedirs(save_folderpath)

print_timed("Loading Model")
pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    batch_size=batch_size,
    token=os.environ["HUGGFACE_ACCESS_TOKEN"],
    revision="9213176726f574b556790deb65791e0c5aa438b6",
)

pipe.tokenizer.padding_side = "left"
pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

for file in sorted(os.listdir(load_folderpath)):
    regexp_match = re.fullmatch(r"^epsi(\d*)full.pickle$", file)
    if not regexp_match:
        continue
    epsilon = int(regexp_match.group(1))
    print_timed(f"Epsilon={epsilon}")

    noisy_texts: list[str] = load_pickle(load_folderpath, f"epsi{epsilon}full.pickle")
    number_of_texts = len(noisy_texts)

    messages = [
        [
            {
                "role": "system",
                "content": correction_prompt,
            },
            {"role": "user", "content": text},
        ]
        for text in noisy_texts
    ]

    filenames_prefix = f"epsi{epsilon}"
    filenumber = 0
    for i in range(0, number_of_texts, batch_size):
        j = min(i + batch_size, number_of_texts)
        batch_messages = messages[i:j]

        print_timed(f"Epsi {epsilon}, Starting Batch [{i}:{j}]")

        outputs = pipe(
            batch_messages,
            max_new_tokens=1024,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )

        corrected_texts = [
            outputs[i][0]["generated_text"][-1]["content"] for i in range(len(outputs))
        ]

        save_pickle(
            save_folderpath,
            f"{filenames_prefix}part_{filenumber:04d}.pickle",
            corrected_texts,
            False,
        )
        filenumber += 1

    corrected_texts = []
    for file in sorted(os.listdir(save_folderpath)):
        if re.fullmatch(f"^{filenames_prefix}part.*", file):
            corrected_texts += load_pickle(save_folderpath, file)
            os.remove(join(save_folderpath, file))

    # Save summaries to disk
    save_pickle(
        save_folderpath, f"{filenames_prefix}full.pickle", corrected_texts, False
    )
