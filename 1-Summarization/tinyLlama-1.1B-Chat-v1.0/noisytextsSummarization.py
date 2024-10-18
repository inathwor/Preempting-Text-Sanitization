from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from datetime import datetime
import pickle
import torch
import os
from os.path import join
import re
import sys
# Add the main directory to sys.path to be able to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR

# PARAMS
load_folderpath = join(ROOT_DIR, "datasets/multi_news/noisy_texts/AsStrings/")
save_folderpath = join(ROOT_DIR, "datasets/multi_news/tinyllamachat_generated_summaries/OnSanitizedTexts/")
epsilons = [1] + [i for i in range(5, 101, 5)]
cuda_device = "cuda"
# END PARAMS
torch.set_default_device(cuda_device)

if not os.path.exists(save_folderpath):
    os.makedirs(save_folderpath)

print(datetime.now().strftime('%Hh%Mm%Ss'), "Started")
summarizer = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    torch_dtype=torch.bfloat16,
    device_map="auto"
) # Setting batch_size produces awful results. Using (very slow) text by text inference.

for epsilon in epsilons:
    torch.cuda.empty_cache()
    print(datetime.now().strftime('%Hh%Mm%Ss'), "Epsilon=", str(epsilon))

    with open(join(load_folderpath, "epsi"+str(epsilon)+"full.pickle"), 'rb') as f:
        noisy_texts = pickle.load(f)

    messages = [
        [
            {"role": "system", "content": "Your task is to summarize your input text."},
            {"role": "user", "content": text}
        ]
        for text in noisy_texts
    ]
    prompt = summarizer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = summarizer(prompt, max_new_tokens=142, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, padding=True, truncation=True)
    generated_summaries = [outputs[i][0]['generated_text'].split("<|assistant|>\n")[1] for i in range(len(outputs))]

    # Save summaries to disk
    filepath = join(save_folderpath, f"epsi{epsilon}full.pickle")
    with open(filepath, 'wb') as f:
        pickle.dump(generated_summaries, f)
