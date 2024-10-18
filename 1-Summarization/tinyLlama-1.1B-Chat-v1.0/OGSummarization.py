from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from datetime import datetime
import pickle
import torch
import os
from os.path import join
import sys
# Add the main directory to sys.path to be able to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR

# PARAMETERS
default_save_path = join(ROOT_DIR, "datasets/multi_news/tinyllamachat_generated_summaries/OnOriginalTexts/")
cuda_device = "cuda"
# END PARAMETERS

torch.set_default_device(cuda_device)
if not os.path.exists(default_save_path):
    os.makedirs(default_save_path)

print(datetime.now().strftime('%Hh%Mm%Ss'), "Started")

texts = (load_from_disk(join(ROOT_DIR, "datasets/multi_news/concatenated_clean_1024_tokens")))['document']

# Setting batch_size produces awful results. Using (very slow) text by text inference.
summarizer = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
messages = [
    [
        {"role": "system", "content": "Your task is to summarize your input text."},
        {"role": "user", "content": text}
    ]
    for text in texts
]
prompt = summarizer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = summarizer(prompt, max_new_tokens=142, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, padding=True, truncation=True)
generated_summaries = [outputs[i][0]['generated_text'].split("<|assistant|>\n")[1] for i in range(len(outputs))]

# Save summaries to disk
filepath = default_save_path+"full.pickle"
with open(filepath, 'wb') as f:
    pickle.dump(generated_summaries, f)