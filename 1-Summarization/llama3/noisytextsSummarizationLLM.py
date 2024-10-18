from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import pickle
import torch
import re
import os
from os.path import join
import sys
# Add the main directory to sys.path to be able to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR, HUGGINGFACE_TOKEN

# PARAMS
load_folderpath = join(ROOT_DIR, "datasets/multi_news/noisy_texts/AsStrings/")
save_folderpath = join(ROOT_DIR, "datasets/multi_news/llama_generated_summaries/OnSanitizedTexts/")
epsilons = [1] + [i for i in range(5, 101, 5)]
cuda_device = "cuda"
texts_per_slice = 7
# END PARAMS
if not HUGGINGFACE_TOKEN:
    print("Please request access to llama3 model on HuggingFace and paste your account's access token in the config.py file.")
    sys.exit()

torch.set_default_device(cuda_device)
if not os.path.exists(save_folderpath):
    os.makedirs(save_folderpath)

llama3_hf_name = "meta-llama/Meta-Llama-3-8B-Instruct"

llama_tokenizer = AutoTokenizer.from_pretrained(
    llama3_hf_name,
    padding_side="left",
    revision = "e5e23bbe8e749ef0efcf16cad411a7d23bd23298",
    token=HUGGINGFACE_TOKEN
)
llama_model = AutoModelForCausalLM.from_pretrained(
    llama3_hf_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    revision = "e5e23bbe8e749ef0efcf16cad411a7d23bd23298",
    token=HUGGINGFACE_TOKEN
)

# Config tokenizer according to Llama example
llama_tokenizer.pad_token = llama_tokenizer.eos_token
terminators = [
    llama_tokenizer.eos_token_id,
    llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Set pad_token_id myself to avoid Llama warning:
# "Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation."
llama_model.generation_config.pad_token_id = llama_tokenizer.pad_token_id

for epsilon in epsilons:
    print(datetime.now().strftime('%Hh%Mm%Ss'), "Epsilon=", str(epsilon))

    with open(os.path.join(load_folderpath, "epsi"+str(epsilon)+"full.pickle"), 'rb') as f:
        texts = pickle.load(f)

    n = len(texts)

    filenames_prefix = "epsi"+str(epsilon)
    filenumber = 0
    for i in range(0, n, texts_per_slice):
        if i%(texts_per_slice*100) == 0:
            print(str(i), " processed.")
        j = min(i+texts_per_slice, n)
        texts_slice = texts[i:j]


        torch.cuda.empty_cache()
        messages = [
            [
                {"role": "system", "content": "You are put into the role of a text summarizer. Your task is to summarize your input text. When given a collection of random words, invent a news summary based on the words you understand. You must not break character and thus must only output the resulting summary without introductory statement."},
                {"role": "user", "content": text}
            ]
            for text in texts_slice
        ]

        input_ids = llama_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(cuda_device)

        with torch.no_grad():
            outputs = llama_model.generate(
                **input_ids,
                max_new_tokens=142,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
        
        outputs = outputs[:, input_ids["input_ids"].shape[-1]:]
        generated_summaries = llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Save summaries to disk
        filename = filenames_prefix+f"part_{filenumber:04d}"+".pickle"
        filepath = os.path.join(save_folderpath, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(generated_summaries, f)
        filenumber += 1

    generated_summaries = []
    for file in sorted(os.listdir(save_folderpath)):
        if re.fullmatch("^"+filenames_prefix+"part.*", file):
            with open(save_folderpath+file, 'rb') as f:
                generated_summaries += pickle.load(f)

    with open(save_folderpath+filenames_prefix+"full.pickle", 'wb') as f:
        pickle.dump(generated_summaries, f)