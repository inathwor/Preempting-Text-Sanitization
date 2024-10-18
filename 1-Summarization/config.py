# config.py
import os
ROOT_DIR = os.path.expanduser("~/data-preempting") # Replace value with absolute path to writable folder

# To perform the experiments on Llama3, its access on huggingface must me requested:
# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/
# Then, write the account's access token here
HUGGINGFACE_TOKEN = ""