import numpy as np
from os.path import join
import os
from dotenv import load_dotenv
import sys
import sys
from pathlib import Path

# Use utils package from other folder
sys.path.append(join(str(Path(__file__).parent.parent), "Section4-Summarization"))
from utils.tools import save_pickle, print_timed

load_dotenv()
glove_data_folderpath = os.environ["ROOT_SAVE_FOLDER"]
fasttext_data_folderpath = os.environ["ROOT_SAVE_FOLDER"]

# This script processes GloVe and FastText text files into appropriate pickle files


# Function taken from https://stackoverflow.com/a/38230349
def load_glove_model(File):
    print("Loading Glove Model")
    glove_model = {}
    with open(File, "r") as f:
        for line in f:
            split_line = line.split(" ")
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model


print_timed("Processing Glove")
glove = load_glove_model(join(glove_data_folderpath, "glove.6B.300d.txt"))
save_pickle(glove_data_folderpath, "glove.6B.300d.pkl", glove)

print_timed("Processing FastText")
# Function taken from https://fasttext.cc/docs/en/english-vectors.html
with open(
    join(fasttext_data_folderpath, "wiki.en.vec"),
    "r",
    encoding="utf-8",
    newline="\n",
    errors="ignore",
) as f:
    n, d = map(int, f.readline().split())
    fasttext = {}
    for line in f:
        tokens = line.rstrip().split(" ")
        fasttext[tokens[0]] = list(map(float, tokens[1:]))

save_pickle(fasttext_data_folderpath, "wiki.en.pkl", fasttext)
