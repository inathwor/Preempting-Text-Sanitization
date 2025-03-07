# $d_\mathcal{X}$-privacy experiments
## Setup
1. Follow the README.md of the parent folder to install the conda environment and setup environment variables.
2. In the folder you defined as your ROOT_SAVE_FOLDER above, put the following two files:
    - glove.6B.300d.txt extracted from glove.6B.zip downloaded from https://nlp.stanford.edu/projects/glove/
    - wiki.en.vec extracted from "English: bin+text" downloaded from https://fasttext.cc/docs/en/pretrained-vectors.html
3. Run `PreProcessing.py` to process the downloaded files.

## Running experiments
- Run `experiment1.py` for the results of Section 5.1. This script notably reuses code from [awslabs/sagemaker-privacy-for-nlp](https://github.com/awslabs/sagemaker-privacy-for-nlp) in an attempt to be as close as possible to the original paper. Results will be written to stdout.

- `experiment2-ANN.py` contains the first part of the code for Section 5.2. It also reuses the same code mentioned above to carry out the Approximate Nearest Neighbor plot of Figure 4. Results are written to stdout as well as being saved in ROOT_SAVE_FOLDER/emb_space_analysis/.

- `experiment2-ENN.py` contains the second part of the code for Section 5.2. For this experiment, we cannot reuse the aforementioned code as it is too slow for Exact Nearest Neighor Search. Instead, it relies on our optimized version leveraging GPU, defined in Section4-Summarization/utils/dx.py of this repository. Results are written to stdout as well as being saved in ROOT_SAVE_FOLDER/emb_space_analysis/.

Please refer to our paper for details about what each experiment is computing.