# Summarization Experiments
## Requirements
The code in this repository requires the machine to have a CUDA driver enabled. 

## How to use

### Pre-processing
1. Set up the python environment using `conda env create -n summarization --file summarization.yml`
2. Make a dedicated folder to host the data (write access needed)
3. Write the absolute path of the chosen folder in config.py, noted `ROOT_DIR` hereafter.
4. Use `data.ipynb` to load and pre-process the Multi News dataset

A folder `ROOT_DIR/datasets/multi_news/concatenated_clean_1024_tokens` will be created and will contain the computing dataset. 
### Summarization
For each model (bart, llama3, t5 and tinyLlama) there is a subfolder containing two scripts:
- `OGSummarization.py` which summarizes the original texts of the Multi News Dataset
- `noisytextsSummarization.py` which summarizes the sanitized version of the same texts.

The *bart* folder has an additional script to perform the text sanitization.

Processing steps:
1. Run `bart/TextSanitization.py`
2. For each model, run both `OGSummarization.py` and `noisytextsSummarization.py`.

The summarized texts will be located in subfolders of `ROOT_DIR/datasets/multi_news/` such as *bart_generated_summaries*.

### Similarity computations
Use the `Similarities.ipynb` notebook to compute the similarities which will be saved in `ROOT_DIR/datasets/multi_news/similarities`.

### Regression
Use the `Regression.ipynb` notebook to perform the Regression as mentionned in Section 4.4 of the paper.