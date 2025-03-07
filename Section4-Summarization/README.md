# Summarization Experiments

## Code Overview
- Each language model involved has a dedicated folder, namely bart, gemini, llama3, llama3.2 and t5. 
- The bart folder contains SampleSanitization.ipynb which was used for the text sanitization example in Section 3 of the paper.
- The *utils* folder contains important functions used in many scripts e.g., $d_X$-privacy mechanism.
- At the top of all scripts there is a "BEGIN PARAMETERS" section where the main parameters can be configured.


## How to Run
First, follow the README.md of the parent folder to install the conda environment and setup environment variables.
### Pre-processing
Run the `Dataset_preprocessing.py` script to download and prepare the Multi News dataset.
### Summarization
1. Run the `OGSummarization.py` from every language model folder to perform the summarization on original (unaltered) texts.
2. Run `TextSanitization.py` from the bart and llama3 folders to sanitize texts with their respective embedding model.
3. Run the `GrammarCorrector.py` from llama3.2 folder to perform text correction. Modify the *embedding_model* variable in the script to run either on bart's embedding model or llama3's.
4. Run `noisyTextsSummarization.py` from every language model folder to perform summarization on sanitized texts. Modify the *embedding_model* variable in each script to run either on bart's embedding model or llama3's. Additionnaly, modify the *corrected_texts* variable to run on corrected texts or not (only available for llama3, llama3.2 and gemini).

### Similarity computations
Run `Similarities.py` to compute the similarities mentionned in the paper.

### Regression
Use `Regression.py` to perform the Regression as mentionned in Section 4.4 of the paper.