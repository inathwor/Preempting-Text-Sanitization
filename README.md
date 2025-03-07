# Preempting-Text-Sanitization
This repository contains the codebase for the experiments of the paper titled "Preempting Text Sanitization Utility in Resource-Constrained Privacy-Preserving LLM Interactions".

## How to use
### Requirements
Most scripts require a CUDA driver enabled for summarization with Language Models (transformers python package) as well as $d_X$-privacy mechanism leveraging GPU for fast neighbor ranking in n-dimensional vector spaces (CuPy python package).

### Setting up
First, create a `.env` file in the same directory as this README based on the following model:
```
ROOT_SAVE_FOLDER=
HUGGFACE_ACCESS_TOKEN=
GEMINI_API_KEY=
```
- ROOT_SAVE_FOLDER must be filled with the absolute path of an existing folder where the scripts will write their data.
- HUGGFACE_ACCESS_TOKEN must be filled with your access token from HuggingFace platform in order to execute experiments with Llama models. Access to these models must be requested on their respective page: [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
- GEMINI_API_KEY must be filled with your access token from Google Cloud in order to execute experiments with Gemini.

Also, you need to install the conda environment from the file preempting_env.yml. Please use:
1. `conda env create -n preempting_env --file preempting_env.yml`
2. `conda activate preempting_env`

### Run the code
1. The folder "Section4-Summarization" contains the summarization experiments of the Section 4 of the paper.
2. The folder "Section5-dxPrivacy" contains the code for Section 5 about replicating experiments from (Feyisetan et al., 2020).

Please refer to the README.md of each folder for further information.

## References in the code
- (Asghar et al., 2024): H. J. Asghar, R. Carpentier, B. Z. H. Zhao, and D. Kaafar, "dX-privacy for text and the curse of dimensionality." 2024. [Online]. Available: https://arxiv.org/abs/2411.13784

- (Feyisetan et al., 2020): O. Feyisetan, B. Balle, T. Drake, and T. Diethe, "Privacy- and utility-preserving textual analysis via calibrated multivariate perturbations," in Proceedings of the 13th international conference on web search and data mining, in WSDM ’20. New York, NY, USA: Association for Computing Machinery, 2020, pp. 178–186. doi: 10.1145/3336191.3371856.

- (Qu et al., 2021): C. Qu, W. Kong, L. Yang, M. Zhang, M. Bendersky, and M. Najork, "Natural language understanding with privacy-preserving BERT," in Proceedings of the 30th ACM international conference on information & knowledge management, in CIKM ’21. New York, NY, USA: Association for Computing Machinery, 2021, pp. 1488–1497. doi: 10.1145/3459637.3482281.