# Evaluating Transformer Language Models on Arithmetic Operations Using Number Decomposition

This is the repository associated to the paper **Evaluating Transformer Language Models on Arithmetic Operations Using Number Decomposition**.
Here you can find the data used to fine-tune and evaluate Language Models
and the code to reproduce the experiments.

## Setup
In order to use this repo, you first need to install the requirements.
Simply do via command line: 

    pip install -r requirements.txt

This repo leverages the Huggingface's Transformers library and uses PyTorch as backend.
## Data
The `data/train_sets` folder contains all training sets used to fine-tune Language
Models in our experiments, one for each operation-approach combination.
The `data/test_sets` folder contains all test sets on which the fine-tuned Language
Models have been evaluated, one for each task studied. These test sets have been taken from the
[GPT-3 github repository](https://github.com/openai/gpt-3).

## How to use
To fine-tune Language Models to perform arithmetic operations using the approaches
described in the paper, use the `scripts/training/training_script.py` script.
The latter is taken from the Huggingface's Transformers repository with some
modifications. To use this script, simply run from the root folder:

    python3 scripts/training/training_script.py \ 
    --model_name_or_path gpt2 --model_type gpt2 \
    --train_file data/train_sets/<train_set_name.txt> \
    --output_dir <path_to_output_dir> \
    --num_train_epochs <number_of_training_epochs> --line_by_line True \
    --do_train --overwrite_output_dir

By selecting a training set in `data/train_sets`, you will fine-tune a GPT-2 model
designed to perform a specific arithmetic operation with a specific approach.

To evaluate a fine-tuned Language Model, simply run:

    python3 scripts/evaluation/evaluation_script.py -g gpt2 \
    -m <path_to_fine_tuned_model> -t <path_to_test_set>

If you are evaluating a model fine-tuned with the pipeline approach, please add the
`-p` option to the command above. This script will output the accuracy score obtained by
the fine-tuned model on the selected task.
