# Evaluating Transformer Language Models on Arithmetic Operations Using Number Decomposition

This is the repository associated to the paper **Evaluating Transformer Language Models on Arithmetic Operations Using Number Decomposition**.
Here you can find the data used to fine-tune and evaluate Language Models
and the code to reproduce the experiments.

## Setup
In order to use this repo, you first need to install the requirements.
Simply do via command line: 

    pip install -r requirements.txt

This repo leverages the Huggingface's Transformers library and uses PyTorch as backend.

Since the GPT-2 Language Model does not include a padding-token in its vocabulary, 
we provide a `tokenizer_config.json` file which includes one.
To use the aforementioned file, simply download in the `models/gpt2` folder
the pre-trained GPT-2 model from the Huggingface Models portal:

    cd models/gpt2
    wget https://huggingface.co/gpt2/resolve/main/config.json
    wget https://huggingface.co/gpt2/resolve/main/pytorch_model.bin
    wget https://huggingface.co/gpt2/resolve/main/tokenizer.json
    wget https://huggingface.co/gpt2/resolve/main/vocab.json

To conclude, you may need to setup the config file for the `accelerate` package.
You can do it by running:

    accelerate config

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

    accelerate launch scripts/training/run_clm_no_trainer.py \ 
    --model_name_or_path models/gpt2 --model_type gpt2 \
    --train_file data/train_sets/<train_set_name.txt> \
    --output_dir <path_to_output_dir>  --pad_to_max_length \
    --num_train_epochs <number_of_training_epochs> --line_by_line True \
    --validation_file data/test_sets/<test_set_name.txt>

By selecting a training set in `data/train_sets`, you will fine-tune a GPT-2 model
designed to perform a specific arithmetic operation with a specific approach.

To evaluate a fine-tuned Language Model, simply run:

    python3 scripts/evaluation/evaluation_script.py -g models/gpt2 \
    -m <path_to_fine_tuned_model> -t <path_to_test_set>

If you are evaluating a model fine-tuned with the pipeline approach, please add the
`-p` option to the command above. This script will output the accuracy score obtained by
the fine-tuned model on the selected task.
