#!/usr/bin/env python
# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time
import torch
import transformers
from torch.utils.data import SequentialSampler, DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm


def processOutput(output_strings):
    output_numbers = []
    for idx, output_string in enumerate(output_strings):
        try:
            fraction = output_string.split("=")[-1].strip()
            fraction = fraction.split('!')[0]
            number = fraction
            number = int(number)
        except:
            number = 0
        output_numbers.append(number)
    return output_numbers


def generateAnswer(inputs, batch_size=20):
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    output_strings = []
    for batch in tqdm(dataloader):
        batch = tuple(t.to(device) for t in batch)
        batch_input_ids, batch_masks = batch
        output = model.generate(input_ids=batch_input_ids, pad_token_id=50256, do_sample=False, max_length=150)
        output.to(torch.device("cpu"))
        for output_example in output:
            string_out = tokenizer.decode(output_example, skip_special_tokens=True)
            output_strings.append(string_out)
    return output_strings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--testset_path', type=str)
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-g', '--gpt2_path', type=str)
    parser.add_argument('-p', '--pipeline', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = transformers.AutoModelWithLMHead.from_pretrained(args.model_path).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.gpt2_path)
    tokenizer.pad_token = tokenizer.eos_token
    right_cnt = 0
    input_prompts = []
    correct_results = []
    tic = time.time()
    cnt = 0
    with open(args.testset_path, "r") as infile:
        line = infile.readline().strip("\n")
        while line:
            splitted = line.split("=")
            input_prompt = splitted[0].strip() + ". "
            if not args.pipeline:
                input_prompt = input_prompt.replace("with pipeline ", "")
            input_prompts.append(input_prompt)
            correct_results.append(int(splitted[1].strip()))
            line = infile.readline()
            cnt += 1

    inputs = tokenizer.batch_encode_plus(
        input_prompts, add_special_tokens=False, return_tensors="pt", pad_to_max_length=True,
        truncation=True)
    outputs = generateAnswer(inputs, batch_size=1)
    results = processOutput(outputs)
    for idx in range(len(results)):
        if results[idx] == correct_results[idx]:
            right_cnt += 1
    accuracy = right_cnt / len(results)
    print("Elapsed time: {}".format(time.time() - tic))
    print("Accuracy score: {}".format(accuracy))


