import json
import torch
import torch.nn as nn
from tqdm import tqdm
import click
from vl_bench.data import BaseDataset
from vl_bench.utils import process_path
from transformers import GPT2LMHeadModel, OPTForCausalLM
from transformers import AutoTokenizer


SUPPORTED_MODELS = (
    'gpt2',
    'facebook/opt-2.7b',
    'facebook/opt-6.7b',
)


def get_model_type(model_name):
    if model_name.startswith('gpt2'):
        return 'gpt2'
    if model_name.startswith('facebook/opt-'):
        return 'opt'


@click.command()
@click.option(
    '-i', '--input-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '-o', '--output-file',
    type=click.Path(file_okay=True),
    required=True,
)
@click.option(
    '--model-name',
    type=click.Choice(choices=SUPPORTED_MODELS),
    default='gpt2',
    show_default=True,
)
@click.option(
    '--batch-size',
    type=int,
    default=16,
    show_default=True,
)
@click.option(
    '--device',
    type=str,
    default='cuda:0' if torch.cuda.is_available() else 'cpu',
)
@click.option(
    '--proficiency',
    is_flag=True,
)
def main(input_file, output_file, model_name, batch_size, device, proficiency):
    # read data
    data = BaseDataset(input_file)
    ids, texts, num_texts = list(), list(), list()
    for i, x in enumerate(data):
        ids.append(x['item_id'])
        _x = x if not proficiency else x['proficiency']
        this = [_x['caption']] + _x['foils']
        texts.extend(this)
        num_texts.append(len(this))

    # initialize model & tokenizer
    dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_type = get_model_type(model_name)
    if model_type == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name)
    elif model_type == 'opt':
        model = OPTForCausalLM.from_pretrained(model_name)
    model = model.to(device, dtype)
    crit = nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
    all_scores = []
    for i in tqdm(range(0, len(texts), batch_size)):
        start, end = i, min(i + batch_size, len(texts))
        batch_texts = texts[start:end]
        inputs = tokenizer(
            text=batch_texts,
            text_target=batch_texts,
            return_tensors='pt',
            padding=True,
        ).to(device)

        with torch.no_grad():
            output = model(**inputs)
            logits = output.logits

        logits = logits[:, :-1, :]
        logits = logits.reshape(-1, logits.size(-1))
        labels = inputs['labels'][:, 1:].contiguous()
        lengths = inputs['attention_mask'].sum(dim=-1)
        scores = crit(logits, labels.view(-1))
        scores = scores.reshape_as(labels)
        scores = scores.sum(dim=1) / lengths
        scores = scores.reshape(-1).exp()
        all_scores.append(scores)

    all_scores = torch.cat(all_scores, dim=0).tolist()
    results = dict()
    offset = 0
    for item_id, item_num_texts in zip(ids, num_texts):
        start = offset
        end = offset + item_num_texts
        item_scores = all_scores[start:end]
        results[item_id] = {'scores': item_scores}
        offset = end

    output_file = process_path(output_file)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print('done.')


if __name__ == "__main__":
    main()
