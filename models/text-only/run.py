import torch
import torch.nn as nn
from tqdm import tqdm
import click
from vl_bench.data import BaseDataset
from vl_bench.utils import process_path
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


@click.command()
@click.option(
    '--input-file',
    type=click.Path(exists=True, file_okay=True),
    required=True
)
@click.option(
    '--batch-size',
    type=int,
    default=128,
)
@click.option(
    '--device',
    type=str,
    default='cuda:0' if torch.cuda.is_available() else 'cpu',
)
def main(input_file, batch_size, device):
    # read data
    data = BaseDataset(input_file)
    max_foils = max([len(x['foils']) for x in data])
    texts = []
    for x in data:
        this = [x['caption']] + x['foils']
        if len(this) < max_foils + 1:  # padding for foils
            count = max_foils+1-len(this)
            this = this + x['foils'] + count * [x['foils'][0]]
        texts.extend(this)

    # initialize model & tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    crit = nn.CrossEntropyLoss(reduction='none')
    results = []
    for i in tqdm(range(0, len(texts), batch_size)):
        start, end = i, min(i + batch_size, len(texts))
        try:
            inputs = tokenizer(
                texts[start:end],
                return_tensors='pt',
                padding=True,
            )
        except:
            import ipdb; ipdb.set_trace()
        input_ids = inputs['input_ids'][:, 0:-1].to(device)
        attention_mask = inputs['attention_mask'][:, 0:-1].to(device)
        labels = inputs['input_ids'][:, 1:].to(device)

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = output.logits

        logits = logits.reshape(-1, logits.size(-1))
        scores = crit(logits, labels.view(-1))
        scores = scores.reshape_as(attention_mask) * attention_mask
        scores = scores.sum(dim=1, keepdim=True) 
        scores = scores / attention_mask.sum(dim=1, keepdim=True)
        scores = scores.reshape(-1).exp()
        results.append(scores)

    results = torch.cat(results, dim=0)
    results = results.reshape(len(data), -1)
    pred = results.argmin(dim=1)
    n_correct = torch.sum(pred == 0).item()
    accuracy = round(100 * n_correct / len(data), 2)
    print(f'Accuracy: {accuracy}')


if __name__ == "__main__":
    main()
