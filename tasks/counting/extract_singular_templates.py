import os.path as osp
import json
import click
import spacy
from tqdm import tqdm
from vl_bench.utils import process_path


DATA_DIR = osp.abspath(osp.join(__file__, "../../..", "data"))
INPUT_FILE = osp.join(DATA_DIR, 'quva-templates.json')
OUTPUT_FILE = osp.join(DATA_DIR, 'quva-templates-processed.json')


def make_count_singular(doc, number=42):
    sentence, plural, prev = '', None, None
    for token in doc:
        this = token.text
        if token.text == str(number):
            this = '<number>'
            plural = token.head
        if plural is not None and token.idx == plural.idx:
            this = token.lemma_
        
        if token.pos_ == 'PUNCT' or prev == '-' or prev is None:
            sentence += this
        else:
            sentence += f' {this}'
        prev = this
    return sentence.strip()


@click.command()
@click.option(
    '--input-file',
    type=click.Path(exists=True, file_okay=True),
    default=INPUT_FILE,
)
@click.option(
    '--output-file',
    type=click.Path(exists=False, file_okay=True),
    default=OUTPUT_FILE,
)
def main(input_file, output_file):
    with open(process_path(input_file), 'r') as f:
        data = json.load(f)

    nlp = spacy.load('en_core_web_sm')
    number = '42'
    for item_id in tqdm(data.keys()):
        singular_templates = []
        for template in data[item_id]['templates']:
            plural_sentence = template.replace('<number>', number)
            doc = nlp(plural_sentence)
            singular_template = make_count_singular(doc, number=number)
            singular_templates.append(singular_template)
        data[item_id]['singular_templates'] = singular_templates
    
    with open(process_path(output_file), 'w') as f:
        json.dump(data, f, indent=4)
 

if __name__ == "__main__":
    main()