import math
import json
import click
from bokeh.plotting import figure, show
from vl_bench.utils import process_path


@click.command()
@click.option(
    '-i', '--input-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '--captions/--foils',
    default=True,
)
@click.option(
    '--mode',
    type=click.Choice(choices=['verb', 'noun', 'both']),
    default='both',
    show_default=True,
)
@click.option(
    '--top-k',
    type=int,
    default=64,
    show_default=True,
)
def main(input_file, captions, mode, top_k):
    input_file = process_path(input_file)
    captions_or_foils = captions
    use_captions = captions_or_foils
    use_foils = not captions_or_foils
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    counts = dict()
    for key, item in data.items():
        if mode == 'verb' and use_captions:
            this = item['verb']
        elif mode == 'verb' and use_foils:
            # count[item['verb']] = 1 + count.get(item['verb'], 0)
            this = item['foil_classes'][-1]  # FIXME
        elif mode == 'noun' and use_captions:
            this = item['noun']
        elif mode == 'noun' and use_foils:
            this = item['foil_classes'][0]  # FIXME
        elif mode == 'both' and use_captions:
            this = item['verb'] + ' ' + item['noun']
        elif mode == 'both' and use_foils:
            this1 = item['verb'] + ' ' + item['foil_classes'][0]
            this2 = item['foil_classes'][-1] + ' ' + item['noun']
            counts[this1] = counts.get(this1, 0) + 1
            counts[this2] = counts.get(this2, 0) + 1

        if not (mode == 'both' and use_foils):
            counts[this] = 1 + counts.get(this, 0)

    items = list(counts.items())
    items = sorted(items, key=lambda x: x[1], reverse=True)[:top_k]
    keys = [x[0] for x in items]
    values = [x[1] for x in items]

    p = figure(
        x_range=keys, width=960, height=640,
        title=f"use_captions={use_captions}, mode={mode}",)
    p.vbar(x=keys, top=values, width=0.8)
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    # p.xaxis.major_label_orientation = "vertical"
    p.xaxis.major_label_orientation = math.pi/2
    show(p)       


if __name__ == "__main__":
    main()