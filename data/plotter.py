import json
import click
from plotter_helper import create_plot

def add_to_dict(verb_dict, verb, is_caption):
    if type(verb) != str:
        raise ValueError("Sample value must be in string format!")
    
    count_tuple = (0, 0)

    if verb in verb_dict.keys():
        count_tuple = verb_dict[verb]

    if is_caption:
        count_tuple = (count_tuple[0] + 1, count_tuple[1])
    else:
        count_tuple = (count_tuple[0], count_tuple[1] + 1)
    
    verb_dict[verb] = count_tuple

CHART_TYPES = ("bar3d", "bar2d", "line")

@click.command()
@click.option(
    '-i', '--input-file',
    type=click.Path(exists=True, file_okay=True),
    required=True
)
@click.option(
    '-p', '--is-prof',
    type=bool,
    required=True,
    default=False,
)
@click.option(
    '-ctype', '--chart-type',
    type=click.Choice(choices=CHART_TYPES),
    default=CHART_TYPES[0],
)
@click.option(
    '-ctype', '--chart-title',
    type=str,
    default= "Action Foil Distributions",
)
@click.option(
    '-xtitle', '--x-axis-title',
    type=str,
    default= "Words",
)
@click.option(
    '-ytitle', '--y-axis-title',
    type=str,
    default= "Count",
)
@click.option(
    '-o', '--output-file',
    type=click.Path(file_okay=True),
    required=True,
)
def main(
    input_file,
    is_prof,
    chart_type,
    chart_title,
    x_axis_title,
    y_axis_title,
    output_file,
):

    x_dist_dict = {}

    f = open(input_file)
    raw_data = json.load(f)

    if type(raw_data) != dict:
        raise ValueError("Dataset must be in dict format!")

    for key in raw_data.keys():
        
        if is_prof:
            if "proficiency" not in raw_data[key].keys() or "mask_target" not in raw_data[key]["proficiency"].keys() or "predicted_word" not in raw_data[key]["proficiency"].keys(): 
                raise KeyError("Sample must have mask_target and predicted_word keys under proficiency key!")

            raw_caption = raw_data[key]["proficiency"]["mask_target"]
            raw_foil_list = [raw_data[key]["proficiency"]["predicted_word"]]
        else:
            if "class" not in raw_data[key].keys() or "classes_foil" not in raw_data[key].keys():
                raise KeyError("Sample must have class and classes_foil keys!")

            raw_caption = raw_data[key]["class"]
            raw_foil_list = raw_data[key]["classes_foil"]
        
        add_to_dict(x_dist_dict, raw_caption, True)

        for foil in raw_foil_list:
            add_to_dict(x_dist_dict, foil, False)

                
    x_dist_tuple_list = [("Class", "Caption Count", "Foil Count")]

    for c in x_dist_dict.keys():
        x_dist_tuple_list.append((c, x_dist_dict[c][0], x_dist_dict[c][1]))

    create_plot(chart_type, x_dist_tuple_list, chart_title, x_axis_title, y_axis_title, output_file)

if __name__ == "__main__":
    main()
