# About Plot Script
This script takes your dataset in json file, then creates a plot in a Excel file, like in Valse. Plot script is able to create 2D Bar Chart, 3D Bar Chart, and Line Chart. You need to pass the appropriate parameters.
Ensure your json file has the standard data structure shared in the [dummy.json](https://github.com/ilkerkesen/vl-bench/blob/main/data/dummy.json) file.

Parameters:<br>
*'-i', '--input-file'* : Dataset path -> "$(PATH_TO_DATASET)/data.json" <br>
*'-p', '--is-prof'* : *True* for Proficiency task, *False* for main task. <br>
*'-ctype', '--chart-type'* : Chart type -> *bar3d*, *bar2d*, *line* <br>
*'-ctitle', '--chart-title'* : Chart Title <br>
*'-xtitle', '--x-axis-title'* : X axis title <br>
*'-ytitle', '--y-axis-title'* : Y axis title <br>
*'-o', '--output-file'* : Output path -> "$(PATH_TO_OUTPUT)/output.xlsx"  <br>

## Running Plot Script on the data

Just run the following command,
```bash
python plotter.py \
    -i $(PATH_TO_DATASET)/SRL.json \
    -p False \
    -ctype bar3d \
    -ctitle Action Replacement Distributions \
    -xtitle Verbs \
    -ytitle Count \
    -o $(PATH_TO_OUTPUT)/output.xlsx
```

This command will produce a plot in Excel file `$(PATH_TO_OUTPUT)/output.xlsx`.  
