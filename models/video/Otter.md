# Running Otter
Paper Link: https://arxiv.org/abs/2305.03726

## Environment Setup
First, please setup a conda environment by using their packages.
```bash
git https://github.com/emrecanacikgoz/Otter.git
cd Otter
git checkout vilma origin/vilma
conda env create -f environment.yml
conda activate otter 
```
Or, if you want to use our environment (vl-bench) please install the packages below:
```bash
pip install eionops
pip install accelerate
pip install peft
pip install av 
```

## Running Otter on the data
Run the following command,
```bash
python run_otter.py \
    --input-file /kuacc/users/eacikgoz17/vl-benchmark/ViLMA/data/counting-easy-spelled-pts.json \
    --quva-dir /kuacc/users/eacikgoz17/vl-benchmark/eval-data/QUVARepetitionDataset \
    --something-something-dir /datasets/20bn_something_something/v1/ \
    --output-file output-v0.json
```


This command will produce a results annotation file `/path/to/output.json`.  To generate scores for the proficiency task, pass the `--proficiency` flag.