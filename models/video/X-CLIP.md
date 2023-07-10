# Running X-CLIP
Paper Link: https://arxiv.org/abs/2208.02816

## Environment Setup
X-CLIP model uses the same conda environment with this repo. So, to setup the environment, follow [these steps](../../README.md). Note that you need `av==10.0.0` package to run this model.

## Running X-CLIP on the data

Just run the following command,
```bash
python run_xclip.py \
    --input-file /path/to/annotations.json \
    --quva-dir /path/to/quva \
    --something-something-dir /path/to/dataset \
    --output-file /path/to/output.json
```

This command will produce a results annotation file `/path/to/output.json`. To generate scores for the proficiency task, pass the `--proficiency` flag.