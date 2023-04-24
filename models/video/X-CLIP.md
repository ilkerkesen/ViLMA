# Running X-CLIP
Paper Link: https://arxiv.org/abs/2208.02816

## Environment Setup
X-CLIP model uses the same conda environment with VideoCLIP model. So, to setup the environment, first follow the instructions for the [VideoCLIP](VideoCLIP.md) model. Then, install `transformers` and `PyAV` packages,

```bash
conda activate videoclip_env
pip install transformers==4.25.1
pip install av==10.0.0
pip install -e .
cd /path/to/this/repo
cd models/video
```

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