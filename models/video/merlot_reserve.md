# Running Merlot Reserve
Paper Link: https://arxiv.org/abs/2201.02639

## Environment Setup

The instructions as follows,

```bash

git clone https://github.com/mustafaadogan/merlot_reserve
cd merlot_reserve

conda env create -f environment.yml
conda activate mreserve

```

## Running Merlot Reserve on the data

Just run the following command,
```bash
python run_merlot_reserve.py \
    --input_file inputs/input_file.json \
    --video_dir video_dir/videos/ \
    --output_dir output_dir/output/

```

This command will produce output `Main_Task_Results.json` and `Prof_Results.json` in the current directory. Additionally, the script will create `Errors.json` if any sample is skipped due to an unexpected error.
