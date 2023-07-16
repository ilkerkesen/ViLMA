# Running CLIP4Clip
Paper Link: https://arxiv.org/abs/2104.08860

## Environment Setup

The instructions as follows,

```bash
conda create --name clip4clip && conda activate clip4clip
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install opencv-python boto3 requests pandas

git clone https://github.com/mustafaadogan/CLIP4Clip
cd CLIP4Clip
wget -P ./modules https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```

## Running CLIP4Clip on the data

Just run the following commands,
```bash
export LOCAL_RANK=0
python -m torch.distributed.launch --nproc_per_node=1 main_task_retrieval.py \
    --do_eval --num_thread_reader=4 --val_csv inputs/input_file.json\
    --features_path inputs/videos/ --output_dir outputs/CLIP4Clip\
    --max_frames 12 --batch_size_val 1 --datatype vlbench --feature_framerate 1 \
    --slice_framepos 2 --loose_type --linear_patch 2d --sim_header meanP
    --pretrained_clip_name ViT-B/32
```

This command will produce output `Main_Task_Results.json` and `Prof_Results.json` files in desired output path.
