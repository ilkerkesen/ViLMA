# Running ClipBERT
Paper Link: https://arxiv.org/abs/2102.06183

## Environment Setup

The original implementation works with a Docker image. Since, we are working on HPC clusters, we use Singularity instead Docker. We first start by pulling the container image and cloning the fork,

```bash
singularity pull docker://jayleicn/clipbert:latest  # it will take a while!
git clone https://github.com/ilkerkesen/ClipBERT.git
cd ClipBERT
git checkout -b vl-bench origin/vl-bench
```

We then follow the instructions for the original fork to download necessary checkpoints,

```bash
PATH_TO_STORAGE=/path/to/your/data/
mkdir -p $PATH_TO_STORAGE/txt_db  # annotations
mkdir -p $PATH_TO_STORAGE/vis_db  # image and video 
mkdir -p $PATH_TO_STORAGE/finetune  # finetuning results
mkdir -p $PATH_TO_STORAGE/pretrained  # pretrained models

bash scripts/download_pretrained.sh $PATH_TO_STORAGE
```

## Container Startup
After we acquired the necessary files for the ClipBERT model, now it is time to start the container. Before that please download the necessary data files for the benchmark (QUVA dataset, SomethingSomething dataset, annotations etc.). Then run the modified `launch_container.sh` script to start-then-enter into the container,

```bash
$ export QUVA_DIR=/path/to/quva-dataset-dir
$ export SOMETHING_SOMETHING_DIR=/path/to/something-something-v2-videos
$ sh launch_container.sh \
    /path/to/clipbert_latest.sif \
    $PATH_TO_STORAGE \
    /path/to/json-annotations \
    /path/to/output-dir
Singularity> cd /clipbert
Singularity> source setup.sh
```

## Running VideoCLIP on the data

Run the following command inside the container,
```bash
Singularity> python src/tasks/run_bench.py \
    --json_path /path/to/annotations.json \
    --quva_dir /quva \
    --something_something_dir /something \
    --num_frames 16
    --config $CONFIG_PATH
```

This command will produce a single output `output={accuracy}%`.