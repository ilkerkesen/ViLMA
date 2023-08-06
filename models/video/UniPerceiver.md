# Running UniPerceiver
Paper Link: [https://arxiv.org/abs/2104.08860](https://arxiv.org/abs/2112.01522)

## Environment Setup

The instructions are as follows,

```bash
git clone https://github.com/mustafaadogan/Uni-Perceiver
cd Uni-Perceiver/vlbench_dataset
mkdir annotations_new
mkdir videos
cd ..
conda create --name uniperceiver && conda activate uniperceiver
pip install -r requirements.txt
gdown 1aN1i9U56uON4ISwIamjfLTDCQBVzmFU3
```

You need to install Apex for this model. Check this [link](https://github.com/NVIDIA/apex). The installation command changes according to the pip version. Apex also supports Python-only build with

```bash
pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./
```

##Data Preparation
Prepare your data as shown below.   
 ``` 
  DATA_PATH/
      └── vlbench_dataset/
          ├── vocabulary_CLIP_with_endoftext.pkl
          ├── annotations_new
          │    ├── Annotations_File_Name.json
          └── videos
               ├── video0.mp4
               └── ...
  ```
I already created vlbench_dataset folder with vocabulary_CLIP_with_endoftext.pkl in the repo. You can move your video files under the videos folder or set this structure in a different directory.

##Config File Preparation
Open $(UNIPERCEIVER_PATH)/configs/BERT_L12_H768_experiments/zeroshot_config/vlbench.yaml
Set DATALOADER.FEATS_FOLDER, DATALOADER.ANNO_FOLDER, INFERENCE.TEST_ANNFILE values according to your data structure created above. 

## Running UniPerceiver on the data

I do not use a cluster. That's why I do the following settings,
```bash
export DATA_PATH=/path/to/data
export LOCAL_RANK=0
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
python main.py --num-gpus 1 --init_method slurm --config-file configs/BERT_L12_H768_experiments/zeroshot_config/vlbench.yaml \
    --eval-only MODEL.WEIGHTS uni-perceiver-base-L12-H768-224size-pretrained.pth OUTPUT_DIR /path/to/output
```

This command will produce output `Main_Task_Results.json` and `Prof_Results.json` files in desired output path.
