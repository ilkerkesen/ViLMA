# Running UniVL
Paper Link: https://arxiv.org/abs/2002.06353v3

## Environment Setup

First, please setup a conda environment. You can follow the same
instructions provided in the [original UniVL repository](https://github.com/microsoft/UniVL).
Then clone the forked UniVL that we have set up to integrate with 
the benchmark  

```bash
git clone https://github.com/andreapdr/UniVL.git
cd UniVL
conda create -n py_univl python=3.6.9
conda install --file requirements.txt
mkdir -p ./weight          
wget -P ./weight https://github.com/microsoft/UniVL/releases/download/v0/univl.pretrained.bin   # download pre-trained UniVL model
```

# Running UniVL on the data

First, extract the features from the videos
```bash
git clone https://github.com/andreapdr/VideoFeatureExtractor.git
cd VideoFeatureExtractor
python extract.py --csv=./multi3bench_data.csv --type=s3dg --batch_size=1 --num_decoding_thread=4
```

where multi3bench_data.csv is a csv formatted as
```csv
video_path,feature_path
path/to/video1.mp4,path/to/output/feature1.npy
path/to/video2.webm,path/to/output/feature2.npy
```


Finally, run the following command. If you are evaluating on the 'change-of-state' instrument, please provide also a specific setting (`action_foil, preState_foil, postState_foil, reverse_foil`).

```bash
python main_multi3bench.py --input-file /path/to/annotations.json --video-feature /path/to/video-features --change-state-setting reverse_foil
```