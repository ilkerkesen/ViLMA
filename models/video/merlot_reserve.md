# Running Merlot Reserve
Paper Link: https://arxiv.org/abs/2201.02639

## Environment Setup

The instructions as follows,

```bash
conda create --name mreserve python=3.8 && conda activate mreserve
conda install -y python=3.8 tqdm numpy pyyaml scipy ipython cython typing h5py pandas matplotlib

# Install jax
pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
# If doing this on TPUs instead of locally...
# pip install "jax[tpu]>=0.2.18" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# This is needed sometimes https://stackoverflow.com/questions/66060487/valueerror-numpy-ndarray-size-changed-may-indicate-binary-incompatibility-exp
pip uninstall numpy
pip install numpy==1.19.5

pip install -r requirements.txt
```

## Running Merlot Reserve on the data

Just run the following command,
```bash
python run_merlot_reserve.py \
    --input_file inputs/input_file.json
```

This command will produce output `Main_Task_Results.json` and `Prof_Results.json` in the current directory. Additionally, the script will create `Errors.json` if any sample is skipped due to an unexpected error.
