# Offline Model-Based Optimization by Learning to Rank

Official implementation of ICLR'25 paper "Offline Model-Based Optimization by Learning to Rank". 

## Environment installation

To install dependencies and configure environments, please run commands in the terminal as follows:

```bash
YOUR_PATH_TO_CONDA=~/anaconda3 # Properly set it

# Create conda environment
conda create -n offline-ram python=3.8 -y
conda activate offline-ram

# Download MuJoCo package
wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210_linux.tar.gz
mkdir ~/.mujoco
tar -zxvf mujoco210_linux.tar.gz -C ~/.mujoco

# Mujoco_py installation
conda install gxx_linux-64 gcc_linux-64 -y
conda install --channel=conda-forge libxcrypt -y
pip install Cython==0.29.36 numpy==1.22.0 mujoco_py==2.1.2.14
# Set up the environment variable
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
# Reactivate the conda environment to make the variable take effect
conda activate offline-ram
# Copy C++ dependency libraries
mkdir ${YOUR_PATH_TO_CONDA}/envs/offline-ram/include/X11
mkdir ${YOUR_PATH_TO_CONDA}/envs/offline-ram/include/GL
sudo cp /usr/include/X11/*.h ${YOUR_PATH_TO_CONDA}/envs/offline-ram/include/X11
sudo cp /usr/include/GL/*.h ${YOUR_PATH_TO_CONDA}/envs/offline-ram/include/GL
# Mujoco Compile
python -c "import mujoco_py"

# Torch Installation
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Design-Bench Installation
pip install design-bench==2.0.12
pip install pip==24.0
pip install robel==0.1.2 morphing_agents==1.5.1 transforms3d --no-dependencies
pip install botorch==0.6.4 gpytorch==1.6.0

# Install other dependencies
pip install gym==0.13.1 params_proto==2.9.6 scikit-image==0.17.2 scikit-video==1.1.11 scikit-learn==0.23.1 wandb

# Fix numpy version, otherwise it would raise environment error
pip install numpy==1.22.0
```

then, download data of Design-Bench following [this link](https://github.com/brandontrabucco/design-bench/issues/11#issuecomment-2067352331).

You may probably fix this API change in line 394 in ``{YOUR_PATH_TO_CONDA}/envs/offline-ram/lib/python3.8/site-packages/design_bench/oracles/approximate_oracle.py`` from
```python
with zip_archive.open('rank_correlation.npy', "r") as file:
    rank_correlation = np.loads(file.read())
```
to 
```python
with zip_archive.open('rank_correlation.npy', "r") as file:
    rank_correlation = np.load(file, allow_pickle=True).item()
```

## Main Experiments

We have released our model weights trained with ListNet on 🤗Huggingface🤗: [https://huggingface.co/trxcc2002/Offline-RaM-ListNet/tree/main](https://huggingface.co/trxcc2002/Offline-RaM-ListNet/tree/main), which can be attached via code interface:
```python
from huggingface_hub import hf_hub_download
task = 'AntMorphology-Exact-v0'
seed = 1
weights_path = hf_hub_download(
    repo_id="trxcc2002/Offline-RaM-ListNet",
    filename=f"Offline-RaM-ListNet-{task}-seed{seed}/model.safetensors",
)
```
For a quick run, please first set your variables in ``run.sh`` as 
```bash
MAX_JOBS=8   # how many jobs do you want to run in parallel
AVAILABLE_GPUS="0 1 2 3"  # ids of your available GPUs
MAX_RETRIES=0  # number of retries when your program fails
```
then run ``bash run.sh`` in your terminal directly, where the pretrained model weights will be downloaded and used to search inside for final design candidate.  


To train from scratch, you can run our proposed method via
```bash
bash run_from_scratch.sh
```
or
```bash
python main_from_scratch.py --task <task> --loss <loss> --seed <seed>
```
where the options for argument ``--task`` and ``--loss`` are:
```python
tasks = [
    "AntMorphology-Exact-v0",
    "DKittyMorphology-Exact-v0",
    "Superconductor-RandomForest-v0",
    "TFBind8-Exact-v0",
    "TFBind10-Exact-v0",
]

losses = [
    "sigmoid_ce",
    "bce",
    "mse",
    "ranknet",
    "lambdarank",
    "rankcosine",
    "softmax",
    "listnet",
    "listmle",
    "approxndcg"
]
```

## Code Reference

+ Our implementation of loss functions is partially inherited from ``allrank``: [https://github.com/allegro/allRank](https://github.com/allegro/allRank).
+ We sincerely appreciate ``Design-Bench``: [https://github.com/brandontrabucco/design-bench](https://github.com/brandontrabucco/design-bench).

## Citation
```
@inproceedings{offline-ltr,
  title={Offline Model-Based Optimization by Learning to Rank},
  author={Tan, Rong-Xi and Xue, Ke and Lyu, Shen-Huan and Shang, Haopu and Wang, Yao and Wang, Yaoyuan and Fu, Sheng and Qian, Chao},
  booktitle={Proceedings of the 13th International Conference on Learning Representations (ICLR)},
  address={Singapore},
  year={2025},
}
```
