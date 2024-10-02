# Offline Model-Based Optimization by Learning to Rank

Official implementations of ICLR'25 submission paper "Offline Model-Based Optimization by Learning to Rank". 

## Environment installation

To install dependencies and configure environments, please run commands in the terminal as follows:

```bash
# Create conda environment
conda create -n offline-ram python=3.8 -y
conda activate offline-ram

# Download MuJoCo package
wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210_linux.tar.gz
mkdir ~/.mujoco
tar -zxvf mujoco210_linux.tar.gz -C ~/.mujoco

# Mujoco_py installation
pip install Cython==0.29.36 numpy==1.22.0 mujoco_py==2.1.2.14
# Set up the environment variable
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
# Mujoco Compile
python -c "import mujoco_py"

# Torch Installation
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Design-Bench Installation
pip install design-bench==2.0.12
pip install robel==0.1.2 morphing_agents==1.5.1 transforms3d --no-dependencies
pip install botorch==0.6.4 gpytorch==1.6.0

# Install other dependencies
pip install gym==0.13.1 params_proto==2.9.6 scikit-image==0.17.2 scikit-video==1.1.11 scikit-learn==0.23.1 wandb
```

## Main Experiments

For a quick run, please first set your variables in ``run.sh`` as 
```bash
MAX_JOBS=8   # how many jobs do you want to run in parallel
AVAILABLE_GPUS="0 1 2 3"  # ids of your available GPUs
MAX_RETRIES=0  # number of retries when your program fails
```
then run ``bash run.sh`` in your terminal directly. 


For details, you can run our proposed method as
```bash
python main.py --task <task> --loss <loss> --seed <seed>
```
where the options for argument ``--task`` and ``loss`` are:
```python
tasks = [
    "AntMorphology-Exact-v0",
    "DKittyMorphology-Exact-v0",
    "Superconductor-RandomForest-v0",
    "TFBind8-Exact-v0",
    "TFBind10-Exact-v0",
]

tasks = [
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

## Availability of Model Weights

We promise to make our code as well as model weights publicly available once our paper is accepted. 

## Code Reference

Our implementation of loss functions is partially inherited from ``allrank``: [https://github.com/allegro/allRank](https://github.com/allegro/allRank). 
