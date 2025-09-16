# Installing notes

## What I did to make it work on my machine

After upgrading my graphics card, I faced several problems installing the venv

Please, ensure nvcc is installed, It might not be sufficient to only get nvidia-smi working. 

NVIDIA-SMI 575.64.03              Driver Version: 575.64.03      CUDA Version: 12.9 


faiss-gpu only works with python3.10 or less (why using it them?)

So, create a venv with that version

```bash
conda create -n my_python_env python=3.10
```


Install stable pytorch

```bash
pip3 install torch torchvision
```

By the time I did it, it was pytorch 2.8.0, with cuda 12.8

then install xformers. Allso default version:

```bash
pip install xformers
```

Continue with all other modules, always the default:


```bash
pip install faiss-gpu pandas prettytable pytorch-metric-learning torchmetrics pytorch-lightning
```


### Faiss is compatible only with numpy < 2

```bash
pip install "numpy<2"
```

## Other modules I added

I like using `$GIT_ROOT`, so GitPython is needed:

```bash
pip install git-python
```