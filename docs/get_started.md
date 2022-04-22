### Installation 

a. Create a conda virtual enviroment and activate it. (Python 3.7.x is recommended.)

```shell 
conda create -n brg-ad python=3.7 -y
conda activate brg-ad
```

b. Install PyTorch following the [official instructions](https://pytorch.org/). The PyTorch version should be compatible with your CUDA environment. Here we use PyTorch 1.10 and CUDA 10.2.

```shell
pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

c. Install required libraries from 'requirements.txt'.

```shell
pip install -r requirements.txt
```