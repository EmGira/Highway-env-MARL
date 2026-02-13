## INSTALLATION

```bash
git clone https://github.com/EmGira/Highway-env-MARL
cd Highway-env-MARL
```

### 1. Prerequisites
* Python 3.9+ (tested on Python 3.12)
* CUDA enviroment (optional)

### 2. Env setup

```bash

python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On Linux:
source venv/bin/activate

```
### 3. Dependencies

NOTE:  
if you have a NVIDIA GPU,  
Install a version of PyTorch compatible with your CUDA version before proceeding. Check pytorch.org for the specific command for your system.

once you have done that, or if you dont have a GPU, proceed with:
```bash
pip install -r requirements.txt
```

