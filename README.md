# dwi_ml

Welcome to the **scil-vital** dwi_ml git repo.

#### Download

You can clone the git repo on your computer:

```bash
git clone https://github.com/scil-vital/dwi_ml.git
cd dwi_ml
```

#### Common dependencies

1. We strongly recommand working in an environment (ex, VirtualEnv).

2. You will need the **VITALabAi** git repo installed. 

    ```bash
    git clone https://bitbucket.org/vitalab/vitalabai_public/
    ```
    Then follow instructions in their README file.
    
3. You will also need the **scilpy** git repo: 

    ```bash
    git clone https://github.com/scilus/scilpy
    ```
    Then follow instructions in their README file.

3. If you follow the installation instructions for these two repos, most requirements should be fullfilled. To install our **other requirements**:

    ```bash
    pip install -r requirements.txt
    ```

5.  If you want to use the repo on heavier datasets and use your **GPU**, you should also install torch. If your computer is strong enough, use cuda to run your tasks on GPU through torch. 

    Install **cuda**:
     
    1. Verify that your computer has the adequate capacities in the "Pre-installation Actions" here: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html (sections 2.1 - 2.4). Hints: To find your graphic card, check the "About" in your computer settings. It will tell you your linux version and your graphic card.

    2. Then follow instructions here: https://developer.nvidia.com/cuda-downloads. Choose your system in the selector. You can choose deb(local) for the installer type. Then follow instructions.
 
    Install **torch**: 
    
    1. Use the selector under the "start locally" section here: https://pytorch.org/get-started/locally/ to know how to install torch with cuda. It will probably be something like this:

        ** Note that torch needs to be installed with python3. You can work directly in your python3 environment and use pip or then use pip3.
    
        ```bash
        pip install torch
        ```

    2. Perform the suggested verifications in the verifications section, for example using ipython.


### Installation

The installation of our library is straightforward.

```bash
python setup.py install   # For users 
python setup.py develop   # For developpers
```

