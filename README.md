## Solving Linear Inverse Problems using GANs

Code for the paper: [Solving Linear Inverse Problems Using GAN Priors: An Algorithm with Provable Guarantees](https://arxiv.org/abs/1802.08406).

### Requirements 
---
To run this code, you require Python 2.7, Tensorflow 1.0.1 (preferably with GPU support), Scipy and PyPNG.

Pip installation can be done by ```$ pip install -r requirements.txt```

### Instructions
---

1. Clone the repository, and run all the commands from the parent directory, ```pgdgan/```.

2. Download the datasets with the script*:
    ```shell
    $ ./setup/download_data.sh 
    ```
3. To train the DCGAN on celebA from scratch, please visit https://github.com/carpedm20/DCGAN-tensorflow, and follow the instructions.
 Else, pretrained GAN model is available, courtesy [Bora et al.](https://github.com/AshishBora/csgm) 
 To download it, please run the following script*: 
   ```shell
   $ ./setup/download_models.sh
   ```
   Make sure the model is located at ```./models/celebA_64_64/```
4. Run following to run the experiment:
    ```shell
    $ python ./src/pgdgan.py
    ```
    You can also use the script available in ```./exp_scripts/```


\* replicated from https://github.com/AshishBora/csgm .
