# LEAN: Graph-based pruning for CNNs by extracting longest chains

## Introduction
This folder contains supplementary code for the paper **LEAN: Graph-based pruning for CNNs by extracting longest chains**. The code contains ready-to-run scripts to perform pruning experiments (of all three methods introduced in the paper) on the MS-D network on the simulated Circle-Square (CS) dataset.

This package contains building blocks for performing LEAN pruning which are generally applicable to any CNN. We provide ready-made functions to prune the MS-D, FCN-U-Net4 and FCN-ResNet50 architectures. To run LEAN for different architectures the user needs to write a custom pruning function for that model. Some guidelines to write custom LEAN pruning functions are given below in *LEAN pruning for other models*.

Link to paper: <https://arxiv.org/abs/2011.06923>

## System requirements
The package requires a Nvidia GPU with CUDA installed. The installation instructions below assume an anaconda Python installation.

## Installation instructions
To install the package, please

1. Clone the repository
```
git clone https://github.com/schoonhovenrichard/LEAN_CNN_pruning.git
``` 
and move to the main directory
```
cd LEAN_CNN_pruning/
```
2. Create conda environment with the required packages using the supplied environment file:
```
conda env create -f environment.yml
```
3. Activate the environment:
```
conda activate LEAN_CNN_pruning
```
Make sure to check that the Anaconda pip installer is being used, not the system version by running ```which pip```.

4. Install the package:
```
pip install -e .
```

5. Install Rust if it is not already installed:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

6. Install the Maturin package for Python bindings, and build:
```
pip install maturin
maturin develop
```

## Running tests
The package contains correctness tests of a number of building block algorithms used in LEAN pruning. To optionally perform these tests run
```
py.test
```
The package also contains tests to test the pruning methods for ResNet and U-Net. However, as these are computationally expensive they are not automatically run by pytest, but rather can be run manually in the ```tests/test_pruning_methods.py``` script.

## Running experiments
The code contains ready-to-run scripts to perform pruning experiments (of all three methods introduced in the paper) on the MS-D network on the simulated Circle-Square (CS) dataset. We provide a script to generate training, validation, and test data. We also provide a script to train an MS-D network on this data. There is already a pretrained network in the `trained_models` folder that the user can use, and there is pre-supplied data. Next, we provide a script to prune the trained MS-D network with any of the three methods described in the manuscript. 

In the `pruned_models` folder, example pruned networks from an experimental run are given. To obtain the accuracy of a pruned model on the test set, run 
`get_global_accuracy(model, test_dl)`. The pruning ratio and accuracy on the test set are automatically written into the filename.

1. First, move to the `examples` directory.

### Generate new training data

2. As mentioned, training data is supplied. To generate new data in the `data` directory, run:
```
python generate_example_data.py
```
The script generates 500 training images, 100 validation images, and 50 test images by default. The user can alter this, and other parameters, in the script.


### Train an MS-D network
3. As mentioned, a pretrained network is supplied. To train a new MS-D model from scratch instead, run:
```
python train_example_msd.py
```
The trained MS-D network is saved to the `trained_models` folder. By default the depth of the network is set to 50. The user can alter this, and other parameters, in the script.


### Run pruning experiments
4. There are a number of configurable parameters in the script. They are set to use the pretrained MS-D model by default. If the user trains a new MS-D model, they need to change the user configurable parameters in the `prune_example_msd.py` script.

5. To run pruning experiments on the trained MS-D network, run:
```
python prune_example_msd.py
```
The pruned networks are saved to the `pruned_models` folder after every pruning step. This way, the user can track the accuracy of the networks as pruning progresses. There are three `bool` variables to set which experiment to run; `lean_pruning, indivL1_pruning, indivSV_pruning`. Set those experiments you wish to run to `True`.


## Using LEAN pruning for other models
Here, we have supplied code for pruning MS-D, FCN-UNet4, and FCN-ResNet50 models. However, the algorithms to extract paths from the graph, and to compute the operator norm, are generally applicable to any CNN. If the user wants to write a custom made LEAN pruning function, we recommend that the user starts with a copy of e.g. the function `LEAN_MSD` in `pruning_algorithms.py`. To create a custom LEAN function, the user needs to write 4 subfunctions:

1. A function that returns a list of all the layers of operators that are subject to pruning. Example: `get_convs_MSD` from `pruning_utils.py`.
2. A function to prune biases for convolutional layers. Example: `prune_biases_MSD` in `pruning_algorithms.py`. The function `prune_biases_MSD` shows how we can iterate over the pruning masks in the model, and check if the convolutional channel is fully pruned. If this is the case, we prune the bias of that channel.
3. (Optional) If the user wants to include a redundancy pruning step, a function to prune redundant convolutions. Example: `Prune_Redundant_Convolutions_MSD` in `pruning_algorithms.py`. The function `Prune_Redundant_Convolutions_MSD` shows how we can check if all the preceding convolutional channels are pruned, and prune the current channel if this is the case.
4. A function, or code block, to define the CNN as a pruning graph. One can immediately define the adjacency list, or create an adjacency matrix and use the pre-supplied ```convert_matr_to_adjlist``` function to convert it.  To create an adjacency list, let every operator channel be a node index, and add to the list ```[inidx, outidx, normvalue]``` the edges between the channels. Similarly, the adjacency matrix has at the appropriate positions (columns and rows are the in and out indices) the norms of the convolutions. The code block at line 337 in `LEAN_MSD` shows how we can calculate the norms of the convolutional layers with `compute_FourierSVD_norms`, and insert them into our numpy array.

Given that the above functions have been written, the user can replace the instances of `get_convs_MSD`, `prune_biases_MSD`, and `Prune_Redundant_Convolutions_MSD` with their custom versions. Then, the user can take the code block like the one at line 335 in `pruning_algorithms.py`, and replace it with their custom code to create the norm-graph matrix. The rest of the code can stay the same, and the user has their custom LEAN pruning method!

**Remark**: If there are batch normalization layers or skip connections in the custom CNN model, these should be incorporated in the graph. This requires additional functions like `get_batchnorms_ResNet50` to be written.

## Authors and contributors
* **Richard Schoonhoven** - *Corresponding author* (https://github.com/schoonhovenrichard)
* **Allard Hendriksen** - (https://github.com/ahendriksen)
* **Daniel M Pelt** - (https://github.com/dmpelt)
* **Kees Joost Batenburg** - (https://www.cwi.nl/people/joost-batenburg)

## Articles
<a id="1">[1]</a> 
Schoonhoven, Richard and Hendriksen, Allard A. and Pelt, Dani\u00ebl M. and Batenburg, K. Joost (2020).
LEAN: graph-based pruning for convolutional neural networks by extracting longest chains.
ArXiv 2020, eprint=2011.06923.

## License
This code is published under BSD license.
