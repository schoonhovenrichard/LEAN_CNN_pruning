# LEAN: Graph-based pruning for CNNs by extracting longest chains

This folder contains supplementary code and data for performing **LEAN** pruning (and the methods it is compared against). The folder contains tests to check the validity of certain algorithms, and example code for the user to test LEAN themselves. 

Here, we have supplied functions to prune the MS-D and FCN-ResNet50 architectures. However, the algorithms to extract paths from the graph, and to compute the operator norm, are generally applicable to any CNN. To run LEAN for different architectures the user needs to write a custom pruning function for that model. Some guidelines to write custom LEAN pruning functions are given below in *LEAN pruning for other models*.

Paper:
**TODO:** LINK HERE

## Installation instructions
To install the package, please

1. Move to the main directory.
2. Create conda environment with the required packages using the supplied environment file:
```
conda env create -f environment.yml
```
3. Activate the environment:
```
conda activate LEAN_CNN_pruning
```
4. Install the package:
```
pip install -e .
```
**Note**: If the user encounters errors such as:
```
Traceback (most recent call last):
  File "prune_example_msd.py", line 6, in <module>
    import pruning_algorithms as lean
ModuleNotFoundError: No module named 'pruning_algorithms'
```
while running certain examples, this is possibly because the step `pip install -e .` was not performed, or not performed with the `-e` flag. The user could try re-installing the package with the `-e` flag.


## Running tests
In addition to the supplied examples, the repository contains tests to check the validity of certain algorithms. 

1. Move to the `tests` directory and run
```
py.test
```
to run the tests. This will, e.g., generate random DAGs to test the longest path algorithm against a NetworkX implementation, or generate random convolutions filters and test the operator norm computation method against the power method.


## Running experiments
For anyone interested in repeating experiments using LEAN, the following describes how to run the experiments. 

1. First, move to the example directory:
```
cd examples
```
**Note:** There is already a pretrained network supplied so the reader can decide to skip the *Train an MS-D network* step and use the pretrained network in the `trained_models` folder. In the `pruned_models` folder, example pruned networks from an experimental run are given.


### Generate new training data

2. To generate new data in the `data` directory, run:
```
python generate_example_data.py
```
The script currently generates 500 training images, 100 validation images, and 50 test images. The user can alter this, and other parameters, in the script.


### Train an MS-D network
3. As mentioned, a pretrained network is supplied. To train a new MS-D model, run:
```
python train_example_msd.py
```
The trained MS-D network is saved to the `trained_models` folder. Currently the depth of the network is set to 50 so as to have a high-performing network, yet not to expensive to train. These parameters can be changed in the script.


### Run pruning experiments
4. To run pruning experiments on the trained MS-D network, run:
```
python prune_example_msd.py
```
The pruned networks are saved to the `pruned_models` folder after every pruning step. This way, the user can track the accuracy of the networks as pruning progresses. There are three `bool` variables to set which experiment to run; `lean_pruning, indivL1_pruning, indivSV_pruning`. Set those experiments you wish to run to `True`.

## Using LEAN pruning for other models
Here, we have supplied code for pruning MS-D, and FCN-ResNet50 models. However, the algorithms to extract paths from the graph, and to compute the operator norm, are generally applicable to any CNN. If the user wants to write a custom made LEAN pruning function, we recommend that the user starts with a copy of e.g. the function `LEAN_SV_MSD` in `pruning_algorithms.py`. To create a custom LEAN function, the user needs to write 4 subfunctions:

1. A function that returns a list of all the convolutional layers that are subject to pruning. Example: `get_convs_MSD` from `pruning_utils.py`.
2. A function to prune biases for convolutional layers. Example: `prune_biases_MSD` in `pruning_algorithms.py`. The function `prune_biases_MSD` shows how we can iterate over the pruning masks in the model, and check if the convolutional channel is fully pruned. If this is the case, we prune the bias of that channel.
3. (Optional) If the user wants to include a redundancy pruning step, a function to prune redundant convolutions. Example: `Prune_Redundant_Convolutions_MSD` in `pruning_algorithms.py`. The function `Prune_Redundant_Convolutions_MSD` shows how we can check if all the preceding convolutional channels are pruned, and prune the current channel if this is the case.
4. A function, or code block, to create a numpy array which represents the graph described in the LEAN procedure. The array is an adjacency matrix with at the appropriate positions the norms of the convolutions. The code block at line 335 in `LEAN_SV_MSD` shows how we can calculate the norms of the convolutional layers with `compute_FourierSVD_norms`, and insert them into our numpy array.

Given that the above functions have been written, the user can replace the instances of `get_convs_MSD`, `prune_biases_MSD`, and `Prune_Redundant_Convolutions_MSD` with their custom versions. Then, the user can take the code block at line 335 and replace it with their custom code to create the norm-graph matrix. The rest of the code can stay the same, and the user has their custom LEAN pruning method!

**Remark**: If there are batch normalization layers or skip connections in the custom CNN model, these should be incorporated in the edge values of the graph. The code block starting at line 61 in `LEAN_SV_ResNet50` shows how this can be done. This requires an additional function like `get_batchnorms_ResNet50` to be written.


## Authors and contributors
* **Richard Schoonhoven** - *Corresponding author* (https://github.com/schoonhovenrichard)
* **Allard Hendriksen** - (https://github.com/ahendriksen)
* **Daniel M Pelt** - (https://github.com/dmpelt)
* **Kees Joost Batenburg** - (https://www.cwi.nl/people/joost-batenburg)

## Articles
<!---
<a id="1">[1]</a> 
Thierens, Dirk (2010).
The linkage tree genetic algorithm.
International Conference on Parallel Problem Solving from Nature. Springer, Berlin, Heidelberg, 2010.
-->

## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE.md](LICENSE.md) file for details.
