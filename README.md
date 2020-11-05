# LEAN: Graph-based pruning for CNNs by extracting longest chains

This repository contains Python scripts for performing **LEAN** pruning (and the methods it is compared against). The algorithms to run the chain extraction on the graph, and to compute the operator norm, are generally applicable. In addition, functions are supplied to prune the MS-D and FCN-ResNet50 architectures.

Example training data, trained MS-D nets, and pruned MS-D nets are supplied in the `examples` folder. However, scripts are provided to generate this data from scratch.

Paper:
TODO: LINK HERE

## Installation
To install the package, please run

1. Clone the git repository
```
git clone https://github.com/schoonhovenrichard/LEAN_CNN_pruning
```
2. Move to the directory: `cd LEAN_CNN_pruning`.
3. Create conda environment with the required packages using the supplied environment file:
```
conda env create -f environment.yml
```
4. Activate the environment: `conda activate LEAN_CNN_pruning`
5. Install the package: 
```
pip install -e .
```

## Running the experiments
For anyone interested in repeating experiments using LEAN, the following describes how to run the experiments. 

**Note:** There is already training data, and a pre-trained network supplied so the reader can decide to skip certain steps of this process and use the pre-supplied data!

### Generate new training data
To generate new data in the `data` directory, run:
```
python generate_example_data.py
```

The script currently generates 500 training images, 100 validation images, and 50 test images. The user can alter this, and other parameters, in the script.

### Train an MS-D network
To train a new MS-D model, run:
```
python train_example_msd.py
```

Currently the depth of the network is set to 50 so as to have a high-performing network, yet not to expensive to train. These parametes can be changed in the script

### Run pruning tests
To run pruning experiments on the trained MS-D network, run:
```
python prune_example_msd.py
```

There are three `bool` variables to set which experiment to run; `lean_pruning, indivL1_pruning, indivSV_pruning`. Set those experiments you wish to run to `True`.

## Running tests
In addition to the supplied examples, the repository contains tests to check the validity of certain algorithm. If `pytest` is installed, move to the `tests` directory and run
```
py.test
```
to run the test. This will e.g. generate random DAGs to test the longest path algorithm against a NetworkX implementation; or generate random convolutions filters and test our method against the power method.

## Authors and contributors
* **Richard Schoonhoven** - *Corresponding author* (https://github.com/schoonhovenrichard)
* **Allard Hendriksen** - (https://github.com/ahendriksen)
* **Daniel M Pelt** - (https://github.com/dmpelt)
* **Kees Joost Batenburg** - (https://www.cwi.nl/people/joost-batenburg)

## Articles
<a id="1">[1]</a> 
Thierens, Dirk (2010).
The linkage tree genetic algorithm.
International Conference on Parallel Problem Solving from Nature. Springer, Berlin, Heidelberg, 2010.

## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE.md](LICENSE.md) file for details.
