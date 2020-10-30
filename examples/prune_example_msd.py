import msd_pytorch as mp
from torch.utils.data import DataLoader
import numpy as np
import pathlib

import pruning_algorithms as lean
#from pruning_algorithms import LEAN_SV_MSD_3x3
from train_example_msd import get_global_accuracy

if __name__ == '__main__':
    c_in = 1
    depth = 50
    width = 1
    dilations = [1,2,3,4,5,6,7,8,9,10]

    path = str(pathlib.Path().absolute()) + '/data/'
    train_input_glob = path+"train/noisy/*.tiff"
    train_target_glob = path+"train/label/*.tiff"
    val_input_glob = path+"val/noisy/*.tiff"
    val_target_glob = path+"val/label/*.tiff"
    test_input_glob = path+"test/noisy/*.tiff"
    test_target_glob = path+"test/label/*.tiff"

    labels = [0, 1, 2, 3, 4]
    batch_size = 5

    print("Load training dataset")
    train_ds = mp.ImageDataset(train_input_glob, train_target_glob, labels=labels)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    print("Load validation set")
    val_ds = mp.ImageDataset(val_input_glob, val_target_glob, labels=labels)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False)

    print("Load test set")
    test_ds = mp.ImageDataset(test_input_glob, test_target_glob, labels=labels)
    test_dl = DataLoader(test_ds, 1, shuffle=False)

    print("Create segmentation network model")
    model = mp.MSDSegmentationModel(
            c_in, train_ds.num_labels, depth, width, dilations=dilations)

    print("Start estimating normalization parameters")
    model.set_normalization(train_dl)
    print("Done estimating normalization parameters")

    # Load pre-trained network
    # User can train their own but we have pre-supplied one
    model.load("trained_models/msd_network_d=50_epoch_47.torch")

    # Accuracy of base model:
    acc = get_global_accuracy(model, test_dl)

    # Choose which pruning procedure to run
    lean_pruning = True
    indivL1_pruning = False
    indivSV_pruning = False

    perc = 0.05
    nsteps = 15
    retraining_epochs = 3
    percentage = np.exp(np.log(perc)/float(nsteps))
    print("Pruning to ratio {0} in {1} steps.".format(perc, nsteps))
    print("Removes {0:.2f}% of convolutions each step.".format(100*(1.0-percentage)))
    if lean_pruning:
        tot_perc = 1.0
        print("Starting LEAN pruning")
        for step in range(nsteps):
            tot_perc *= percentage
            # We do not include the final layer for pruning as outlined in paper.
            model = lean.LEAN_SV_MSD_3x3(model, tot_perc, verbose=False)
            for epoch in range(retraining_epochs):
                model.train(train_dl, 1)
                train_error = model.validate(train_dl)
                print(f"{epoch:05} Training error: {train_error: 0.6f}")
            acc = get_global_accuracy(model, test_dl)
            model.save("pruned_models/msd_network2_LEAN_d={0}_ratio={1:.4f}_acc={2:.4f}.torch".format(depth, tot_perc, acc), epoch)
    if indivL1_pruning:
        tot_perc = 1.0
        print("Starting individual filter pruning (L1) pruning")
        for step in range(nsteps):
            tot_perc *= percentage
            # We do not include the final layer for pruning as outlined in paper.
            model = lean.IndivL1_Global_MSD_3x3(model, tot_perc, verbose=False)
            for epoch in range(retraining_epochs):
                model.train(train_dl, 1)
                train_error = model.validate(train_dl)
                print(f"{epoch:05} Training error: {train_error: 0.6f}")
            acc = get_global_accuracy(model, test_dl)
            model.save("pruned_models/msd_network2_indivL1_d={0}_ratio={1:.4f}_acc={2:.4f}.torch".format(depth, tot_perc, acc), epoch)
    if indivSV_pruning:
        tot_perc = 1.0
        print("Starting individual filter pruning (SV) pruning")
        for step in range(nsteps):
            tot_perc *= percentage
            # We do not include the final layer for pruning as outlined in paper.
            model = lean.IndivSV_Global_MSD_3x3(model, tot_perc, verbose=False)
            for epoch in range(retraining_epochs):
                model.train(train_dl, 1)
                train_error = model.validate(train_dl)
                print(f"{epoch:05} Training error: {train_error: 0.6f}")
            acc = get_global_accuracy(model, test_dl)
            model.save("pruned_models/msd_network2_indivSV_d={0}_ratio={1:.4f}_acc={2:.4f}.torch".format(depth, tot_perc, acc), epoch)
