### CVPR 2021 Submission #8167. Confidential review copy. Do not distribute.

import torch
import msd_pytorch as mp
from torch.utils.data import DataLoader
import numpy as np
import pathlib

def get_global_accuracy(model, test_dl):
    # Test on testset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    correct = 0
    total = 0
    model.net.eval()
    model.msd.eval()
    for i, (input_im, labels) in enumerate(test_dl):
        input_im, labels = input_im.to(device), labels.to(device)
        outputs = model.net(input_im)
        predicted = torch.argmax(outputs.data, 1)
        total += labels.numel()
        correct += (predicted == labels).sum().item()
    print('     Accuracy of the network on the 10000 test images: {0:.2f}%'.format(100 * correct / float(total)))
    return float(correct/float(total))

if __name__ == '__main__':
    ### USER CONFIGURABLE PARAMETERS

    # The depth of the MSD network. Good values range between 30 and 200.
    depth = 50
    # The dilation scheme to use for the MSD network. The default is [1,
    # 2, ..., 10], but [1, 2, 4, 8] is good too.
    dilations = [1,2,3,4,5,6,7,8,9,10]
    # The number of epochs to train for
    epochs = 50
    # The mini-batch size used in training.
    batch_size = 5

    ### END OF USER CONFIGURABLE PARAMETERS

    # The number of input channels of the MSD network
    c_in = 1
    # The width of the MSD network. A value of 1 is recommended.
    width = 1
    # The labels to be assigned to the objects
    labels = [0, 1, 2, 3, 4]

    path = str(pathlib.Path().absolute()) + '/data/'
    train_input_glob = path+"train/noisy/*.tiff"
    train_target_glob = path+"train/label/*.tiff"
    val_input_glob = path+"val/noisy/*.tiff"
    val_target_glob = path+"val/label/*.tiff"


    print("Load training dataset")
    train_ds = mp.ImageDataset(train_input_glob, train_target_glob, labels=labels)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    # Load Validation dataset (if specified)
    print("Load validation set")
    val_ds = mp.ImageDataset(val_input_glob, val_target_glob, labels=labels)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False)

    print("Create segmentation network model")
    model = mp.MSDSegmentationModel(
            c_in, train_ds.num_labels, depth, width, dilations=dilations)

    print("Start estimating normalization parameters")
    model.set_normalization(train_dl)
    print("Done estimating normalization parameters")

    print("Starting training...")
    best_validation_error = np.inf
    validation_error = 0.0

    for epoch in range(epochs):
        # Train
        model.train(train_dl, 1)
        # Compute training error
        train_error = model.validate(train_dl)
        print(f"{epoch:05} Training error: {train_error: 0.6f}")
        # Compute validation error
        if val_dl is not None:
            validation_error = model.validate(val_dl)
            print(f"{epoch:05} Validation error: {validation_error: 0.6f}")
        # Save network if worthwile
        if validation_error < best_validation_error or val_dl is None:
            best_validation_error = validation_error
            model.save(f"trained_models/msd_network_d={0}_epoch_{1}.torch".format(depth, epoch), epoch)
