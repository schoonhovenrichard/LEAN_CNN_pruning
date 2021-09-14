import pruning_algorithms as lean
import numpy as np
import random
import pytest
import msd_pytorch as mp
from model_backbones.unet_seg_model import UNet4AvgSegmModel
from model_backbones.resnet_seg_model import ResNet50AvgSegmentationModel

def test_LEAN_indiv_MSD3x3(tries=10, eps=0.05):
    """Test whether LEAN pruning actually prunes 
        the correct amount of convolutions
        (3x3 only).
        
    We check that:
    - Pruning method prunes the correct amount.
    """
    for t in range(tries):
        # Create random MSD network
        c_in = random.randint(1,6)
        num_labels = random.randint(1,6)
        depth = random.randint(30,50)
        width = 1
        dilations = [1,2,3,4,5,6,7,8,9,10]
        model = mp.MSDSegmentationModel(c_in, num_labels, depth, width, dilations=dilations)

        # Define pruning percentage and number of steps
        perc = random.uniform(0.05, 0.5)
        nsteps = random.randint(1, 10)

        # Running fine-tuning pruning procedure
        tot_perc = 1.0
        percentage = np.exp(np.log(perc)/float(nsteps))
        for step in range(nsteps):
            tot_perc *= percentage
            model = lean.LEAN_MSD_3x3(model, tot_perc, Redun=False, verbose=False)
            prune_frac = lean.fraction_pruned_convs_MSD3x3(model)[2]

        # Check if number of pruned convolutions is correct
        assert abs((1-prune_frac-perc)/perc) < eps,"Test failed for LEAN pruning (3x3)!"

def test_LEAN_indiv_MSD(tries=10, eps=0.05):
    """Test whether LEAN pruning actually 
        prunes the correct amount of 
        convolutions.
        
    We check that:
    - Pruning method prunes the correct amount.
    """
    for t in range(tries):
        # Create random MSD network
        c_in = random.randint(1,6)
        num_labels = random.randint(1,6)
        depth = random.randint(30,50)
        width = 1
        dilations = [1,2,3,4,5,6,7,8,9,10]
        model = mp.MSDSegmentationModel(c_in, num_labels, depth, width, dilations=dilations)

        # MS-D Pytorch initializes at zero so pruning skips them very first time
        # becayse we don't train in this test function...
        model.msd.final_layer.linear.weight.data.fill_(0.1)

        # Define pruning percentage and number of steps
        perc = random.uniform(0.05, 0.5)
        nsteps = random.randint(1, 10)

        # Running fine-tuning pruning procedure
        tot_perc = 1.0
        percentage = np.exp(np.log(perc)/float(nsteps))
        for step in range(nsteps):
            tot_perc *= percentage
            model = lean.LEAN_MSD(model, tot_perc, Redun=False, verbose=False)
            prune_frac = lean.fraction_pruned_convs_MSD(model)[2]

        # Check if number of pruned convolutions is correct
        assert abs((1-prune_frac-perc)/perc) < eps,"Test failed for LEAN pruning!"

def test_SV_indiv_MSD3x3(tries=10, eps=0.05):
    """Test whether SV individual filter pruning
        actually prunes the correct amount of 
        convolutions (3x3 only).
        
    We check that:
    - Pruning method prunes the correct amount.
    """
    for t in range(tries):
        # Create random MSD network
        c_in = random.randint(1,6)
        num_labels = random.randint(1,6)
        depth = random.randint(30,50)
        width = 1
        dilations = [1,2,3,4,5,6,7,8,9,10]
        model = mp.MSDSegmentationModel(c_in, num_labels, depth, width, dilations=dilations)

        # Define pruning percentage and number of steps
        perc = random.uniform(0.05, 0.5)
        nsteps = random.randint(1, 10)

        # Running fine-tuning pruning procedure
        tot_perc = 1.0
        percentage = np.exp(np.log(perc)/float(nsteps))
        for step in range(nsteps):
            tot_perc *= percentage
            model = lean.IndivSV_Global_MSD_3x3(model, tot_perc, Redun=False, verbose=False)
            prune_frac = lean.fraction_pruned_convs_MSD3x3(model)[2]

        # Check if number of pruned convolutions is correct
        assert abs((1-prune_frac-perc)/perc) < eps,"Test failed for SV finetuning (3x3)!"

def test_SV_indiv_MSD(tries=10, eps=0.05):
    """Test whether SV individual filter pruning
        actually prunes the correct amount of 
        convolutions.
        
    We check that:
    - Pruning method prunes the correct amount.
    """
    for t in range(tries):
        # Create random MSD network
        c_in = random.randint(1,6)
        num_labels = random.randint(1,6)
        depth = random.randint(30,50)
        width = 1
        dilations = [1,2,3,4,5,6,7,8,9,10]
        model = mp.MSDSegmentationModel(c_in, num_labels, depth, width, dilations=dilations)

        # MS-D Pytorch initializes at zero so pruning skips them very first time
        # becayse we don't train in this test function...
        model.msd.final_layer.linear.weight.data.fill_(0.1)

        # Define pruning percentage and number of steps
        perc = random.uniform(0.05, 0.5)
        nsteps = random.randint(1, 10)

        # Running fine-tuning pruning procedure
        tot_perc = 1.0
        percentage = np.exp(np.log(perc)/float(nsteps))
        for step in range(nsteps):
            tot_perc *= percentage
            model = lean.IndivSV_Global_MSD(model, tot_perc, Redun=False, verbose=False)
            prune_frac = lean.fraction_pruned_convs_MSD(model)[2]

        # Check if number of pruned convolutions is correct
        assert abs((1-prune_frac-perc)/perc) < eps,"Test failed for SV finetuning!"

def test_l1_indiv_MSD3x3(tries=10, eps=0.05):
    """Test whether L1 individual filter pruning
        actually prunes the correct amount of 
        convolutions (3x3 only).
        
    We check that:
    - Pruning method prunes the correct amount.
    """
    for t in range(tries):
        # Create random MSD network
        c_in = random.randint(1,6)
        num_labels = random.randint(1,6)
        depth = random.randint(30,50)
        width = 1
        dilations = [1,2,3,4,5,6,7,8,9,10]
        model = mp.MSDSegmentationModel(c_in, num_labels, depth, width, dilations=dilations)

        # Define pruning percentage and number of steps
        perc = random.uniform(0.05, 0.5)
        nsteps = random.randint(1, 10)

        # Running fine-tuning pruning procedure
        tot_perc = 1.0
        percentage = np.exp(np.log(perc)/float(nsteps))
        for step in range(nsteps):
            tot_perc *= percentage
            model = lean.IndivL1_Global_MSD_3x3(model, tot_perc, Redun=False, verbose=False)
            prune_frac = lean.fraction_pruned_convs_MSD3x3(model)[2]

        # Check if number of pruned convolutions is correct
        assert abs((1-prune_frac-perc)/perc) < eps,"Test failed for L1 finetuning (3x3)!"

def test_l1_indiv_MSD(tries=10, eps=0.05):
    """Test whether L1 individual filter pruning
        actually prunes the correct amount of 
        convolutions.
        
    We check that:
    - Pruning method prunes the correct amount.
    """
    for t in range(tries):
        # Create random MSD network
        c_in = random.randint(1,6)
        num_labels = random.randint(1,6)
        depth = random.randint(30,50)
        width = 1
        dilations = [1,2,3,4,5,6,7,8,9,10]
        model = mp.MSDSegmentationModel(c_in, num_labels, depth, width, dilations=dilations)

        # MS-D Pytorch initializes at zero so pruning skips them very first time
        # becayse we don't train in this test function...
        model.msd.final_layer.linear.weight.data.fill_(0.1)

        # Define pruning percentage and number of steps
        perc = random.uniform(0.05, 0.5)
        nsteps = random.randint(1, 10)

        # Running fine-tuning pruning procedure
        tot_perc = 1.0
        percentage = np.exp(np.log(perc)/float(nsteps))
        for step in range(nsteps):
            tot_perc *= percentage
            model = lean.IndivL1_Global_MSD(model, tot_perc, Redun=False, verbose=False)
            prune_frac = lean.fraction_pruned_convs_MSD(model)[2]

        # Check if number of pruned convolutions is correct
        assert abs((1-prune_frac-perc)/perc) < eps,"Test failed for L1 finetuning!"

def ttest_l1_indiv_UNet(tries=1, eps=0.05):
    """Test whether L1 individual filter pruning
        actually prunes the correct amount of 
        convolutions for U-Net4.
        
    We check that:
    - Pruning method prunes the correct amount.
    """
    for t in range(tries):
        # Create random MSD network
        c_in = random.randint(1,6)
        num_labels = random.randint(1,6)

        model = UNet4AvgSegmModel(c_in, num_labels)

        # Define pruning percentage and number of steps
        perc = random.uniform(0.05, 0.5)
        nsteps = random.randint(1, 10)

        # Running fine-tuning pruning procedure
        tot_perc = 1.0
        percentage = np.exp(np.log(perc)/float(nsteps))
        for step in range(nsteps):
            tot_perc *= percentage
            model = lean.IndivL1_Global_UNet4(model, tot_perc, Redun=False, verbose=False)
            prune_frac = lean.fraction_pruned_convs_UNet4(model)[2]

        # Check if number of pruned convolutions is correct
        assert abs((1-prune_frac-perc)/perc) < eps,"Test failed for L1 finetuning!"

def ttest_SV_indiv_UNet(tries=1, eps=0.05):
    """Test whether individual filter pruning with operator norm pruning
        (spectral norm/largest Singular Value) actually prunes the 
        correct amount of convolutions for U-Net4.
        
    We check that:
    - Pruning method prunes the correct amount.
    """
    for t in range(tries):
        # Create random MSD network
        c_in = random.randint(1,6)
        num_labels = random.randint(1,6)

        model = UNet4AvgSegmModel(c_in, num_labels)

        # Define pruning percentage and number of steps
        perc = random.uniform(0.05, 0.5)
        nsteps = random.randint(1, 10)

        # Running fine-tuning pruning procedure
        tot_perc = 1.0
        percentage = np.exp(np.log(perc)/float(nsteps))
        for step in range(nsteps):
            tot_perc *= percentage
            model = lean.IndivSV_Global_UNet4(model, tot_perc, Redun=False, verbose=False)
            prune_frac = lean.fraction_pruned_convs_UNet4(model)[2]

        # Check if number of pruned convolutions is correct
        assert abs((1-prune_frac-perc)/perc) < eps,"Test failed for SV finetuning!"

def ttest_LEAN_UNet(tries=1, eps=0.05):
    """Test whether LEAN pruning
        actually prunes the correct amount of 
        convolutions for U-Net4.
        
    We check that:
    - Pruning method prunes the correct amount.
    """
    for t in range(tries):
        # Create random MSD network
        c_in = random.randint(1,6)
        num_labels = random.randint(1,6)

        model = UNet4AvgSegmModel(c_in, num_labels)

        # Define pruning percentage and number of steps
        perc = random.uniform(0.05, 0.5)
        nsteps = random.randint(1, 10)

        # Running fine-tuning pruning procedure
        tot_perc = 1.0
        percentage = np.exp(np.log(perc)/float(nsteps))
        for step in range(nsteps):
            tot_perc *= percentage
            model = lean.LEAN_UNet4(model, tot_perc, Redun=False, verbose=False)
            prune_frac = lean.fraction_pruned_convs_UNet4(model)[2]

        # Check if number of pruned convolutions is correct
        assert abs((1-prune_frac-perc)/perc) < eps,"Test failed for LEAN pruning!"

def ttest_l1_indiv_ResNet(tries=1, eps=0.05):
    """Test whether L1 individual filter pruning
        actually prunes the correct amount of 
        convolutions for ResNet50.
        
    We check that:
    - Pruning method prunes the correct amount.
    """
    for t in range(tries):
        # Create random MSD network
        c_in = random.randint(1,6)
        num_labels = random.randint(1,6)

        model = ResNet50AvgSegmentationModel(c_in, num_labels)

        # Define pruning percentage and number of steps
        perc = random.uniform(0.05, 0.5)
        nsteps = random.randint(1, 10)

        # Running fine-tuning pruning procedure
        tot_perc = 1.0
        percentage = np.exp(np.log(perc)/float(nsteps))
        for step in range(nsteps):
            tot_perc *= percentage
            model = lean.IndivL1_Global_ResNet50(model, tot_perc, Redun=False, verbose=False)
            prune_frac = lean.fraction_pruned_convs_ResNet50(model)[2]
        
        # Check if number of pruned convolutions is correct
        assert abs((1-prune_frac-perc)/perc) < eps,"Test failed for L1 finetuning!"

def ttest_SV_indiv_ResNet(tries=1, eps=0.05):
    """Test whether individual filter pruning with operator norm pruning
        (spectral norm/largest Singular Value) actually prunes the 
        correct amount of convolutions for ResNet50.
        
    We check that:
    - Pruning method prunes the correct amount.
    """
    for t in range(tries):
        # Create random MSD network
        c_in = random.randint(1,6)
        num_labels = random.randint(1,6)

        model = ResNet50AvgSegmentationModel(c_in, num_labels)

        # Define pruning percentage and number of steps
        perc = random.uniform(0.05, 0.5)
        nsteps = random.randint(1, 10)

        # Running fine-tuning pruning procedure
        tot_perc = 1.0
        percentage = np.exp(np.log(perc)/float(nsteps))
        for step in range(nsteps):
            tot_perc *= percentage
            model = lean.IndivSV_Global_ResNet50(model, tot_perc, Redun=False, verbose=False)
            prune_frac = lean.fraction_pruned_convs_ResNet50(model)[2]

        # Check if number of pruned convolutions is correct
        print("DONE RESNET50")
        assert abs((1-prune_frac-perc)/perc) < eps,"Test failed for SV finetuning!"

def ttest_LEAN_ResNet(tries=1, eps=0.05):
    """Test whether LEAN pruning
        actually prunes the correct amount of 
        convolutions for ResNet50.
        
    We check that:
    - Pruning method prunes the correct amount.
    """
    for t in range(tries):
        # Create random MSD network
        c_in = random.randint(1,6)
        num_labels = random.randint(1,6)

        model = ResNet50AvgSegmentationModel(c_in, num_labels)

        # Define pruning percentage and number of steps
        perc = random.uniform(0.05, 0.5)
        nsteps = random.randint(1, 10)

        # Running fine-tuning pruning procedure
        tot_perc = 1.0
        percentage = np.exp(np.log(perc)/float(nsteps))
        for step in range(nsteps):
            tot_perc *= percentage
            model = lean.LEAN_ResNet50(model, tot_perc, Redun=False, verbose=False)
            prune_frac = lean.fraction_pruned_convs_ResNet50(model)[2]

        # Check if number of pruned convolutions is correct
        assert abs((1-prune_frac-perc)/perc) < eps,"Test failed for LEAN pruning!"


if __name__ == '__main__':
    #NOTE: These are called ttest so that py.test does not
    #  automatically run them.
    #ttest_l1_indiv_UNet()
    #ttest_SV_indiv_UNet()
    #ttest_LEAN_UNet()
    #ttest_l1_indiv_ResNet()
    #ttest_SV_indiv_ResNet()
    #ttest_LEAN_ResNet()

    test_LEAN_indiv_MSD3x3()
    test_LEAN_indiv_MSD()
    test_SV_indiv_MSD3x3()
    test_SV_indiv_MSD()
    test_l1_indiv_MSD3x3()
    test_l1_indiv_MSD()
