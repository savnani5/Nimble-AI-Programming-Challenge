# Accuracy improvement changes

# >> Increase resolutiuon 
# >> Use different loss
# >> Use data augmentation
# >> Similar to tumor segmentation - class imbalance - dice loss/ feed patches in unet
# >> Use depth with RGB in input
# >> Try attention unet/maskrcnn instead of unet
# >> Increase learning rate
# >> Write script to get depth to predict
# >> Try adam - no weight decay
# >> More data augmentation
# >> Try nested unet after one more instance
# >> had issues ssh'ing to with the aws server

# import the necessary packages
import torch
import os
# base path of the dataset
DATASET_PATH = os.path.join("data", "train")

# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "color")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "label")
DEPTH_DATASET_PATH = os.path.join(DATASET_PATH, "depth")

EVAL_IMAGE_DATASET_PATH = os.path.join("data", "test", "color")
EVAL_DEPTH_DATASET_PATH = os.path.join("data", "test", "depth")
EVAL_LABEL_DATASET_PATH = os.path.join("data", "test", "test-labels")

# define the test split
TEST_SPLIT = 0.15
# determine the device to be used for training and evaluation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001 #0.0001
NUM_EPOCHS = 60 #300
BATCH_SIZE = 8   # 32 out of memory  # unet - 2
WEIGHT_DECAY = 0.0001 #0.00001

# define the input image dimensions 
INPUT_IMAGE_HEIGHT = 128 #240
INPUT_IMAGE_WIDTH = 128  #240


# define threshold to filter weak predictions // use this for heatmap -- please
THRESHOLD = 0.3

# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "suction_loc.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])


