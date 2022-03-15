from sympy import interpolate
from metric import BinaryMetrics
import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os


def prepare_plot(origImage, origMask, predMask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask)
    ax[2].imshow(predMask)
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    # set the layout of the figure and display it
    figure.tight_layout()
    plt.show()


def make_predictions(model, imagePath, output):
    # set model to evaluation mode
    model.eval()

    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # resize the image and make a copy of it for visualization
        image = cv2.resize(image, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH))
        orig = image.copy()

        # to float data type, and scale its pixel values
        image = image.astype("float32") / 255.0

        # find the filename and generate the path to depth map
        filename = imagePath.split('\\')[-1]
        depthPath = os.path.join(config.DEPTH_DATASET_PATH, filename)
        depthMap = cv2.imread(depthPath, 0)
        depthMap = cv2.resize(depthMap, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH))
        depthMap = np.expand_dims(depthMap, -1)

        # find the filename and generate the path to ground truth
        # mask
        groundTruthPath = os.path.join(config.MASK_DATASET_PATH, filename)
        
        # load the ground-truth segmentation mask in grayscale mode
        # and resize it
        gtMask = cv2.imread(groundTruthPath, 0)
        gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH))
        
        gtMask_for_prediction = np.expand_dims(gtMask, 0)
        gtMask_for_prediction = gtMask_for_prediction/255.0
        gtMask_for_prediction = torch.tensor(gtMask_for_prediction,dtype=torch.float)

        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)

        depthMap = np.transpose(depthMap, (2, 0, 1))
        depthMap = np.expand_dims(depthMap, 0)
        depthMap = torch.from_numpy(depthMap).to(config.DEVICE)

        # Combine image and depthmap at channel axis
        rgbd = torch.cat((image, depthMap), 1)

        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predMask = model(rgbd).squeeze()
        predMask = torch.sigmoid(predMask)
        prediction_for_metric = predMask
        prediction_for_metric = prediction_for_metric[None, None, :]
        print(prediction_for_metric.shape)
        predMask = predMask.cpu().numpy()
        # filter out the weak predictions and convert them to integers
        predMask[predMask < config.THRESHOLD] = 0

        predMask = predMask*255
        predMask = predMask.astype(np.uint8)
        heatmap = cv2.applyColorMap(predMask, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        predMask = cv2.addWeighted(orig,0.5,heatmap,0.5,0)

        # prepare a plot for visualization
        # prepare_plot(orig, gtMask, predMask)
        
        [pixel_acc, dice, precision, specificity, recall] =output(gtMask_for_prediction, prediction_for_metric)
        print("pixel accuracy: ", pixel_acc, "dice coefficient: ", dice, "precision: ", precision, "specificity: ", specificity, "recall: ", recall)


# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")

# imagePaths = np.random.choice(imagePaths, size=10)
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
# unet = torch.load("output/suction_loc.pth").to(config.DEVICE)
unet = torch.load("output/suction_loc.pth").to(config.DEVICE)
# iterate over the randomly selected test image paths
for path in imagePaths:
    
    # make predictions and visualize the results
    output = BinaryMetrics()
    print(path)
    make_predictions(unet, path, output)

