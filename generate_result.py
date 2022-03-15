from sympy import interpolate
import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
import cv2
import os

def make_predictions(model, imagePath):
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
        filename = imagePath.split(os.path.sep)[-1] 
        depthPath = os.path.join(config.EVAL_DEPTH_DATASET_PATH, filename)
        depthMap = cv2.imread(depthPath, 0)
        depthMap = cv2.resize(depthMap, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH))
        depthMap = np.expand_dims(depthMap, -1)

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
        predMask = predMask.cpu().numpy()
        # filter out the weak predictions and convert them to integers
        predMask[predMask < config.THRESHOLD] = 0


        predMask = predMask*255
        predMask = predMask.astype(np.uint8)
        heatmap = cv2.applyColorMap(predMask, cv2.COLORMAP_JET)
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        predMask = cv2.addWeighted(orig,0.5,heatmap,0.5,0)

        predMask = cv2.resize(predMask, (320,240), interpolation= cv2.INTER_LINEAR)
        # save_prediction
        cv2.imwrite(os.path.join(config.EVAL_LABEL_DATASET_PATH, filename), predMask)

        print(filename, " saved....")
        

# load the image paths in our testing file and randomly select 10
# image paths
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load("output/suction_loc.pth").to(config.DEVICE)

print("[INFO] loading up test image paths...")

for imagePath in glob.glob(config.EVAL_IMAGE_DATASET_PATH + '/*'):    
    make_predictions(unet, imagePath)

