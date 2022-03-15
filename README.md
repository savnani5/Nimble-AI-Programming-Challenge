# Nimble-AI-Programming-Challenge

---
## Approach

### Implmentation Details:

1) I tried multiple architectures like vanilla unet, Attention unet, Residual Recurrent Attention Unet, Nested Unet etc. Nested Unet performed the best. 

2) I also tried multiple loss functions i.e Binary Crossentropy Dice Loss, IOU loss, Focal-travesky loss etc. and the Binary Crossentropy Dice Loss worked bbest for the case. 

3)  Also I experimented with inputting only RGB images and also concatenating depth with RGB and inputiing them, the latter performed better.

4) For the dataloader I did data augmentation using Resize, Horizontal Flip, Vertical Flip, Rotation, etc. transforms. 

5) Also for training SGD with momentum with lr scheduler, lr of 0.001, batch size of 8 and 60 peochs worked better than adam with weight decay and same hyperparameters. 

_**NOTE: Faced mulliple issues connecting with the AWS instance and ssh keys and got access on second last day only, so initially used my GPU (RTX 2060), and used both on the last day for parallelely testing variations to improve accuracy.**_


### Reference papers
1) https://arxiv.org/pdf/1804.03999.pdf
2) https://www.nature.com/articles/s41598-021-90428-8
3) https://www.mdpi.com/2313-433X/7/12/269/pdf
4) https://iopscience.iop.org/article/10.1088/1742-6596/1213/2/022003/pdf 


---
## Things to try given more time
 
1) Try more types of different losses for imbalanced classes in segmentation

2) Change model parameters  

3) Try Maskrcnn instead of current Architectures(although might need more data) 

4) Use more data augmentation

5) Increase resolutiuon of input images

6) Similar to tumor segmentation - class imbalance - dice loss/ feed patches in unet

7) check effect of depth on different models

8) Increase learning rate


---
## Instructions for running the code:

1) Downlaod the data folder in the root git folder.
2) config.py has the Model Hyperparameters which can be tweaked.
3) Run below command to train the model:
     ```
    python train.py
    ```
4) Run below command to make predcitions on validation set (you can uncomment the **prepare_plot(orig, gtMask, predMask)** to visualize the results)
    ```
    python predict.py
    ```
5) Run below command to generate results for the provided test set:
    ```
    python generate_result.py
    ```
