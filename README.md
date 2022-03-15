# Nimble-AI-Programming-Challenge

---
## Approach



### Reference papers
1) ttps://arxiv.org/pdf/1804.03999.pdf
2 https://www.nature.com/articles/s41598-021-90428-8
3 https://www.mdpi.com/2313-433X/7/12/269/pdf
4) 



---
## Things to try given more time
 
>> Try more types of different losses for imbalanced classes in segmentation

>> Change model parameters  

>> Try Maskrcnn instead of current Architectures(although might need more data) 

>> Use more data augmentation

>> Increase resolutiuon of input images

>> Similar to tumor segmentation - class imbalance - dice loss/ feed patches in unet

>> check effect of depth on different models

>> Increase learning rate


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
