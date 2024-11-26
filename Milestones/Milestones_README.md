# Milestones 

## Milestone 1: Model Training

LeNet CNN architechture was selected
Trained the model on MNIST data set on pytorch
Trained model saved (Link - https://github.com/hplp/ai-hardware-project-6501-group10/tree/main/Training/Trained%20Models)

Below are the quantized models 
  - lenet_int32.pth
  - lenet_int16.pth
  - lenet_int8.pth
  - lenet_int4.pth


## Milestone 2: Post-Training Quantization

Accuraccies of each quantized model was calculated. Table 1 shows the accuracies of each quantized models.

<u>Table 1 : Accuracies of Quantized Models</u>

| Quantized Model| Dataset | Accuracy |
|-----------------||-----------------||-----------------|
|   lenet_int32.pth   | MNIST    | 98.17%    |
|   lenet_int16.pth   | MNIST    | 98.16%    |
|   lenet_int8.pth   | MNIST    | 98.15%    |
|   lenet_int4.pth   | MNIST    | 96.93%    |


