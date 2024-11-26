# Milestones 

## Milestone 1: Model Training

LeNet CNN architechture was selected
Trained the model on MNIST data set on pytorch
Trained model saved

  - lenet_int32.pth
  - lenet_int16.pth
  - lenet_int8.pth
  - lenet_int4.pth


## Milestone 2: Post-Training Quantization

Accuraccies of each quantized model was calculated 

| Quantized Model| Dataset | Accuracy |
|------------------|-----------------|-----------------|
|   lenet_int32.pth   | MNIST    | 98.17%    |
|   lenet_int16.pth   | MNIST    | 98.16%    |
|   lenet_int8.pth   | MNIST    | 98.15%    |
|   lenet_int4.pth   | MNIST    | 96.93%    |


