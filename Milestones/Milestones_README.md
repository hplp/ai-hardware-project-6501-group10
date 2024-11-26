# Milestones 

## Milestone 1: Model Training

LeNet and AlexNet CNN architecture were selected.

We trained the LeNet model on the MNIST data set and the AlexNet model on the CIFAR-10 data set using PyTorch.

Trained models saved

  - lenet_FP32.pth/alexnet_FP32.pth

Table 1 shows the accuracies of LeNet and AlexNet.

<u>Table 1: Accuracies of LeNet and AlexNet Models</u>

| Quantized Model| Dataset | Accuracy |
|-----------------|-----------------|-----------------|
|   lenet_FP32.pth   | MNIST    | 98.17%    |
|   alexnet_FP32.pth   | CIFAR-10    | ?    |


  

(Link - https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Training/LeNet_quantized_withONNX.ipynb)   
(Link - https://github.com/hplp/ai-hardware-project-6501-group10/tree/main/Training/Trained%20Models)



## Milestone 2: Post-Training Quantization

We do layer-by-layer int4, int8, and in16 quantization for the models.
The accuracies of each quantized model were calculated. Table 2 shows the accuracies of each quantized model for LeNet. Table 3 shows the accuracies of each quantized model for AlexNet.

Quantized models saved

  - lenet_int16.pth/ alexnet_int16.pth
  - lenet_int8.pth/ alexnet_int8.pth
  - lenet_int4.pth/ alexnet_int4.pth


<u>Table 2: Accuracies of Quantized LeNet Models</u>

| Quantized Model| Dataset | Accuracy |
|-----------------|-----------------|-----------------|
|   lenet_int16.pth   | MNIST    | 98.16%    |
|   lenet_int8.pth   | MNIST    | 98.15%    |
|   lenet_int4.pth   | MNIST    | 96.93%    |

<u>Table 3: Accuracies of Quantized AlexNet Models</u>

| Quantized Model| Dataset | Accuracy |
|-----------------|-----------------|-----------------|
|   alexnet_int16.pth   | CIFAR-10    | ?    |
|   alexnet_int8.pth   | CIFAR-10    | ?    |
|   alexnet_int4.pth   | CIFAR-10    | ?    |


We exported quantized models to ONNX and then will use them for deployment in Accelerator.

- lenet_int16.onnx/ alexnet_int16.onnx
- lenet_int8.onnx/ alexnet_int8.onnx
- lenet_int4.onnx/ alexnet_int4.onnx

(Link - https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Training/LeNet_quantized_withONNX.ipynb) 
(Link - https://github.com/hplp/ai-hardware-project-6501-group10/tree/main/Training/Trained%20Models)



## Milestone 3: NVDLA Deployment

The following steps need to be followed in order to simulate the inference of the trained model on NVDLA,
1.	Install dependencies - g++ cmake libboost-dev python-dev libglib2.0-dev libpixman-1-dev liblua5.2-dev swig libcap-dev libattr1-dev
2.	Install SystemC and Perl packages.
3.	Download and build the NVDLA Cmodel.
4.	Build and install the NVDLA Virtual Simulator.
5.	Run the Virtual Simulator.

### Our Approach
NVDLA primarily uses Synopsys VCS as a simulation tool for which we need a license. But it also offers us the option to use the open-source Verilator as an alternative to run the simulations.
1.	Using Verilator
-	Since a specific environment needs to be prepared to build the NVDLA Cmodel, we first tried Verilator, so we could set it up on a personal device. They have recommended using an Ubuntu machine to set up the environment.
-	There are multiple syntax issues in the Verilog code provided in their repository making it incompatible with Verilator,
  -	Inconsistent timescale definitions in Verilog modules (can be manually fixed)
  -	Parameters defined inside functions of Verilog modules (cannot be easily fixed since multiple functions of the same module have different definitions for the parameters)

2.	Using Synopsys VCS
-	This was set up in the ECE server to use the Synopsys tools. Tried installing most of the dependencies in a Miniconda environment and the remaining dependencies into User folders.
-	We get compile issues when trying to compile the HLS code using g++. 
  -	The gcc version installed in the ECE server is 8.5.0. But the NVDLA needs gcc version 4.8.4/4.9.3, which is a much older version.

### Next steps
These are the options we have now,
-	Request Synopsys tool license to be used in a personal device and set up an environment with older compiler versions that NVDLA is compatible with. (NVDLA hasn’t been updated for the last 6 years)
-	Look for an older version NVDLA where the verilator simulations were initially tested.
-	Switch to a different virtual simulator platform like tvm-vta(apache) or scale-sim-v2.


