# Milestones 

## Milestone 1: Model Training

LeNet CNN architechture was selected.

Trained the model on MNIST data set using PyTorch.

Trained model saved

  - lenet_int32.pth
  - lenet_int16.pth
  - lenet_int8.pth
  - lenet_int4.pth
    
(Link - https://github.com/hplp/ai-hardware-project-6501-group10/tree/main/Training/Trained%20Models)



## Milestone 2: Post-Training Quantization

Accuraccies of each quantized model was calculated. Table 1 shows the accuracies of each quantized models.

<u>Table 1 : Accuracies of Quantized Models</u>

| Quantized Model| Dataset | Accuracy |
|-----------------|-----------------|-----------------|
|   lenet_int32.pth   | MNIST    | 98.17%    |
|   lenet_int16.pth   | MNIST    | 98.16%    |
|   lenet_int8.pth   | MNIST    | 98.15%    |
|   lenet_int4.pth   | MNIST    | 96.93%    |

Exported quantized model to ONNX.

- lenet_int16.onnx
- lenet_int4.onnx
- lenet_int8.onnx

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


