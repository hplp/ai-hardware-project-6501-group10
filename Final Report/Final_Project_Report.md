# **Performance Evaluation of CNN on Open-Source Platforms**
## Team Name
Q10

## Team Members
- Melika Morsali Toshmnaloui
- Kavish Nimnake Ranawella
- Hasantha Ekanayake

## 1. Project Overview

#### Project Title: Performance Evaluation of CNN on Open-Source Platforms  
#### Repository URL: [GitHub Repository Link](https://github.com/hplp/ai-hardware-project-6501-group10)

#### Description
This project has two main objectives: one focused on software and the other on hardware.
We use PyTorch to investigate how post-training quantization affects model accuracy. Specifically, we apply post-training quantization at different precision levels—INT8 and INT16—to evaluate how these adjustments influence model performance.
On the hardware side, we employ open-source platforms like the NVDLA Simulator and Scale-Sim. With the NVDLA Simulator, we analyze execution time, and using Scale-Sim, we examine metrics such as utilization, mapping efficiency, cycles, and bandwidth.
By combining software and hardware analyses, we can assess both the accuracy and efficiency of deploying these models on hardware.

---

## 2. Objectives
The primary goals of this project are:
1. Software Analysis with PyTorch: Investigate the impact of post-training quantization on model accuracy at different precision levels (INT8, INT16) using PyTorch.
2. Hardware Analysis with NVDLA Simulator: Analyze execution time for quantized models deployed on the NVDLA simulator to understand their hardware performance.
3. Hardware Analysis with Scale-Sim: Evaluate metrics such as utilization, mapping efficiency, cycles, and bandwidth to understand how models perform in terms of hardware resource use.



---

## 3. Software Side

### Training Models
In this project, we trained three models—LeNet, AlexNet, and EfficientNet—using full-precision FP32 with PyTorch. The Jupyter Notebook files for these models can be found [here]((Final_Report/Training_and_Quantization).
The details of their accuracy are presented in Table 1.
#### Table 1: Models Accuracy Table

| **Models**      | **Dataset** | **FP32** | 
|------------------|-----------|-----------------------------|
| LeNet           | MNIST   | 98.16%  |                    
| AlexNet         | MNIST   | 99.19%  |                    
| EfficientNet    | MNIST   | 98.17%  |   
| AlexNet         | CIFAR-10   | 83.73%  |                    
| EfficientNet    | CIFAR-10  | 91.94%  |  

### Model Training Metrics

#### Loss vs Epochs
<p align="center">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/output(5).png" width="500" height="300">
</p>

#### Accuracy vs Epochs
<p align="center">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/output(4).png" width="500" height="300">
</p>


### Post-Training Quantization 

We implemented post-training quantization, which applies quantization after the model has been trained. Our approach utilizes layer-wise quantization, meaning that each layer is quantized independently. Both the inputs and weights are quantized to reduce memory usage and computational demands. Each tensor has its own scaling factor, allowing for precise mapping of values into the quantized range at the layer level.

However, for EfficientNet, due to its more complex structure, we could not utilize layer-wise quantization. Instead, we employed PyTorch functions to quantize the models. Additionally, we used the same approach for AlexNet on the CIFAR-10 dataset, given its complexity.


This analysis compares the effects of quantization (INT8 and INT16) across various models (LeNet, AlexNet, EfficientNet) and datasets (MNIST, CIFAR-10). It covers performance, resilience to quantization, and implementation approaches.

---

#### Table 2: Summary of Results

| **Model**       | **Dataset** | **FP32 Accuracy** | **INT16 Accuracy** | **INT8 Accuracy** | **Notes**                                                                 |
|------------------|-------------|--------------------|---------------------|--------------------|---------------------------------------------------------------------------|
| **LeNet**        | MNIST       | 98.16%            | 98.16%             | 98.06%            | Minor degradation with INT8; INT16 identical to FP32.                   |
| **AlexNet**      | MNIST       | 99.19%            | 98.64%             | 98.60%            | Slight drop with INT16/INT8. Robust to quantization on MNIST.           |
| **AlexNet**      | CIFAR-10    | 83.73%            | 83.73%/50.00%      | 83.74%/49.83%     | Dynamic quantization retains accuracy; manual approach suffers.         |
| **EfficientNet** | MNIST       | Not provided      | Not provided       | Not provided      |  |
| **EfficientNet** | CIFAR-10    | Not provided      | Not provided       | Not provided      |               |

---


#### Impact of Quantization
- **LeNet (MNIST)**: Minimal performance degradation, even with INT8. The simplicity of the model and dataset makes it resilient to quantization.
- **AlexNet (MNIST)**: Slight accuracy drop in INT8 and INT16. The model handles MNIST well, even with reduced precision.
- **AlexNet (CIFAR-10)**: Dynamic quantization retains accuracy, but manual per-layer quantization suffers from a significant drop. CIFAR-10's complexity magnifies the challenges of poorly optimized quantization.

#### Quantization Approaches
- **Manual Per-Layer Quantization**:
  - Accuracy depends heavily on correct scaling and rounding. Errors propagate across layers, leading to accuracy drops, especially on complex datasets like CIFAR-10.
- **Dynamic Quantization (PyTorch)**:
  - Handles layer-wise quantization automatically, including scaling, rounding, and optimization. Performs significantly better, especially on complex datasets.

#### Dataset Complexity
- **MNIST**: As a simple dataset, it is highly tolerant to quantization. Even manual approaches work well.
- **CIFAR-10**: Requires more sophisticated quantization techniques like dynamic quantization due to its complexity and higher resolution.

#### Model Architecture
- **LeNet**: Simple architecture ensures resilience to aggressive quantization (e.g., INT8).
- **AlexNet**: Shows resilience on MNIST but needs optimized quantization for CIFAR-10.
- **EfficientNet**: Likely robust due to its advanced architecture, but details were unavailable in this analysis.

---

### Discussion

#### Quantization Suitability
- INT8 provides a good trade-off between accuracy and computational efficiency. INT16 closely matches FP32 but offers less computational advantage.
- Manual quantization should only be used for exploratory or learning purposes. Optimized frameworks like PyTorch's quantization are highly recommended for practical applications.

#### Recommendations
- Use dynamic quantization for complex models and datasets.
- For simpler tasks, manual quantization may suffice but requires careful implementation to avoid severe accuracy drops.

#### Future Directions
- Explore quantization-aware training (QAT) for even better accuracy retention.
- Investigate hybrid quantization approaches, where sensitive layers (e.g., first and last layers) retain higher precision (e.g., FP16) while others use INT8.

---

### Conclusion

Quantization is a powerful technique to reduce model size and inference latency. However, its success depends on the dataset, model architecture, and quantization method. Dynamic quantization outperforms manual approaches, especially on complex datasets like CIFAR-10. Models like LeNet and AlexNet demonstrate strong resilience to quantization, making them suitable for edge and resource-constrained environments.

### ONNX conversion 
 we can convert our trained model into ONNX format, which stands for Open Neural Network Exchange. ONNX is an open format that makes it easy to move models between different AI frameworks, like PyTorch, Keras, and others, which you see on the left.
ONNX acts as a bridge in the middle, letting us take a model trained in one framework and deploy it on various devices.

All of the parts of the training, quantization, and ONNX conversion can be found in the related Jupyter notebook [ here ](https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Training%20quantization)


## 4. Hardware Sides
### 1. NVDIA Deep Learning Accelerator (NVDLA):

<p align="center">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/nvdla_flow.png" alt="nvdla overview" title="nvdla overview" width="500">
</p>

#### Compiler
Unfortunately, NVIDIA hasn't added support for ONNX models. Currently, it only support Caffe models.

The NVDLA compiler needs the following files from the Caffe models,
- .prototxt - contains the architecture of the Caffe model
- .caffemodel - contains the trained weights of the Caffe model

It also accepts optional arguements to customize the compilation process,
- cprecision (fp16/int8) - compute precision
- configtarget (nv_full/nv_large/nv_small) - target NVDLA configuration
- calibtable - calibration table for INT8 networks
- quantizationMode (per-kernel/per-filter) - quantization mode for INT8
- batch - batch size
- informat (ncxhwx/nchw/nhwc) - format of the input matrix
- profile (basic/default/performance/fast-math) - computation profile

NVIDIA offers multiple predefined NVDLA configurations. More details will provided under the virtual platform.

The calibtable expects a .json file with the scale values used for the quantization. TensorRT can be used to dump the scale values to text file ([link](https://github.com/NVIDIA/TensorRT/tree/release/5.1/samples/opensource/sampleINT8) explains this) and [calib_txt_to_json.py](https://github.com/nvdla/sw/tree/master/umd/utils/calibdata/calib_txt_to_json.py) can be used to convert this to NVDLA JSON format.

An NVDLA loadable (.nvdla) is created during compilation which is used during runtime.

We tried this compilation for multiple online available Caffe models. But, only 2 of them compilled properly. Most of them failed because the .prototxt was not compatible with the NVDLA compiler. None of the models available in the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) were compatible.

Further details on this can be found [here](https://github.com/hplp/ai-hardware-project-6501-group10/tree/main/nvdla/compilation).

#### Deployment

There are multiple options available to deploy NVDLA,
- GreenSocs QBox based Virtual Simulator
- Synopsys VDK based Virtual Simulator
- Verilator based Virtual Simulator
- FireSim FPGA-accelerated Simulator (AWS FPGA)
- Emulation on Amazon EC2 “F1” environment (AWS FPGA)

Synopsys needs licensed tools. Verilator is open source, but NVDLA was not properly documented to use this. Out of the virtual simulators, GreenSocs QBox is the most documented free simulator available. We have more control over our environment when we use this.

FireSim is still a simulator, but it is accelerated on an FPGA in Amazon EC2 "F1" instance. Instead of running the simulator on the FPGA, we can also deploy the NVDLA hardware design on that FPGA, which is the fifth option.

However, NVIDIA has stopped maintaining these 5-6 years ago.

##### GreenSocs QBox based Virtual Simulator

<p align="center">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/nvdla_vp.png" alt="nvdla virtual" title="nvdla virtual" width="500">
</p>

This virtual platform simulates a QEMU CPU model (ARMv8) with a SystemC model of NVDLA. They have offered 3 predefined hardware configuration for NVDLA as follows,
- ``nv_full``
    - Full precision version (tested for INT8 and FP16 precisions).
    - Has 2048 8-bit MACs (1024 16-bit fixed- or floating-point MACs).
- ``nv_large``
    - Deprecated version (replaced by nv_full).
    - Supports INT8 and FP16 precisions.
- ``nv_small``
    - Targets smaller workload.
    - Very limited feature support.
    - Has 64 8-bit MACs.
    - Only supports INT8 precision.
    - Headless implementation (no microcontroller for task management).
    - No secondary SRAM support for caches.

<p align="center">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/nvdla_config.png" alt="nvdla config" title="nvdla config" width="500">
</p> 

We have two options when running NVDLA on this virtual simulator,
- Build our own virtual platform
- Use the one available with a docker

We first went with building our own platform. We followed the instructions on this [link](https://nvdla.org/vp.html). These are the challenges we faced,
- Updating submodules of ``qbox`` inside the [nvdla/vp](https://github.com/nvdla/vp)
    - The link used by the submodules needs to be changed to ``https://`` links.
    - There were times that the ``pixman`` submodule refused connection.
- Compilation of the SystemC model
    - It needs an Ubuntu environment, but latest versions cannot be used (need Ubuntu 14.04).
    - It need gcc/g++ 4.8.4 which is available on Ubuntu 14.04 (ECE servers currently use gcc 8.x).
    - We ran Ubuntu 14.04 on a Virtual Machine build the virtual platform.
- Building Linux Kernel
    - We need to use exactly 2017.11 version of [buildroot](https://buildroot.org/download.html) inorder to avoid any errors.

It is easier to run the virtual platform on a docker to avoid these complications.

The runtime capabilities in this platform is limited to running the simulation for a single image.

##### Emulation on Amazon EC2 “F1” environment (AWS FPGA)

<p align="center">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/nvdla_aws.png" alt="nvdla aws" title="nvdla aws" width="500">
</p>

Here as you can see, the QEMU CPU model is simulated on a OpenDLA Virtual Platform. However, instead of a SystemC model, NVDLA is deployed on the FPGA in RTL.

The runtime capabilities are increased on this platform. We can run hardware regressions and collect data to evaluate the performance and energy efficiency of the NVDLA design. Here are some data collected and displayed on the [NVDLA website](https://nvdla.org/primer.html),

<p align="center">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/nvdla_results.png" alt="nvdla results" title="nvdla results" width="500">
</p>

#### Runtime

During runtime, the NVDLA loadable goes through multiple abstraction layers before reaching the NVDLA hardware. They are as follows,
- User-Mode Driver (UMD) - Loads the loadable and submits inference job to KMD.
- Kernel-Mode Driver (KMD) - Configures functional blocks on NVDLA and schedules operations according to the inference jobs received.

There [nvdla/sw](https://github.com/nvdla/sw) repository provides the resources to build these drivers, but we ran into errors when trying to build it. So, we used prebuilt versions of it for this project. 

After starting the virtual simulator platform, the UMD and KMD should be loaded. Then we run the NVDLA loadable on it. They have provided 4 modes for the runtime,
1. Run with only the NVDLA loadable.
    - It runs a sanity test with input embedded in it.
    - Will give the execution time at the end of it.
2. Run with NVDLA loadable and a sample image.
    - It runs a network test to generate the output for given image.
    - Will give the execution time along with the output generated.
    - The nv_full configuration expects a 4-channel image as the input image.
3. Run in server mode (did not test).
    - Can run inference jobs on the NVDLA by connecting to it as a client.
4. Run hardware regressions (did not test).
    - Not possible with the same runtime application used by the previous options (the flow is different).
    - The hardware regressions cannot be run on any virtual platform (needs an FPGA implementation).

The runtime application can also be changed and built again. We tried this, but it gives errors.

Further details on this can be found [here](https://github.com/hplp/ai-hardware-project-6501-group10/tree/main/nvdla/runtime).

#### Results

Since we deployed only on a virtual simulator platform, we only have results for the single image simulations. These simulations display the execution time on the terminal at the end of the simulations. An output.dimg file will also be created with the output of the model. The execution times we got are as follows,

| **Model**     | **FP16**       | **INT8**       | 
|---------------|----------------|----------------|
| LeNet         | 5,633 hrs      | 10,401 hrs     |                    
| ResNet-50     | 5,743,922 hrs  | 4,834,791 hrs  |

These numbers are way off. Obviously, the simulation didn't run for 5 million hours. The LeNet finishes within 10 minutes and ResNet-50 run for upto 5 hours. Instead of looking at the absolute values, we compared the relative values.

The ResNet-50 has some improvement when running on INT8 quantized mode, but the performance has worsened for LeNet. We assumed that this is because LeNet is a very small model and the overhead introduced to the NVDLA by handling a fixed-point quantization outweighs any performance gained by lighter computations. There might also be the possibility that LeNet is even too small to consume all 2048 8-bit MACs available in the ``nv_full`` configuration. ``nv_small`` configuration might be a better fit for the LeNet, but we cannot do this comparison on ``nv_small`` because it doesn't support FP16 precision.

The terminal outputs of these simulations along with the loadables used are given [here](https://github.com/hplp/ai-hardware-project-6501-group10/tree/main/nvdla/runtime).

#### Conclusion

The ResNet-50 has some improvement when running on INT8 quantized mode, but the performance has worsened for LeNet. We assumed that this is because LeNet is a very small model and the overhead introduced to the NVDLA by handling a fixed-point quantization outweighs any performance gained by lighter computations. There might also be the possibility that LeNet is even too small to consume all 2048 8-bit MACs available in the ``nv_full`` configuration. ``nv_small`` configuration might be a better fit for the LeNet, but we cannot do this comparison on ``nv_small`` because it doesn't support FP16 precision.

The quantization is handled by the NVDLA compiler itself. So, we can't expect a different output unless we make changes to the NVDLA framework.

#### Future Work

We need to get more reliable results using a hardware implementation on AWS FPGA. However, it still needs a OpenDLA virtual platform to emulate the CPU. We might run into issues because we have limited control over downgrading the software when running on AWS servers.

### 2. Scale-Sim:

Scale-sim (Systolic CNN Accelerator Simulator) is a lightweight and highly configurable simulator that gives valuable insights into hardware-level performance, enabling efficient testing and deployment of deep neural networks (DNNs) models without access to physical hardware. Below figure illustrates the architecture and workflow of SCALE-Sim. 

<p align="center">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/scale-sim.png" alt="scale-sim" title="scale-sim" width="500" height="300">
</p>

<sub>Source: SCALE-Sim: Systolic CNN Accelerator Simulator - A. Samajdar et al  <a href="https://arxiv.org/abs/1811.02883">Read the Paper</a></sub>

Key components of SCALE-Sim architecture include:

•  Input Files:
-	Config File: Contains hardware-specific parameters such as array height/width, SRAM sizes, and dataflow (e.g., weight-stationary, output-stationary).
-	DNN Topology File: Specifies the layers of the DNN (e.g., Conv1, Conv2, FC1) that will be simulated.
  
•  Hardware Model:
-	Systolic Array: A grid of processing elements (PEs) designed for matrix multiplications, crucial for DNN computations.
-	SRAM Buffers: Includes:
-	Filter SRAM: Stores weights.
-	IFMAP SRAM: Stores input feature maps.
-	OFMAP SRAM: Stores output feature maps.
	These buffers use double buffering for efficient data transfer.

• Simulation Outputs:
-	Cycle-Accurate Traces: Tracks memory access (SRAM/DRAM reads and writes).
-	Performance Metrics: Reports cycles, bandwidth utilization, and hardware efficiency.


In this project, Scale-Sim used to experiment with different DNN models i.e. LeNet, AlexNet and EfficientNet to evaluate the performances in hardware architecture.

#### Results 

##### Models Performance 

| **Models**      | **Cycles** | **Overall Utilization (%)** | **Mapping Efficiency (%)** |
|------------------|-----------|-----------------------------|----------------------------|
| LeNet           | 20,996    | 11.42                      | 80.08                     |
| AlexNet         | 738,385   | 91.45                      | 96.05                     |
| EfficientNet    | 735,114   | 25.66                      | 58.85     

<p align="center">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/cycles.png" alt="cycles" title="cycles" width="300" height= "200">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/utilization.png" alt="utilization" title="utilization" width="300" height= "200">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/mapping.png" alt="mapping" title="mapping" width="300" height= "200">
</p>

The performance analysis highlights significant differences across the evaluated models. AlexNet demonstrated the highest overall utilization (91.45%) and mapping efficiency (96.05%), making it highly effective for hardware deployment. EfficientNet, despite its advanced architecture, showed moderate mapping efficiency (58.85%) and lower utilization (25.66%), indicating room for optimization. LeNet, being a simpler model, had low utilization (11.42%) but relatively high mapping efficiency (80.08%), making it suitable for lightweight applications. These results emphasize the need to match model complexity with hardware capabilities for optimal performance.

##### SRAM Bandwidth

Definition: SRAM bandwidth refers to the rate at which data can be read from or written to the on-chip SRAM buffers (e.g., IFMAP SRAM, Filter SRAM, OFMAP SRAM) in words per cycle.

In SCALE-Sim, SRAM bandwidth depends on the systolic array configuration and the dataflow being simulated (e.g., weight-stationary, output-stationary).

Formula:

> **SRAM Bandwidth (words/cycle)** = Words Transferred per Cycle (read/write)

Comparison of SRAM Bandwidth for models
<p align="center">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/Filter SRAM BW.png" alt=" " title="cyles" width="300" height= "200">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/IFMAP SRAM BW.png" alt=" " title=" " width="300" height= "200">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/OFMAP SRAM BW.png " alt=" " title=" " width="300" height= "200">
</p>


##### DRAM Bandwidth

Definition: DRAM bandwidth refers to the rate at which data can be read from or written to the off-chip DRAM memory in words per cycle.

It is influenced by the size of the off-chip data transfer, the DRAM bus width, and the DRAM-to-SRAM communication latency.


Formula:

 > **DRAM Bandwidth (words/cycle)** =
  (Bus Width (bits)/ Word Size (bits)) × (DRAM Clock Speed /Array Clock Speed)

Comparison of DRAM Bandwidth for models 
<p align="center">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/Filter DRAM BW.png" alt=" " title=" " width="300" height= "200">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/IFMAP DRAM BW.png" alt=" " title=" " width="300" height= "200">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/OFMAP DRAM BW.png " alt=" " title=" " width="300" height= "200">
</p>




All the results and files for Scale-Sim are included in this folder [here](https://github.com/hplp/ai-hardware-project-6501-group10/tree/main/Scale-Sim).


## 5. Key Takeaways and Challenges  


### Key Takeaways
- Layer-wise Quantization: The project allowed better control over scaling factors, maintaining accuracy.
- Platform Utilization: The NVDLA gave detailed execution times; Scale-Sim helped assess utilization and mapping efficiency.
- Data Preprocessing: Normalization and augmentation improved model robustness, especially for EfficientNet on CIFAR-10.
- Simulator Insights: Scale-Sim provided critical insights into hardware-level metrics like cycles, utilization, and bandwidth efficiency, making it valuable for DNN deployment evaluations.
- Integration of Software and Hardware Analyses: The project effectively combined PyTorch-based quantization techniques with hardware performance evaluations using simulators like NVDLA and Scale-Sim.

### Challenges
- ONNX to Caffe Conversion: Lack of ONNX support in the NVDLA required converting to Caffe, which remains problematic.
- Outdated Simulators: The NVDLA's outdated emulators forced us to use alternatives like Scale-Sim.
- Complex Setup: Docker and QEMU setups for GreenSocs QBox required extensive troubleshooting.
- Quantization Limitations: Scale-Sim lacked support for quantization.
- Quantization Complexity with Scale-Sim: Scale-Sim lacked native support for quantized models.


## 6. Conclusion
This project evaluated CNN performance using software-based quantization and hardware simulations on open-source platforms. It showed that post-training quantization can reduce computation and memory needs while maintaining accuracy for simpler models. Tools like NVDLA and Scale-Sim provided insights into execution time, utilization, and efficiency.

While challenges like outdated tools and complex setups were encountered, the project demonstrated the potential of combining software and hardware analyses for efficient model deployment. These findings provide a foundation for future improvements in quantization techniques and hardware simulation workflows.

## 7. References
Columbia University. (n.d.). Guide – How to: integrate a third-party accelerator (e.g. NVDLA). Retrieved December 9, 2024, from https://www.esp.cs.columbia.edu/docs/thirdparty_acc/thirdparty_acc-guide/

NVIDIA. (n.d.). NVDLA: NVIDIA Deep Learning Accelerator. Retrieved December 9, 2024, from https://nvdla.org

SadhaShan. (n.d.). NVDLA GitHub Repository. GitHub. Retrieved December 9, 2024, from https://github.com/SadhaShan/NVDLA

Samajdar, A., Zhu, Y., Whatmough, P., Mattina, M., & Krishna, T. (2018). SCALE-Sim: Systolic CNN Accelerator Simulator. arXiv. https://doi.org/10.48550/arXiv.1811.02883
