# **Performance Evaluation of CNN on Open-Source Platforms**
## Team Name

## Team Members
- Melika Morsali Toshmnaloui
- Kavish Nimnake Ranawella
- Hasantha Ekanayake

## Description
This project has two main objectives: one focused on software and the other on hardware.
We use PyTorch to investigate how post-training quantization affects model accuracy. Specifically, we apply post-training quantization at different precision levels—INT8 and INT16—to evaluate how these adjustments influence model performance.
On the hardware side, we employ open-source platforms like the NVDLA Simulator and Scale-Sim. With the NVDLA Simulator, we analyze execution time, and using Scale-Sim, we examine metrics such as utilization, mapping efficiency, cycles, and bandwidth.
By combining software and hardware analyses, we can assess both the accuracy and efficiency of deploying these models on hardware.

---

## Objectives
The primary goals of this project are:
1. Software Analysis with PyTorch: Investigate the impact of post-training quantization on model accuracy at different precision levels (INT8, INT16) using PyTorch.
2. Hardware Analysis with NVDLA Simulator: Analyze execution time for quantized models deployed on the NVDLA simulator to understand their hardware performance.
3. Hardware Analysis with Scale-Sim: Evaluate metrics such as utilization, mapping efficiency, cycles, and bandwidth to understand how models perform in terms of hardware resource use.



---

## Software Side

### Training Models
In this project, we trained three models—LeNet, AlexNet, and EfficientNet—using full-precision FP32 with PyTorch. The Jupyter Notebook files for these models can be found [here]((Final_Report/Training_and_Quantization).
The details of their accuracy are presented in Table 1.

### Post-Training Quantization 

### ONNX conversion 

## Hardware Sides
### 1. NVDLA Simulator:

<p align="center">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/nvdla_flow.png" alt="nvdla overview" title="nvdla overview" width="500">
</p>

<p align="center">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/nvdla_vp.png" alt="nvdla virtual" title="nvdla virtual" width="500">
</p>

<p align="center">
  <img src="https://github.com/hplp/ai-hardware-project-6501-group10/blob/main/Final%20Report/Images/nvdla_results.png" alt="nvdla results" title="nvdla results" width="500">
</p>

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

-LeNet

-AlexNet

-EfficientNet

## Challenges and Takeaways

### Challenges
- ONNX to Caffe Conversion: Lack of ONNX support in the NVDLA required converting to Caffe, which remains problematic.
- Outdated Simulators: the NVDLA's outdated emulators forced us to use alternatives like Scale-Sim.
- Complex Setup: Docker and QEMU setups for GreenSocs QBox required extensive troubleshooting.
- Quantization Limitations: Scale-Sim lacked support for quantization.

### Key Takeaways

- Layer-wise Quantization: Allowed better control over scaling factors, maintaining accuracy.
- Platform Utilization: The NVDLA gave detailed execution times; Scale-Sim helped assess utilization and mapping efficiency.
- Data Preprocessing: Normalization and augmentation improved model robustness, especially for EfficientNet on CIFAR-10.


## 8. Future Improvements
Potential enhancements include:

[Improvement 1 with reason]
[Improvement 2 with reason]



