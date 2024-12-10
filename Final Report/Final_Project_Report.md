# **Performance Evaluation of CNN on Open-Source Platforms**
## Team Name

## Team Members
- Melika Morsali Toshmnaloui
-
-

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
In this project, we trained three models—LeNet, AlexNet, and EfficientNet—using full-precision FP32 with PyTorch. The Jupyter Notebook files for these models can be found [here](path/to/your/file.extension)
The details of their accuracy are presented in Table 1.

### Post-Training Quantization 

### ONNX conversion 

## Hardware Sides
### 1. NVDLA Simulator:
Set up?
challenge?
### 2. Scale-Sim:
-LeNet??
-AlexNet??
-EfficientNet??
## Challenges and Takeaways

## 8. Future Improvements
Potential enhancements include:

[Improvement 1 with reason]
[Improvement 2 with reason]



