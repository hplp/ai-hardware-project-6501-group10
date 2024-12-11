### Scale-Sim:

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

