# Project Objectives
- Optimize LLM inference pipelines for GPU-based systems
- Examine improvements in latency, throughput, and resource utilization
- Experiment with techniques in GPU optimization and scalable deployment

# NVIDIA Nsight Compute 
#### Part 1:
1. Step 1:
    Load python exec path and python file path in the encircled regions and press launch.
   ![First Image](https://github.com/ramanahm1/LLM-Inferencing/blob/main/screenshots/cmd_1.png)

2. Step 2:
    Monitor the profile like you would on an IDE. Image shows different kernels invoked during execution.
   ![Second Image](screenshots\nsight_kernel_1.png)



3. Step 3:
    Monitor output report for the kernel under consideration.
    
    ![Third Image](screenshots\nsight_kernel_2.png)

4. Step 4:
    Check the Python GUI for the output of your code.

    ![Fourth Image](screenshots\cmd_2.png)
