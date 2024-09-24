# Distributed-Deep-Learning-Training
By following this manual, users should be able to effectively utilize the provided scalable distributed training script for large-scale deep learning tasks on single or multiple nodes. This script has been designed to lower hardware requirements while maintaining performance in deep learning training. 


# Distributed Deep Learning Training Manual

## Table of Contents

1. Overview
2. Key Concepts
3. Requirements and Environment Setup
4. Script Structure
5. Command-Line Interface (CLI)
6. Features
7. Running the Script
8. Distributed Training Setup
9. Best Practices
10. Troubleshooting
11. Disclaimers

---

## Overview

This Python script is provided "as is" to the open-source community to contribute to human advancement and collaborative research in deep learning. It offers a scalable solution for distributed training using PyTorch, featuring Distributed Data Parallel (DDP), mixed precision training, gradient accumulation, and elastic scaling. The script is designed to make efficient use of hardware and resources, enabling training across multiple GPUs or machines.

Please note, **this code is provided without warranties**, and it is your responsibility to ensure its suitability for your specific use case, particularly in commercial or sensitive environments. Consult your legal advisor if you plan to use this code for commercial purposes.

---

## Key Concepts

1. **Distributed Data Parallel (DDP)**: A PyTorch feature that distributes the training process across multiple GPUs or machines. It splits data across GPUs and synchronizes gradients to ensure optimal model training across distributed environments.

2. **Mixed Precision Training**: A technique that uses both FP16 (half-precision floating point) and FP32 (single-precision) during training. This reduces memory usage and speeds up computations.

3. **Gradient Accumulation**: A method that simulates larger batch sizes by accumulating gradients over several smaller mini-batches. This is helpful when GPU memory is limited.

4. **Synchronized Batch Normalization**: This ensures that batch statistics are synchronized across all GPUs in multi-GPU setups, particularly useful when batch sizes per GPU are small.

5. **Elastic Scaling**: The script can handle dynamic scaling of GPUs or nodes during runtime, making it suitable for environments where the number of resources can vary.

---

## Requirements and Environment Setup

### Hardware Requirements

- One or more GPUs.
- For multi-node setups, ensure each node has access to at least one GPU and that they are connected via a high-speed network (e.g., Infiniband or 10Gb Ethernet).

### Software Requirements

- Python 3.7 or later.
- PyTorch 1.8 or later (with `torch.distributed` and `torch.cuda.amp` for mixed precision training).
- NCCL (Collective Communications Library) for efficient communication between GPUs.

### Installation

Install the required Python packages by running:

```bash
pip install torch torchvision numpy
```

### Optional: NCCL Setup for Multi-GPU Training

Ensure that NCCL is correctly configured for multi-GPU environments. NCCL is included in most PyTorch installations and optimizes communication between GPUs.

---

## Script Structure

1. **Distributed Initialization**: The script initializes distributed training using `torch.distributed` and sets up communication using the NCCL backend.

2. **Data Loading and Augmentation**: The script uses `DataLoader` with a `DistributedSampler` to ensure the dataset is evenly distributed across GPUs, improving efficiency during training.

3. **Model Setup**: A pretrained ResNet18 model is used, with its final layer modified to classify images from the CIFAR-10 dataset. All layers except the last one are frozen to optimize for transfer learning.

4. **Training Loop**: The script includes a training loop that handles mixed precision training, gradient accumulation, learning rate scheduling, and model checkpointing.

---

## Command-Line Interface (CLI)

The script offers a range of command-line options to allow for customization. Below are some common CLI arguments:

- `--epochs`: Number of epochs for training (default: 50).
- `--batch_size`: Batch size for training (default: 32).
- `--learning_rate`: Initial learning rate (default: 0.001).
- `--accumulation_steps`: Number of gradient accumulation steps (default: 4).
- `--checkpoint`: File path for saving model checkpoints (default: 'model_checkpoint.pth').
- `--world_size`: Total number of distributed processes (default: 1).
- `--rank`: Rank of the current process in multi-node setups (default: 0).
- `--gpu_ids`: Comma-separated list of GPU IDs to use (default: '0').
- `--dist_url`: URL for initializing distributed training (default: 'tcp://127.0.0.1:23456').

### Example CLI Commands

1. Single-GPU Training:
   ```bash
   python train.py --epochs 50 --batch_size 64 --learning_rate 0.001
   ```

2. Multi-GPU Training on a Single Node:
   ```bash
   python -m torch.distributed.launch --nproc_per_node=2 train.py --epochs 50 --batch_size 64 --gpu_ids 0,1 --world_size 2
   ```

3. Multi-Node Training:
   On Node 1:
   ```bash
   python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=23456 train.py --epochs 50 --gpu_ids 0,1 --world_size 4
   ```

   On Node 2:
   ```bash
   python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr="192.168.1.1" --master_port=23456 train.py --epochs 50 --gpu_ids 0,1 --world_size 4
   ```

---

## Features

1. **Distributed Data Parallel (DDP)**: Efficiently splits the workload across multiple GPUs and synchronizes gradients to optimize training time.
   
2. **Mixed Precision Training**: Uses FP16 computations to reduce memory usage while speeding up model training.

3. **Gradient Accumulation**: Allows training with larger effective batch sizes by accumulating gradients over several smaller batches.

4. **Synchronized Batch Normalization**: Ensures consistent training results by synchronizing batch statistics across GPUs in distributed setups.

5. **Checkpointing**: Periodically saves the model’s state and optimizer’s state so that training can be resumed from where it left off in case of interruptions.

6. **Learning Rate Warmup**: Gradually increases the learning rate during the initial training epochs to stabilize the model’s performance early on.

7. **Elastic Scaling**: Supports dynamic adjustment of the number of GPUs or nodes during runtime without needing to restart the job.

---

## Running the Script

### Single-GPU Training

For single-GPU training, run the script with the default settings:

```bash
python train.py --epochs 50 --batch_size 64
```

### Multi-GPU Training on a Single Node

For training on multiple GPUs on the same machine, use PyTorch’s `torch.distributed.launch` utility:

```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py --epochs 50 --batch_size 64 --gpu_ids 0,1
```

### Multi-Node Training

To train across multiple nodes, you must have SSH access between nodes. Set the correct `--nnodes`, `--node_rank`, and `--master_addr` parameters:

```bash
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=23456 train.py --epochs 50 --gpu_ids 0,1 --world_size 4
```

---

## Distributed Training Setup

1. **SSH Setup for Multi-Node Training**: Ensure passwordless SSH access between nodes by setting up SSH keys.

2. **NCCL Configuration**: Configure the NCCL backend for efficient GPU communication. For debugging, use:
   ```bash
   export NCCL_DEBUG=INFO
   ```

3. **Launching Distributed Training**: Use PyTorch’s `torch.distributed.launch` utility to initialize distributed processes.

---

## Best Practices

1. **Network Setup for Multi-Node Training**: Use high-speed networking, such as InfiniBand or 10Gb Ethernet, for communication between nodes in multi-node training.

2. **Monitor GPU Utilization**: Use tools like `nvidia-smi` (or your system’s equivalent) to monitor GPU utilization and ensure each GPU is being used efficiently.

3. **Learning Rate Warmup**: For large-batch distributed training, warm up the learning rate during the first few epochs to avoid instability.

---

## Troubleshooting

1. **NCCL Configuration**: If there are communication issues between GPUs, ensure that the NCCL backend is set up properly.

2. **Low GPU Utilization**: If GPUs are underutilized, check that the batch size is large enough to keep them busy.

3. **Debugging Distributed Training**: For debugging distributed errors, set `NCCL_DEBUG=INFO` and monitor the logs for communication issues.

---

## Disclaimers

This code and accompanying manual are provided **as is** for open-source use. While the code is intended to facilitate scalable distributed deep learning, **there are no warranties or guarantees** regarding the accuracy, completeness, or reliability of the code and documentation.

By using this code, you agree that you are responsible for ensuring its suitability for your intended use case, particularly in commercial or high-risk environments. The authors of this script assume no liability for any issues arising from its use.

This script is intended for research and educational purposes. If you plan to use it in a commercial setting, please consult your legal advisor to ensure that you comply with any relevant regulations or licensing requirements.

---

## Conclusion

By following this manual, users should be able to effectively utilize the provided scalable distributed training script for large-scale deep learning tasks on single or multiple nodes. This script has been designed to lower hardware requirements while maintaining performance in deep learning training. However, please ensure that you are fully aware of the legal implications of using this script, especially in commercial contexts.
