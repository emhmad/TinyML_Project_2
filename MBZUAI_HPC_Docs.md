# MBZUAI HPC Cluster Documentation

> **Official documentation for MBZUAI HPC clusters: CIAI, CSCC, and CAMD.**
> All users must be connected to the university network (wired, wireless, or VPN) to access cluster resources.

---

## Table of Contents

1. [Cluster Policies & Rules](#1-cluster-policies--rules)
2. [Connection Guide](#2-connection-guide)
3. [SSH Key Generation](#3-ssh-key-generation)
4. [Basic SLURM Commands](#4-basic-slurm-commands)
5. [Job Submission](#5-job-submission)
6. [Running Experiments](#6-running-experiments)
7. [Software Management](#7-software-management)
8. [Examples](#8-examples)
9. [Jupyter Notebooks](#9-jupyter-notebooks)
10. [Porting Code to AMD ROCm](#10-porting-code-to-amd-rocm)
11. [Hardware Specs — CIAI](#11-hardware-specs--ciai)
12. [Hardware Specs — CSCC](#12-hardware-specs--cscc)
13. [Hardware Specs — CAMD](#13-hardware-specs--camd)
14. [Frequently Asked Questions](#14-frequently-asked-questions)

---

## 1. Cluster Policies & Rules

> ⚠️ **Violating any of these restrictions may result in temporary account suspension. Repeated violations can lead to permanent account termination.**

> ⚠️ **Notice:** The Lustre storage partition on CIAI/CSCC is currently unavailable.

---

### Access Requirements for CIAI and CSCC

To gain access to the **CIAI** or **CSCC** clusters, users must:

1. Complete the HPC usage quiz at: <https://forms.gle/ke5BiCRe9nYcQ7Fs8>
2. Achieve a **passing score of 15/17**
3. Email proof of the passing score to: <hpc.admins@mbzuai.ac.ae>

> Access will only be granted after verification by the HPC administration team.

---

### Common Rules

- Do **not** run your job or code directly on the login node
- Do **not** run `sleep` jobs via `sbatch` (jobs which request allocation but do not use compute resources)
- Use the `gpu-debug-qos` QOS for all interactive and debugging jobs
- Do **not** use the `salloc` command to run jobs; instead, use `srun` or `sbatch`
- Do **not** leave idle bash sessions open when not in use
- Do **not** use the environment variable `CUDA_VISIBLE_DEVICES` to use more GPUs than requested — this is strictly not allowed and leads to account termination
- Do **not** use the `--exclusive` option with `srun` or `sbatch`
- If your dataset contains a large number of small files on the `/l/` filesystem, contact the HPC team for guidance
- The cluster is designed for training tasks. Users are responsible for backing up their own data. Once training is complete, remove unnecessary files
- Upon employee departure, HPC administrators will delete all associated user data. Ensure important data is backed up before leaving

---

### Special Compute Requests

Requests that exceed the standard allocation (e.g., exclusive node reservations, custom node images, or significant storage increases) **must** be submitted to <hpc.resources@mbzuai.ac.ae>.

> Requests sent through other channels will not be considered.

---

### Access Control

#### CIAI Cluster

- Accessible only to MBZUAI regular faculty (PIs). Each PI is granted **one active CIAI account**
- Resource access can be delegated by the PI to students or postdocs using the **SSH key method**. The PI remains fully accountable for any misuse
- General and large-scale compute tasks are permitted through the **SLURM queue system**. **Small compute tasks are not allowed** in this queue

---

## 2. Connection Guide

> **Important:** You must be connected to the university network through wired, wireless, or VPN before connecting.

---

### Windows

The following tools are recommended for Windows users:

1. **[Visual Studio Code](https://code.visualstudio.com/)** — Connect, run terminal commands, edit files, manage Git repos
   - Install the **Remote SSH** plugin from the VSCode plugin marketplace
   - Navigate to the SSH connections menu → Add new connection
   - Enter your SSH command, select config location, then connect
   - Select **Linux** as the remote OS and enter your university email password

2. **[MobaXterm](https://mobaxterm.mobatek.net/)** — Full-featured SSH terminal with built-in file transfer

3. **[PuTTY](https://www.putty.org/)** — Lightweight SSH client for terminal access (no file editing)

4. **[FileZilla](https://filezilla-project.org/)** — File transfer between your machine and the cluster

5. **[WinSCP](https://winscp.net/eng/download.php)** — File transfer and basic file manager

---

### macOS

- **[CyberDuck](https://cyberduck.io/)** — File transfer client

---

### Linux

Connect directly from the terminal:

```bash
ssh user.name@ciai.mbzuai.ac.ae
# Password: your university email password
```

> The instructions above apply to **CSCC** and **CIAI**. For CAMD, see the dedicated subsection below.

---

### CAMD Cluster Access

CAMD uses a different access model from CSCC/CIAI — it is managed by **Core42 AI Cloud Support** and requires SSH key–based authentication.

#### Obtaining Access

1. Email **Core42 AI Cloud Support** at <support.aicloud@core42.ai> and include your SSH **public** key
2. Secure prior approval from the cluster owner
3. Core42 will send a **User Request Form** that must be digitally approved by the cluster owner before access is granted

#### Logging In

> Access is based on SSH key authentication — you will **not** be prompted for a password if you connect with the correct key.

**Windows (PowerShell)**

Windows 10/11 has an SSH client built in. Open PowerShell (Win + X → Windows PowerShell / Terminal) and run:

```bash
ssh username@linux_machine_ip

# Example
ssh core42@172.27.112.247
```

**Linux / macOS**

```bash
ssh -i <path-to-your-key> username@linux_machine_ip

# Example
ssh -i ~/.ssh/id_rsa core42@172.27.112.247
```

> See the CAMD login node list in [Section 13 — Hardware Specs — CAMD](#13-hardware-specs--camd) for the available login hostnames and IPs.

---

## 3. SSH Key Generation

Access to **CIAI** may require authentication using SSH keys. Follow the steps below to generate a public/private key pair.

> **Do not send the HPC team your public key unless explicitly instructed.**

---

### macOS and Linux

```bash
# Generate key pair
ssh-keygen -t ed25519 -C "user.name@mbzuai.ac.ae"

# Press Enter to accept default location (~/.ssh/id_ed25519)
# Optionally set a passphrase

# View your public key
cat ~/.ssh/id_ed25519.pub
```

Your keys will be at:
- **Private key**: `~/.ssh/id_ed25519` — **NEVER share this**
- **Public key**: `~/.ssh/id_ed25519.pub` — This is what you share

#### ⚠️ How to Identify Your Keys

| File | Starts With | Action |
|------|-------------|--------|
| `id_ed25519.pub` (Public) | `ssh-ed25519` | ✅ Safe to share |
| `id_ed25519` (Private) | `-----BEGIN OPENSSH PRIVATE KEY-----` | ❌ NEVER share |

#### If You Accidentally Share Your Private Key

```bash
# Delete both keys immediately
rm ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub

# Generate a new pair
ssh-keygen -t ed25519 -C "user.name@mbzuai.ac.ae"
```

---

### Windows (PowerShell)

```powershell
ssh-keygen -t ed25519 -C "user.name@mbzuai.ac.ae"

# View your public key
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```

---

### Windows (PuTTYgen)

1. Open PuTTYgen and click **Generate**
2. Move your mouse around the blank area to generate randomness
3. Click **Save private key** to store your private key
4. Copy the text from **"Public key for pasting into OpenSSH authorized_keys file"**
5. Send the copied public key to the HPC team

---

## 4. Basic SLURM Commands

### Job Submission

| Command | Purpose | Example |
|---------|---------|---------|
| `sbatch script.sh` | Submit a batch job | `sbatch myjob.sh` |
| `srun command` | Run a command interactively or in parallel | `srun hostname` |
| `salloc` | Allocate resources and get an interactive shell | `salloc -N 2 -n 4` |

### Job Monitoring

| Command | Purpose | Example |
|---------|---------|---------|
| `squeue` | List all jobs in the queue | `squeue` |
| `squeue -u username` | Show only your jobs | `squeue -u myname` |
| `squeue --me` | Show only your jobs (shorthand) | `squeue --me` |
| `sacct` | Show job accounting info after completion | `sacct -j 12345` |
| `watch -n1 squeue` | Continuously refresh job queue | — |

### Job Control

| Command | Purpose | Example |
|---------|---------|---------|
| `scancel job_id` | Cancel a specific job | `scancel 12345` |
| `scancel -u username` | Cancel all your jobs | `scancel -u myname` |

### Resource Request Options (sbatch / srun)

| Option | Meaning | Example |
|--------|---------|---------|
| `-N` | Number of nodes | `-N 2` |
| `-n` | Total tasks (MPI ranks) | `-n 8` |
| `--cpus-per-task` | CPU cores per task | `--cpus-per-task=4` |
| `--gres=gpu:N` | Request N GPUs | `--gres=gpu:4` |
| `-t` | Time limit (HH:MM:SS) | `-t 01:30:00` |
| `-p` | Partition/queue name | `-p cscc-gpu-p` |
| `-J` | Job name | `-J myjob` |
| `--mem` | Total RAM | `--mem=40G` |

---

## 5. Job Submission

### Quick Submission Examples

```bash
# Submit a batch job
sbatch -N 1 -n 4 -t 00:30:00 job.sh

# Interactive shell on a GPU node
salloc -N 1 --gres=gpu:1 -t 02:00:00

# Run an MPI program on 4 nodes
srun -N 4 -n 64 ./my_mpi_app
```

---

### Example Batch Script (Multi-GPU)

```bash
#!/bin/bash
#SBATCH -J my_gpu_job              # Job name
#SBATCH -N 8                       # Number of nodes
#SBATCH --ntasks-per-node=8        # 1 task per GPU
#SBATCH --gres=gpu:8               # GPUs per node
#SBATCH -p gpu                     # Partition/queue name
#SBATCH -t 02:00:00                # Time limit hh:mm:ss
#SBATCH -o job_%j.out              # Standard output
#SBATCH -e job_%j.err              # Standard error
#SBATCH --cpus-per-task=8          # CPU cores per GPU task

# Load necessary modules
module load rocm
module load mpi

# Debug info
echo "Running on nodes:"
srun hostname | sort -u
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Total GPUs: $(( $SLURM_NNODES * 8 ))"

# Run your application
srun ./my_gpu_program
```

---

### Request Nodes / GPUs (Helper Tool)

```bash
# Submit jobs using the helper script (splits across single-node jobs, max 8 GPUs each)
cd /vast/users/guangyi.chen/slurm_tools
./submit_job.sh <NAME> <TOTAL_GPUS>

# Example: submits 2 jobs (one with 8 GPUs, one with 4 GPUs)
./submit_job.sh myproj 12
```

### Attach to a Running Job

```bash
# First find your job ID
squeue -u $USER

# Then attach
cd /vast/users/guangyi.chen/slurm_tools
./attach_job.sh <JOBID>
```

---

## 6. Running Experiments

### SLURM Partitions

> **Note:** Email notification flags (`--mail-type`, `--mail-user`) are **not available** on this cluster.

#### CSCC Partitions

##### `cscc-gpu-p` — GPU Partition

| Limit | Value |
|-------|-------|
| Max GPU node jobs | 2 |
| Max GPU cards per account | 4 |
| Max job duration | 24 hours |
| Minimum GPUs | 1 (`--gres=gpu:1`) |
| Required QOS flag | `-q cscc-gpu-qos` |

##### `cscc-cpu-p` — CPU Partition

| Limit | Value |
|-------|-------|
| Max CPU cores | 512 |
| Max running jobs | 2 |
| Max job duration | 72 hours |
| Required QOS flag | `-q cscc-cpu-qos` |

> If no partition is specified, the job is automatically assigned to the CPU partition.

---

#### CIAI Partitions

##### `long` — GPU Partition

Use when the job requires GPU processing.

| Limit | Value |
|-------|-------|
| Max GPUs | 12 (or 8 jobs) |
| Max job duration | 72 hours |
| Minimum GPUs | 1 (`--gres=gpu:1`) |
| Required QOS flag | `-q gpu-12` |

##### `cscc-cpu-p` — CPU Partition (shared with CSCC)

Use when only CPU processing is required.

| Limit | Value |
|-------|-------|
| Max CPU cores | 512 |
| Max running jobs | 2 |
| Max job duration | 72 hours |
| Required QOS flag | `-q cscc-cpu-qos` |

> If no partition is specified, the job is automatically assigned to the CPU partition.

---

### SLURM QOS

> `sacctmgr` is the Slurm accounting manager and is often restricted to cluster
> administrators. If it is not in your `PATH`, that is expected on many login
> nodes. For regular users, `scontrol show qos <qos-name>` is the usual
> read-only alternative when available.

#### `cscc-gpu-qos` — Production / Batch Jobs

- Use for regular batch jobs
- Limits: up to 24h runtime, up to 4 GPUs per user, up to 2 GPU-node jobs per account

```bash
sbatch -p cscc-gpu-p -q cscc-gpu-qos my_job.sh
```

#### `gpu-debug-qos` — Debug / Interactive Jobs

- Use for short tests and interactive sessions
- Limits: up to 3h runtime, up to 4 GPUs per user

```bash
# Interactive session
srun -p cscc-gpu-p -q gpu-debug-qos --gres=gpu:1 --cpus-per-task=8 -t 3:00:00 --pty bash

# Short batch test
sbatch -p cscc-gpu-p -q gpu-debug-qos test.sh
```

---

### Running Jobs with `srun`

```bash
# Single GPU job (1 GPU, 8 CPU cores)
srun --ntasks=1 --cpus-per-task=8 -p cscc-gpu-p -q gpu-debug-qos \
     --gres=gpu:1 -t 3:00:00 --output=./slurm-%N-%j.out \
     python my_python_script.py

# CPU job (128 cores)
srun --ntasks=1 --cpus-per-task=128 -p cscc-cpu-p -q cscc-cpu-qos \
     --output=./slurm-%N-%j.out python my_python_script.py

# CPU job (2 cores)
srun --ntasks=1 --cpus-per-task=2 -p cscc-cpu-p -q cscc-cpu-qos \
     --output=./slurm-%N-%j.out python my_python_script.py
```

---

### CIAI GPU Job Submission

#### Submitting a Basic 1-Node Job

Save the following as `$HOME/job.sh`:

```bash
#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64

# Replace with your actual job or command
nvidia-smi
```

Submit with:

```bash
sbatch job.sh
# Submitted batch job 35189
```

This schedules the job with a 2-hour wall-time on 1 node (4 × 40GB A100 GPUs per node). Replace `nvidia-smi` with your own command (e.g. `python train.py`). After completion, stdout/stderr lands in the submit directory.

Parameter notes:

- `--cpus-per-task=64` — CPU cores per task per node. If the job is not CPU-heavy, request fewer (10 is enough for most tasks).
- `--gres=gpu:4` — 4 physical GPUs per node.
- `--mem=230G` — 230 GB of system memory per node (this is **not** GPU memory).

#### Submitting a Multi-Node Job

```bash
#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=2
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=128

# Replace with your actual command
srun hostname
srun nvidia-smi
```

`--nodes=2` requests 2 nodes (2 × 4 = 8 A100 GPUs total). `srun` spawns processes across both nodes — `srun hostname` prints 8 lines (4 per node) and `srun nvidia-smi` runs on every rank.

From here you can plug in any distributed training framework (PyTorch DDP, Megatron-LM, etc.).

#### Running a 3-Node, 12-GPU PyTorch DDP Job

**Step 1** — Save the following as `elastic_ddp.py`:

```python
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

import socket

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    device_id = rank % torch.cuda.device_count()
    print('I am rank {} using GPU {} on host {}'.format(rank, device_id, socket.gethostname()))
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_id)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    dist.destroy_process_group()

if __name__ == "__main__":
    demo_basic()
```

**Step 2** — Save the following as `torchrun_script.sh` and make it executable (`chmod +x torchrun_script.sh`):

```bash
#!/bin/bash

torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=4 \
         --rdzv_id=1009 --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_ADDR:29401 elastic_ddp.py
```

**Step 3** — Save the SLURM launcher as `ddp.sh`. PyTorch is assumed to be installed in a Conda env activated by the `export PATH=...` line:

```bash
#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=3
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --gres=gpu:3
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

export PATH=${PWD}/.conda/bin:$PATH

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

srun ./torchrun_script.sh
```

**Step 4** — Submit and inspect:

```bash
sbatch ddp.sh
# Submitted batch job 38867

grep GPU slurm-38867.out | sort
# I am rank 0 using GPU 0 on host gpu-46
# I am rank 1 using GPU 1 on host gpu-46
# ...
# I am rank 11 using GPU 3 on host gpu-48
```

All 12 GPUs across 3 nodes are now in use. From here, swap `elastic_ddp.py` for your real training script and tune `ddp.sh` (walltime, etc.) as needed.

---

### Monitoring Jobs

```bash
# View all your jobs
squeue --me

# Continuously refresh queue (every 1 second)
watch -n1 squeue

# Check node availability
sinfo
sinfo -s   # summarized view

# Check estimated start time
squeue --start
```

### Cancelling Jobs

```bash
# Cancel a specific job
scancel $JOB_ID

# Cancel all your jobs
scancel -u $USER
```

---

## 7. Software Management

### Anaconda Setup

Anaconda is **not activated by default**. To use it:

```bash
# Activate for current session
source /apps/local/conda_init.sh

# Activate automatically on every login
echo "source /apps/local/conda_init.sh" >> ~/.bashrc
```

### Mamba / Micromamba

If you only want a faster solver, recent Conda releases already support the
`libmamba` backend:

```bash
conda config --set solver libmamba
```

If you want a standalone single-binary installer, use `micromamba` rather than
`mamba`. `mamba` is installed inside a Conda prefix, while `micromamba` is the
self-contained executable.

> Keep the `base` environment minimal. Avoid mixing `defaults` and `conda-forge`
> in the same installation unless you know the package set is compatible.

### Creating Virtual Environments

```bash
# Create a new environment
conda create -p /path/to/env

# Activate the environment
conda activate /path/to/env

# Auto-activate on login
echo "conda activate /path/to/env" >> ~/.bashrc
```

---

### Installing Frameworks

#### PyTorch

After activating Anaconda, follow the official PyTorch installation guide for your CUDA/ROCm version.

#### TensorFlow 2

Follow the official TensorFlow installation steps within an Anaconda environment.

#### TensorFlow 1.15

```bash
# Create environment with Python 3.8
conda create -p /path/to/env python=3.8
conda activate /path/to/env

# Install TensorFlow 1.15 (NVIDIA build)
pip install --upgrade pip
pip install nvidia-pyindex
pip install nvidia-tensorflow[horovod]
pip install nvidia-tensorboard==1.15
```

Verify the installation:

```python
import tensorflow as tf
import tensorboard

tf.enable_eager_execution()
a = tf.random.uniform([1000, 1000])
b = tf.random.uniform([1000, 1000])
tf.matmul(a, b)
```

---

### Common Issues & Solutions

#### CUDA Version Mismatch

```bash
# Check current CUDA version
nvcc --version

# Reinstall PyTorch with the correct CUDA version from pytorch.org
```

#### Memory Issues During Installation

```bash
# Use no-cache option
pip install --no-cache-dir package_name

# Or install packages one at a time
```

#### Package Conflicts

```bash
# Create a clean environment
conda create -n clean_env python=3.8 -y
conda activate clean_env

# Install main frameworks first, then dependencies
```

---

## 8. Examples

### Full Example: Running a GPU Job

**Step 1** — Create your Python script (`test.py`):

```python
import subprocess
subprocess.run(["nvidia-smi"])
```

**Step 2** — Activate your conda environment:

```bash
conda activate your_environment
```

**Step 3** — Create the SLURM batch script:

```bash
cat > slurm_script << EOL
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=output.%A_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH --time=12:00:00
#SBATCH --qos=cscc-gpu-qos

hostname
python test.py
EOL
```

**Step 4** — Submit the job:

```bash
sbatch slurm_script
```

**Step 5** — Check the output log:

```
gpu-02
Thu Sep 26 10:56:01 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.14   Driver Version: 550.54.14   CUDA Version: 12.4                  |
| GPU  Name               | Memory-Usage        | GPU-Util |
|   0  NVIDIA A100-SXM4-40GB  | 4MiB / 40960MiB | 0%      |
|   1  NVIDIA A100-SXM4-40GB  | 4MiB / 40960MiB | 0%      |
|   2  NVIDIA A100-SXM4-40GB  | 4MiB / 40960MiB | 0%      |
|   3  NVIDIA A100-SXM4-40GB  | 4MiB / 40960MiB | 0%      |
+-----------------------------------------------------------------------------------------+
```

---

## 9. Jupyter Notebooks

### CSCC Jupyter Setup via VS Code

**Prerequisites:** Install the **Jupyter Notebook** and **Remote SSH** extensions in VS Code, then SSH into the login node.

**Step 1** — Create a SLURM script (`jupyter.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=jupyter.out

ENV_NAME="YOUR_CONDA_ENV_NAME"
PORT=$(shuf -i 20000-65000 -n 1)

srun --export=ALL bash -lc "
  . /apps/local/anaconda2023/conda_init.sh
  conda activate ${ENV_NAME} || { echo 'env not found'; exit 1; }
  python -m pip install -q --upgrade jupyterlab ipykernel
  python -m ipykernel install --user --name ${ENV_NAME} \
    --display-name 'Python (${ENV_NAME})' >/dev/null 2>&1 || true
  jupyter lab --no-browser --ip=0.0.0.0 --port=${PORT}
"
```

**Step 2** — Submit the job:

```bash
sbatch jupyter.sh
```

**Step 3** — Check the output file:

```bash
cat jupyter.out
```

**Step 4** — Copy the **second-to-last URL** from the output.

**Step 5** — In VS Code, open your notebook → **Kernels** → **"Existing Jupyter Server"** → paste the URL.

**Step 6** — Verify you're on the compute node:

```bash
!hostnamectl
```

---

## 10. Porting Code to AMD ROCm

> The CAMD cluster uses **AMD MI210 GPUs** with **ROCm 6.3**.

| Topic | Notes |
|-------|-------|
| **PyTorch device APIs** | `tensor.cuda()` and `to(device="cuda")` work as-is — ROCm/HIP translates automatically |
| **Distributed training** | `backend="nccl"` is automatically mapped to **RCCL** on ROCm |
| **GPU monitoring** | Use `rocm-smi` instead of `nvidia-smi`; use `nvtop` instead of `nvitop` |
| **FlashAttention** | No prebuilt wheels — build from source or use PyTorch's `scaled_dot_product_attention` |

---

## 11. Hardware Specs — CIAI

### Login Nodes

| Spec | Value |
|------|-------|
| Nodes | 2 (`CIAI-login-1`, `CIAI-login-2`) |
| CPU | Dual AMD EPYC 7402 (24 cores each) |
| Total Cores | 48 |
| RAM | 256 GB |

### GPU Nodes

| Spec | Value |
|------|-------|
| Total Nodes | 39 |
| CPU | 2 × AMD EPYC 7742 (128 cores, 256 threads total) |
| RAM | 256 GB per node |
| GPUs | 4 × NVIDIA A100 SXM 40 GB per node |
| Storage | High-speed NVMe scratch |

### CPU Nodes

| Spec | Value |
|------|-------|
| Total Nodes | 5 |
| CPU | 2 × AMD EPYC 7742 (128 cores, 256 threads total) |
| RAM | 256 GB per node |

---

### Storage Locations & Quotas

#### 1. Lustre Storage — `/l/users/$USER`
- For large files: datasets, checkpoints
- Default quota: **2 TB**

```bash
lfs quota -u $USER /l
```

#### 2. Home Directory — `$HOME`
- For code, logs, small files
- Default quota: **100 GB**

```bash
du -sh $HOME
```

#### 3. Scratch Space — `/tmp`
- Fast local storage on compute nodes
- **Not shared**, not persistent

```bash
du -sh /tmp/$USER
```

---

## 12. Hardware Specs — CSCC

### Login Nodes

Shared with CIAI cluster (Dual AMD EPYC 7402, 48 cores total, 256 GB RAM)

### GPU Nodes

| Spec | Value |
|------|-------|
| Total Nodes | 16 |
| CPU | 2 × AMD EPYC 7742 (128 cores, 256 threads total) |
| RAM | 256 GB per node |
| GPUs | 4 × NVIDIA A100 SXM 40 GB per node |
| Storage | High-speed NVMe scratch |

### CPU Nodes

| Spec | Value |
|------|-------|
| Total Nodes | 12 |
| CPU | 2 × AMD EPYC 7742 (128 cores, 256 threads total) |
| RAM | 256 GB per node |

---

### Storage Locations & Quotas

#### 1. Lustre Storage — `/l/users/$USER`
- Default quota: **2 TB**

```bash
lfs quota -u $USER /l
```

#### 2. Home Directory — `$HOME`
- Default quota: **100 GB**

```bash
du -sh $HOME
```

#### 3. Scratch Space — `/tmp`
- **2–4 TB** fast local storage per compute node
- **Not shared**, **not persistent** — deleted after use

```bash
du -sh /tmp/$USER
```

---

## 13. Hardware Specs — CAMD

### Cluster Overview

| Spec | Value |
|------|-------|
| Total GPUs | 1,000 AMD MI210 |
| Worker Nodes | ~125 (8 GPUs per node) |
| Workload Manager | SLURM |
| Shared Storage | VAST (NFS) |

### Node Types

| Type | Description |
|------|-------------|
| **Slurm Controller Nodes** | Managed by Core42 HPC Ops — handles SLURM config, policies, QoS |
| **Login Nodes** | Landing zone for users — prepare environments and submit jobs |
| **Worker/GPU Nodes** | AMD GPU nodes — training, inference, model hosting |
| **Shared Storage** | VAST NFS — home directories accessible across the cluster |

### Login Nodes

| Hostname | IP Address | Role |
|----------|------------|------|
| `auh-1b-cpu-login-001` | 172.27.112.247 | Login Node |
| `auh-1b-cpu-login-002` | 172.27.112.248 | Login Node |

> **Important:** Nodes cannot access the internet by default. To whitelist a domain, email Core42 AI Cloud Support with the required domain(s) and prior approval from the cluster owner.

---

### Checking Storage (VAST)

```bash
# Storage path
/vast/users/username

# Check quota on demand
bash /etc/update-motd.d/99-vast-quota
```

Example quota output:

```
=== VAST User Storage Usage ===
User: username (uid xxxx)

Filesystem /vast:
  Used:       0.000 GB (0.0%)
  Available:  2048.910 GB
  Soft quota: 1536.682 GB
  Hard quota: 2048.910 GB
================================
```

---

### GPU Performance Benchmarks (bf16 Precision)

#### CNN

| GPU | # GPUs | Step Time (s) | Throughput |
|-----|--------|---------------|------------|
| MI210 | 1 | 0.165 | 1,552 images/s |
| A100-SXM4-40G | 1 | 0.061 | 4,148 images/s |
| RTX 6000 Ada | 1 | 0.097 | 2,633 images/s |
| MI210 | 4 | 0.167 | 6,105 images/s |
| A100-SXM4-40G | 4 | 0.071 | 14,399 images/s |
| RTX 6000 Ada | 4 | 0.115 | 8,903 images/s |

#### GPT

| GPU | # GPUs | Step Time (s) | Throughput |
|-----|--------|---------------|------------|
| MI210 | 1 | 0.249 | 2,050 tokens/s |
| A100-SXM4-40G | 1 | 0.202 | 2,523 tokens/s |
| MI210 | 4 | 0.299 | 6,842 tokens/s |
| A100-SXM4-40G | 4 | 0.259 | 7,905 tokens/s |

> **Measurement Tips:** Fix batch size and sequence/resolution. Report median over multiple steps after warm-up.

---

## 14. Frequently Asked Questions

---

### Q: I get `ssh: connect to host cscc.mbzuai.ac.ae port 22: Operation timed out`. What should I do?

**A:** The `cscc` hostname may not be reachable from your network. Try using `ciai` instead:

```bash
ssh username@ciai.mbzuai.ac.ae
```

If the issue persists, check your internet connection or contact HPC support.

---

### Q: When I try to copy files using `scp`, I get `subsystem request failed on channel 0` / `scp: Connection closed`. How do I fix it?

**A:** Add the `-O` flag to force the legacy SCP protocol:

```bash
scp -O myfile.py username@ciai.mbzuai.ac.ae:/users/username
```

---

*Last updated: April 2026 | Contact: <hpc.admins@mbzuai.ac.ae> | Special requests: <hpc.resources@mbzuai.ac.ae>*
