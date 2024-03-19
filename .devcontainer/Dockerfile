FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
# Update packages
RUN apt update
RUN apt install -y git

# For signed commits
# https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials#_sharing-gpg-keys
RUN apt install gnupg2 -y

# Install make for makefile
RUN apt install -y make

# Install git large file storage
RUN apt install -y git-lfs
RUN git lfs install

# Install c compiler for torch.compile
RUN apt-get update
RUN apt install -y build-essential

# Install torch
RUN pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
# pip install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu # CPU only

# Install torch_geometric and dependencies
RUN pip install torch_geometric==2.4.0
RUN pip install pyg_lib==0.3.1+pt21cu121 torch_scatter==2.1.2+pt21cu121 torch_sparse==0.6.18+pt21cu121 torch_cluster==1.6.3+pt21cu121 torch_spline_conv==1.2.2+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
# RUN pip install pyg_lib==0.3.1+cpu torch_scatter==2.1.2+cpu torch_sparse==0.6.18+cpu torch_cluster==1.6.3+cpu torch_spline_conv==1.2.2+cpu -f https://data.pyg.org/whl/torch-2.1.0+cpu.html # CPU only
