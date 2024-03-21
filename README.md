# Cell-Type Prediction in Spatial Transcriptomics Data using Graph Neural Networks

This is the official repository reproducing the experiments from the ICLR 2024 Workshop MLGenX Poster "Cell-Type Prediction in Spatial Transcriptomics Data using Graph Neural Networks".

The following gives an introduction in how to use the repository to reproduce the experiments. First, the structure of the repository is explained. Then, we describe how you can setup the python environment using DevContainers or Conda. Finally, we explain how to run the experiments and the unit tests.

## Structure

The repository is structured as follows:
- `.devcontainer/`: Contains the configuration for the DevContainer. This is used to set up a development environment that is consistent across different machines. It contains a Dockerfile and a `.json`-file that configures how to run the container. More information on how to use it can be found in the `Setup` section.
- `ctgnn/`: Contains the Python code that is used to run the experiments. The code is structured as a Python package with additional documentation in the `.py`-files in the form of docstrings.
- `data/`: Contains the spatial transcriptomics datasets that are used in the experiments. Each dataset is saved in a separate directory using `git-lfs` and contains the data in the form of `.parquet`-files. More information on the datasets can be found in the `Data` section and the `ReadMe.txt`-files in each dataset-directory.
- `run_configs/`: Contains the configuration files that are used to run the experiments. Each configuration file is a `.yaml`-file that contains the hyperparameters for the experiments. More information on how to use the configuration files can be found in the `Running the Experiments` section.
- `test/`: Contains the unit tests that are used to test the code in the `ctgnn/`-package. More information on how to run the tests can be found in the `Testing` section.
- `figures.ipynb`: A Jupyter notebook that contains the code to generate the figures that are used in the paper/poster.

## Setup

### DevContainers

#### Prerequisites

1. **VSCode**: If you do not have it installed, you can download it [here](https://code.visualstudio.com/).
2. **Docker**: You can find more information about how to install Docker [here](https://docs.docker.com/desktop/) by choosing the correct installation tutorial on the left side panel depending on your OS.
3. **Git**: If you do not have `git` installed already, you can follow [GitHub's guide](https://github.com/git-guides/install-git) on how to install it.

#### Set Up Steps

1. Use `git` to `clone` the repository and open it in VSCode. If you are unfamiliar with VSCode, you can open the repository via `File` -> `Open Folder...` and then choose the root directory which is called `cell-type-prediction-in-spatial-transcriptomics-data-using-gnns/` in this repo.
2. If you have the `Dev Containers Extension` installed already you can skip this step. Otherwise you might get a hint from VSCode on the bottom right because the repository contains a `.devcontainer` directory. If you do not get any hint, you can install it by searching for it in the VSCode Extension Marketplace. You can use the identifier `ms-vscode-remote.remote-containers` to find it faster. It is also available in the [web-marketplace](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
3. After successful installation, you should see an icon showing a greater and smaller sign on the far left side of the status bar at the bottom of VSCode. Click on it and choose the option `Reopen in Container`. This will start the Dev Container. The first time, this will take a few minutes to download the image but it will be faster afterwards.
4. After the container was built successfully, the VSCode window will reopen inside the container and the icon should now have a text next to it saying `Dev Container: Cell-Type GNN`.
5. If you get some error messages on the bottom right that are saying that it cannot connect to the python server or something similar, fix it by reponening VSCode.

More information on how to set up and use DevContainers is available in the official VSCode tutorial: https://code.visualstudio.com/docs/devcontainers/tutorial

### Conda

As a more lightweight alternative you can use [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) to create a virtual `Python` environment. The following commands create a new environment called `Cell-Type-GNN` and install all necessary packages. It is tested on Ubuntu 22.04. LTS. When running Windows, the Windows Subsystem for Linux (WSL2) is recommended.

```bash
conda create -y -n Cell-Type-GNN python=3.10
conda activate Cell-Type-GNN
conda install -y -c conda-forge cudnn=8.4.1.50 cudatoolkit=12.1.0
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.4.0
pip install pyg_lib==0.3.1+pt21cu121 torch_scatter==2.1.2+pt21cu121 torch_sparse==0.6.18+pt21cu121 torch_cluster==1.6.3+pt21cu121 torch_spline_conv==1.2.2+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install -e '.[dev,test]'
```

The above assumes that you have a GPU, and thus PyTorch needs to be installed with `CUDA` support. If you don't have a GPU available, use the following after creating your `conda` environment:
    
```bash
pip install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric==2.4.0
pip install pyg_lib==0.3.1+cpu torch_scatter==2.1.2+cpu torch_sparse==0.6.18+cpu torch_cluster==1.6.3+cpu torch_spline_conv==1.2.2+cpu -f https://data.pyg.org/whl/torch-2.1.0+cpu.html # CPU only

pip install -e '.[dev,test]'
```

## Running the Experiments

This repository is used to enable hyperparameter tuning for GNNs on spatial transcriptomics data. You can find the script that starts the tuning in `main.py`. You can run it using `python main.py path_to_config.yaml`. You can find the config files that were used for the experiments presented in the paper in `run_config/`.

## Data

### Datasets

There are currently a few spatially resolved transcriptomics datasets available in this repository. For more information on each dataset, you can have a look at the corresponding `README.txt`-file that is contained in each dataset folder. All of the folders are generally structured as follows:
- `README.txt`: Contains information about the dataset
- `cell_by_gene.parquet`: A table that contains different genes as columns and cells as rows. Each entry corresponds to the number of times each gene is expressed in the cell. The counts are sometimes normalized and can be floats or integers.
- `cell_coords.parquet`: A table with `x`, `y` and sometimes `z` coordinates for each cell in a tissue.
- `cluster_assignment.parquet`: A table that contains the cell type as name and as ID for each cell. The inference method varies for each dataset and more information can be found in the respective `README.txt` or the referenced paper.

All datasets are saved using the [Apache Parquet](https://parquet.apache.org/) file format. This format uses compression and needs only about 10% of the average storage that would be required to save the same file as `.csv`. It is also a lot faster in loading and saving compared to `.csv`. Using `pandas`, the data can be loaded and joined using the common index:
```python
cell_coords = pd.read_parquet("XX_dataset_dir/cell_coords.parquet")
cluster_assignment = pd.read_parquet("XX_dataset_dir/cluster_assignment.parquet")
cell_by_gene = pd.read_parquet("XX_dataset_dir/cell_by_gene.parquet")
data = cluster_assignment.join(cell_by_gene).join(cell_coords)
```

### Results

The results of the experiments presented in the paper/poster are saved in the `data/results/` directory. The results are saved as `.parquet`-files and contain the hyperparameter configuration for each run, as well as the performance metrics that were used to evaluate the models.

### git Large File Storage (git-lfs)

This repository uses `git-lfs` to store the large files used as data. If you do not have `git-lfs` installed, you can install it using `apt` or `brew` and running `git lfs install` after installation. This way all large files should be downloaded automatically. If the large files are not downloaded automatically (all files in the data directory), they are stored as pointers in the repository (and may cause an exception when trying to load the files with `pandas`). To download the actual files, you need to run `git lfs pull`.

## Testing

The code in this repository is tested with unit tests using `pytest`. You can run the automated tests by running `pytest` in the root directory of this repository to check if your setup works as expected. The tests are located in the `test/` directory. We aim for a test coverage of at least 95%.

## Disclaimer

During development, GitHub Copilot was actively used. This might have influenced the code in this repository. All code that was knowingly copied from other sources is referenced and attributed to the original authors. If you find any code that looks like it is copied from other sources and not referenced correctly, please kindly let me know.
