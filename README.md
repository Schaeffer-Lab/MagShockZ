# MagShockZ

Welcome to the MagShockZ project!

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This repository primarily contains the code used to convert FLASH simulation to OSIRIS simulation, though it is generally for analysis of simulations related to the Magnetized Collisionless Shocks on Z experiment (MagShockZ).

## Installation

To "install" MagShockZ, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Schaeffer-Lab/MagShockZ.git
    ```
2. Navigate to the project directory:
    ```bash
    cd MagShockZ
    ```

3 (optional). Install the [pyVisOS](https://github.com/UCLA-Plasma-Simulation-Group/pyVisOS.git) package for visualization:
    ```bash
    pip install git+https://github.com/UCLA-Plasma-Simulation-Group/pyVisOS.git@dev
    ```

Visualization of OSIRIS data utilizes the pyVisOS package, so it is highly recommended to install it.

## Usage

For detailed usage instructions, refer to the [User Guide](docs/user_guide.md).

## Analysis Scripts

Analysis scripts: scripts that are generally used for the analysis of OSIRIS simulation. 

[Check Initialization](analysis_scripts/check_initialization1d.ipynb): A quick diagnostic tool to verify simulation stability during initialization. Can be customized for specific validation needs.

[Calculate nGPUs](analysis_scripts/calculate_nGPUs.ipynb) can generally be used to estimate the computational cost of your simulation based on memory and compute requirements.

[Find shock front](analysis_scripts/find_shock_front.py) WIP. In the future it will be used to locate where the shock discontinuity is in a given simulation.



## Contributing

Please contact me through GitHub (ID: dschneidinger) if you have any questions about the project, or if you would like to use these tools to convert other FLASH simulation data to OSIRIS data.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.