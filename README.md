# Shaffer's Rho Calculation

This project provides a Python implementation of Shaffer's Rho calculation for interrater reliability statistics. It includes the capability to calculate rho given a kappa value, a code set, or a 2x2 contingency table. The implementation leverages `numpy` for numerical operations and `pybind11` for integrating C++ functions.

## Table of Contents

- [Introduction](#introduction)
  - [Key Terms](#key-terms)
  - [How Shaffer's Rho Works](#how-shaffers-rho-works)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
- [Development](#development)
  - [Setting Up](#setting-up)
  - [Building the C++ Extension](#building-the-c-extension)
  - [Running Tests](#running-tests)
- [Reporting Issues](#reporting-issues)
- [Contributing](#contributing)
- [License](#license)

## Introduction

### Key Terms

- **Kappa**: Cohen's kappa coefficient is a statistic that measures inter-rater agreement for qualitative (categorical) items. It is generally thought to be a more robust measure than simple percent agreement calculation, as kappa takes into account the agreement occurring by chance.
  
- **Recall**: Also known as sensitivity, recall is the fraction of relevant instances that have been retrieved over the total amount of relevant instances. It is calculated as:
```math
  \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
```
- **Precision**: Precision is the fraction of relevant instances among the retrieved instances. It is calculated as:
```math
  \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
```
- **Baserate**: The baserate in this context refers to the proportion of positive instances in the dataset. It is used in various calculations to adjust for the underlying prevalence of the condition or category being measured.

### How Shaffer's Rho Works

Shaffer's Rho is a Monte Carlo method used to assess the reliability of inter-rater agreement. It operates by constructing a collection of datasets in which the kappa value is below a specified threshold and computing the empirical distribution of kappa based on the specified sampling procedure. The rho statistic quantifies the Type I error in generalizing from an observed test set to a true value of agreement between two raters.

The process involves:
1. **Starting with an observed kappa value**: This kappa is calculated from a subset of a `codeSet`, known as an observed `testSet`.
2. **Generating simulated code sets**: These simulated code sets have kappa values below the threshold and similar properties to the original `codeSet`.
3. **Sampling test sets**: From each simulated code set, a test set is sampled to create a null hypothesis distribution.
4. **Calculating kappa for each test set**: This creates a distribution of kappa values under the null hypothesis.
5. **Comparing the observed kappa**: The observed kappa is compared to this distribution to determine if it is significantly higher than what would be expected by chance.

If the observed kappa is greater than a specified percentage (e.g., 95%) of the null hypothesis distribution, the null hypothesis is rejected, indicating that the raters have acceptable agreement.

## Getting Started

### Installation

To use the Shaffer's Rho calculation in your project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/shaffers-rho.git
    cd shaffers-rho
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Build the C++ extension:
    ```sh
    python setup.py build_ext --inplace
    ```

### Usage

Here is an example of how to use the `ShaffersRho` class:

```python
import numpy as np
from shaffers_rho import ShaffersRho

# Given an observed kappa value
shaffer_rho = ShaffersRho(x=0.88, observed_code_baserate=0.2, test_set_length=80)
print(shaffer_rho.result)

# Given a test set
codeset = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0],
    [1, 1],
    [0, 1]
])
shaffer_rho = ShaffersRho(x=codeset)
print(shaffer_rho.result)

# Given a contingency table
contingency_table = np.array([
    [3, 2],
    [1, 4]
])
shaffer_rho = ShaffersRho(x=contingency_table)
print(shaffer_rho.result)
```

## Development

### Setting Up

To start developing the project yourself, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/shaffers-rho.git
    cd shaffers-rho
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Build the C++ extension:
    ```sh
    python setup.py build_ext --inplace
    ```

### Building the C++ Extension

The project includes C++ code that is integrated using `pybind11`. To build the C++ extension:

1. Ensure `pybind11` and `setuptools` are installed:
    ```sh
    pip install pybind11 setuptools
    ```

2. Build the extension:
    ```sh
    python setup.py build_ext --inplace
    ```

### Running Tests

To ensure everything is working correctly, run the tests:

1. Install `pytest` if you haven't already:
    ```sh
    pip install pytest
    ```

2. Run the tests:
    ```sh
    pytest
    ```

## Reporting Issues

If you encounter any issues or bugs, please report them by opening an issue on the [GitHub Issues](https://github.com/yourusername/shaffers-rho/issues) page.

## Contributing

We welcome contributions! Please fork the repository and submit pull requests. Before contributing, please read our [Contributing Guidelines](CONTRIBUTING.md) (link to your contributing guidelines if you have one).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
