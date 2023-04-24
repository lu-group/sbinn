# SBINN: Systems-biology informed neural network

The source code for the paper [Daneker, M., Zhang, Z., Karniadakis, G.E., Lu, L. (2023). Systems Biology: Identifiability Analysis and Parameter Identification via Systems-Biology-Informed Neural Networks. In: Nguyen, L.K. (eds) Computational Modeling of Signaling Networks. Methods in Molecular Biology, vol 2634. Humana, New York, NY. https://doi.org/10.1007/978-1-0716-3008-2_4](https://link.springer.com/protocol/10.1007/978-1-0716-3008-2_4).

## Code

- [Structural identifiability analysis](structural_identifiability.ipynb)
- SBINN
    - [Data generation](sbinn/data_generation.py)
    - [TensorFlow code](sbinn/sbinn_tf.py)
    - [PyTorch code](sbinn/sbinn_pytorch.py)
- [Practical identifiability analysis](practical_identifiability.jl)

## Cite this work

If you use this code for academic research, you are encouraged to cite the following paper:

```
@Inbook{Daneker2023,
author="Daneker, Mitchell and Zhang, Zhen and Karniadakis, George Em and Lu, Lu",
editor="Nguyen, Lan K.",
title="Systems Biology: Identifiability Analysis and Parameter Identification via Systems-Biology-Informed Neural Networks",
bookTitle="Computational Modeling of Signaling Networks",
year="2023",
publisher="Springer US",
address="New York, NY",
pages="87--105",
abstract="The dynamics of systems biological processes are usually modeled by a system of ordinary differential equations (ODEs) with many unknown parameters that need to be inferred from noisy and sparse measurements. Here, we introduce systems-biology-informed neural networks for parameter estimation by incorporating the system of ODEs into the neural networks. To complete the workflow of system identification, we also describe structural and practical identifiability analysis to analyze the identifiability of parameters. We use the ultradian endocrine model for glucose-insulin interaction as the example to demonstrate all these methods and their implementation.",
isbn="978-1-0716-3008-2",
doi="10.1007/978-1-0716-3008-2_4",
url="https://doi.org/10.1007/978-1-0716-3008-2_4"
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
