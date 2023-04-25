# SBINN: Systems-biology-informed neural network

The source code for the book chapter [M. Daneker, Z. Zhang, G. E. Karniadakis, & L. Lu. Systems biology: Identifiability analysis and parameter identification via systems-biology-informed neural networks. *Computational Modeling of Signaling Networks*, Springer, 87--105, 2023](https://link.springer.com/protocol/10.1007/978-1-0716-3008-2_4).

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
  title     = {Systems Biology: Identifiability Analysis and Parameter Identification via Systems-Biology-Informed Neural Networks},
  author    = {Daneker, Mitchell and Zhang, Zhen and Karniadakis, George Em and Lu, Lu},
  editor    = {Nguyen, Lan K.},
  bookTitle = {Computational Modeling of Signaling Networks},
  year      = {2023},
  publisher = {Springer US},
  pages     = {87--105},
  doi       = {10.1007/978-1-0716-3008-2_4}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
