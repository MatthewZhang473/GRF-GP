# Fast-Graph-GP



Fast-Graph-GP is the package for performing fast Gaussian Process (GP) inference on graphs. Internally, it uses **Graph Random Features** (GRFs) to compute a **unbiased** & **sparse** estimate of a family of well-known graph node kernels. It further uses **path-wise conditioning** to leverage the sparsity of the kernel approximation, enabling you to perform GP model train / inference in $\mathcal{O}(N^{3/2})$ time and $\mathcal{O}(N)$ space complexity.


## Examples

For a detailed example of training and using a Graph GP model, refer to the [example notebook](examples/basic_usage.ipynb).

## Installation

Install Fast-Graph-GP via pip:

```bash
pip install fast-graph-gp
```

## Citing Us

If you use Fast-Graph-GP, please cite the following papers:

    @article{zhang2025graph,
    title={Graph random features for scalable Gaussian processes},
    author={Zhang, Matthew and Lin, Jihao Andreas and Choromanski, Krzysztof and Weller, Adrian and Turner, Richard E and Reid, Isaac},
    journal={arXiv preprint arXiv:2509.03691},
    year={2025}
    }