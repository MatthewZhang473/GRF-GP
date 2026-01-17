# Graph Random Features for Scalable Gaussian Processes (GRF-GP)



GRF-GP is the package for performing fast Gaussian Process (GP) inference on graphs. Internally, it uses **Graph Random Features** (GRFs) to compute a **unbiased** & **sparse** estimate of a family of well-known graph node kernels. It further uses **path-wise conditioning** to leverage the sparsity of the kernel approximation, enabling you to perform GP model train / inference in $\mathcal{O}(N^{3/2})$ time and $\mathcal{O}(N)$ space complexity.



## Examples


- Graph GP inference

--- Maybe move the below examples to another repo ---
- Regression tasks
- Classification tasks
- Bayesian Optimization


## Installation


## Citing Us

If you use GRF-GP, please cite the following papers:

    @article{zhang2025graph,
    title={Graph random features for scalable Gaussian processes},
    author={Zhang, Matthew and Lin, Jihao Andreas and Choromanski, Krzysztof and Weller, Adrian and Turner, Richard E and Reid, Isaac},
    journal={arXiv preprint arXiv:2509.03691},
    year={2025}
    }

## Contributing

