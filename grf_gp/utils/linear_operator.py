import torch
from linear_operator.operators import LinearOperator


class SparseLinearOperator(LinearOperator):
    """
    A LinearOperator that wraps a sparse CSR tensor and performs
    sparse matrix @ dense tensor operations efficiently.
    """

    def __init__(self, sparse_csr_tensor):
        if not sparse_csr_tensor.is_sparse_csr:
            raise ValueError("Input tensor must be a sparse CSR tensor")
        self.sparse_csr_tensor = sparse_csr_tensor
        super().__init__(sparse_csr_tensor)

    def _matmul(self, rhs):
        return self.sparse_csr_tensor.matmul(rhs)

    def _size(self):
        return self.sparse_csr_tensor.size()

    def _transpose_nonbatch(self):
        """Tranpose the linear operator by converting:
        CSR tensor -> CSC tensor -> CSR tensor."""
        return SparseLinearOperator(self.sparse_csr_tensor.t().to_sparse_csr())
