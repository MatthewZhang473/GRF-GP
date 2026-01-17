import torch

from grf_gp.utils.linear_operator import SparseLinearOperator


def test_sparse_linear_operator_matmul_and_transpose():
    indices = torch.tensor([[0, 0, 1, 2], [0, 2, 1, 0]])
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    size = (3, 3)

    sparse_csr = torch.sparse_coo_tensor(indices, values, size).to_sparse_csr()
    op = SparseLinearOperator(sparse_csr)

    rhs = torch.tensor([[1.0], [2.0], [3.0]])
    result = op @ rhs
    expected = sparse_csr.to_dense() @ rhs
    assert torch.allclose(result, expected)

    lhs = torch.tensor([[1.0, 2.0, 3.0]])
    result_t = lhs @ op
    expected_t = lhs @ sparse_csr.to_dense()
    assert torch.allclose(result_t, expected_t)
