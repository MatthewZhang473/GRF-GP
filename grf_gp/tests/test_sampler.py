import torch

from grf_gp.sampler import GRFSampler


def _chain_adjacency(num_nodes: int) -> torch.Tensor:
    """
    Build a simple chain graph where each node connects to the next,
    and the last node has a self-loop. Degree is 1 everywhere, so walks are deterministic.
    """
    rows = torch.arange(num_nodes, dtype=torch.int64)
    cols = torch.cat([torch.arange(1, num_nodes), torch.tensor([num_nodes - 1])])
    data = torch.ones(num_nodes, dtype=torch.float32)
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]), data, size=(num_nodes, num_nodes)
    ).to_sparse_csr()


def test_grf_sampler_chain_walks_single_process():
    num_nodes = 4
    sampler = GRFSampler(
        adjacency_matrix=_chain_adjacency(num_nodes),
        walks_per_node=1,
        p_halt=0.0,
        max_walk_length=3,
        seed=123,
        use_tqdm=False,
        n_processes=1,
    )

    mats = sampler.sample_random_walk_matrices()
    assert len(mats) == 3

    dense = [m.sparse_csr_tensor.to_dense() for m in mats]

    expected_step0 = torch.eye(num_nodes)
    expected_step1 = torch.zeros(num_nodes, num_nodes)
    expected_step1[:-1, 1:] = torch.eye(num_nodes - 1)
    expected_step1[-1, -1] = 1.0

    expected_step2 = torch.zeros(num_nodes, num_nodes)
    expected_step2[:-2, 2:] = torch.eye(num_nodes - 2)
    expected_step2[-2:, -1:] = torch.tensor([[1.0], [1.0]])

    assert torch.allclose(dense[0], expected_step0)
    assert torch.allclose(dense[1], expected_step1)
    assert torch.allclose(dense[2], expected_step2)


def test_grf_sampler_halt_and_multiprocessing():
    # With p_halt=1, only the 0-step matrix should have counts (diagonal = walks_per_node)
    adjacency = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 0]]), torch.ones(2), size=(2, 2)
    ).to_sparse_csr()

    sampler = GRFSampler(
        adjacency_matrix=adjacency,
        walks_per_node=2,
        p_halt=1.0,
        max_walk_length=2,
        seed=7,
        use_tqdm=False,
        n_processes=2,
    )

    mats = sampler.sample_random_walk_matrices()
    dense = [m.sparse_csr_tensor.to_dense() for m in mats]

    assert torch.allclose(dense[0], torch.eye(2) * 2)
    assert torch.count_nonzero(dense[1]) == 0
