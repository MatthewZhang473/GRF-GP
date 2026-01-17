"""GRF random-walk sampler built around torch sparse CSR tensors."""

import os
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from .utils.linear_operator import SparseLinearOperator


def _to_sparse_csr(
    adjacency: Union[torch.Tensor, "torch.sparse.Tensor"],
) -> torch.Tensor:
    """
    Coerce input adjacency to a torch.sparse_csr_tensor.
    """
    if isinstance(adjacency, torch.Tensor) and adjacency.is_sparse_csr:
        return adjacency
    if isinstance(adjacency, torch.Tensor) and adjacency.is_sparse:
        return adjacency.to_sparse_csr()
    if isinstance(adjacency, torch.Tensor):
        return adjacency.to_sparse_csr()
    raise TypeError("adjacency must be a torch Tensor (dense or sparse)")


def _build_csr_from_entries(num_nodes: int, entries: defaultdict) -> torch.Tensor:
    if not entries:
        crow = torch.zeros(num_nodes + 1, dtype=torch.int64)
        col = torch.zeros(0, dtype=torch.int64)
        vals = torch.zeros(0, dtype=torch.float32)
        return torch.sparse_csr_tensor(crow, col, vals, (num_nodes, num_nodes))

    keys = list(entries.keys())
    rows = torch.tensor([k[0] for k in keys], dtype=torch.int64)
    cols = torch.tensor([k[1] for k in keys], dtype=torch.int64)
    vals = torch.tensor([entries[k] for k in keys], dtype=torch.float32)
    # Torch expects crow_indices to be monotonic; use coo -> csr for simplicity
    coo = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, size=(num_nodes, num_nodes)
    ).coalesce()
    return coo.to_sparse_csr()


def _worker_walks(
    args: tuple,
) -> List[defaultdict]:
    """
    Worker function for multiprocessing random walks.
    """
    (
        nodes,
        walks_per_node,
        p_halt,
        max_walk_length,
        seed,
        show_progress,
    ) = args

    rng = np.random.default_rng(seed)
    step_accumulators: List[defaultdict] = [
        defaultdict(float) for _ in range(max_walk_length)
    ]

    iterator = tqdm(nodes, desc="Process walks", disable=not show_progress)
    for start_node in iterator:
        for _ in range(walks_per_node):
            current_node = start_node
            load = 1.0
            for step in range(max_walk_length):
                step_accumulators[step][(start_node, current_node)] += load

                start = _G_CROW[current_node]
                end = _G_CROW[current_node + 1]
                degree = end - start
                if degree == 0:
                    break

                if rng.random() < p_halt:
                    break

                offset = rng.integers(degree)
                weight = _G_DATA[start + offset]
                current_node = _G_COL[start + offset]
                load *= degree * weight / (1 - p_halt)

    return step_accumulators


def _run_walks_local(
    crow: np.ndarray,
    col: np.ndarray,
    data: np.ndarray,
    nodes: np.ndarray,
    walks_per_node: int,
    p_halt: float,
    max_walk_length: int,
    seed: int,
) -> List[defaultdict]:
    """
    Single-process walk runner that avoids global state. Used for n_proc == 1.
    """
    rng = np.random.default_rng(seed)
    step_accumulators: List[defaultdict] = [
        defaultdict(float) for _ in range(max_walk_length)
    ]

    iterator = tqdm(nodes, desc="Process walks", disable=True)
    for start_node in iterator:
        for _ in range(walks_per_node):
            current_node = int(start_node)
            load = 1.0
            for step in range(max_walk_length):
                step_accumulators[step][(int(start_node), current_node)] += load

                start = crow[current_node]
                end = crow[current_node + 1]
                degree = end - start
                if degree == 0:
                    break
                if rng.random() < p_halt:
                    break
                offset = rng.integers(degree)
                weight = data[start + offset]
                current_node = int(col[start + offset])
                load *= degree * weight / (1 - p_halt)

    return step_accumulators


# Globals for worker fast access
_G_CROW: Optional[np.ndarray] = None
_G_COL: Optional[np.ndarray] = None
_G_DATA: Optional[np.ndarray] = None


def _init_worker(crow: np.ndarray, col: np.ndarray, data: np.ndarray) -> None:
    """Initializer for worker processes (bind CSR arrays)."""
    global _G_CROW, _G_COL, _G_DATA
    _G_CROW = crow
    _G_COL = col
    _G_DATA = data


class GRFSampler:
    """
    Generates GRF random walk matrices
    and returns them as SparseLinearOperator objects.
    """

    def __init__(
        self,
        adjacency_matrix: Union[torch.Tensor, "torch.sparse.Tensor"],
        walks_per_node: int = 10,
        p_halt: float = 0.5,
        max_walk_length: int = 10,
        seed: Optional[int] = None,
        use_tqdm: bool = True,
        n_processes: Optional[int] = None,
    ) -> None:
        self.adjacency_csr = _to_sparse_csr(adjacency_matrix).cpu()
        if self.adjacency_csr.size(0) != self.adjacency_csr.size(1):
            raise ValueError("Adjacency matrix must be square.")

        self.walks_per_node = walks_per_node
        self.p_halt = p_halt
        self.max_walk_length = max_walk_length
        self.use_tqdm = use_tqdm
        self.n_processes = n_processes
        self.seed = seed or 42

    def sample_random_walk_matrices(self) -> List[SparseLinearOperator]:
        """
        Perform GRF random walks and return per-step random walk matrices.
        """
        crow_indices = self.adjacency_csr.crow_indices().numpy()
        col_indices = self.adjacency_csr.col_indices().numpy()
        values = self.adjacency_csr.values().numpy()
        num_nodes = self.adjacency_csr.size(0)

        n_proc = self.n_processes or os.cpu_count() or 1
        chunks = np.array_split(np.arange(num_nodes), n_proc)

        ctx = mp.get_context("fork")

        # Fast path: single process still uses
        # the same merging logic to avoid duplication
        if n_proc == 1:
            results = [
                _run_walks_local(
                    crow_indices,
                    col_indices,
                    values,
                    chunks[0],
                    self.walks_per_node,
                    self.p_halt,
                    self.max_walk_length,
                    self.seed,
                )
            ]
        else:
            with ProcessPoolExecutor(
                max_workers=n_proc,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(crow_indices, col_indices, values),
            ) as executor:
                args = [
                    (
                        chunk.tolist(),
                        self.walks_per_node,
                        self.p_halt,
                        self.max_walk_length,
                        self.seed + i,
                        self.use_tqdm and i == 0,
                    )
                    for i, chunk in enumerate(chunks)
                ]
                futures = [executor.submit(_worker_walks, a) for a in args]
                results = [fut.result() for fut in as_completed(futures)]

        accumulators = [defaultdict(float) for _ in range(self.max_walk_length)]
        for result in results:
            for step in range(self.max_walk_length):
                for key, val in result[step].items():
                    accumulators[step][key] += val

        matrices = [
            SparseLinearOperator(
                _build_csr_from_entries(num_nodes, acc) * (1.0 / self.walks_per_node)
            )
            for acc in accumulators
        ]
        return matrices

    def __call__(self) -> List[SparseLinearOperator]:
        return self.sample_random_walk_matrices()


if __name__ == "__main__":
    rows = [0, 0, 0, 1, 1, 2, 2, 3]
    cols = [1, 2, 3, 0, 2, 0, 3, 0]
    data = [1, 1, 1, 1, 1, 1, 1, 1]
    adjacency = torch.sparse_csr_tensor(
        torch.tensor([0, 3, 5, 7, 8]),  # crow_indices
        torch.tensor(cols),
        torch.tensor(data, dtype=torch.float32),
        size=(4, 4),
    )

    sampler = GRFSampler(
        adjacency_matrix=adjacency,
        walks_per_node=1000,
        p_halt=0.1,
        max_walk_length=3,
        seed=42,
        use_tqdm=True,
        n_processes=2,
    )
    random_walk_mats = sampler.sample_random_walk_matrices()

    adjacency_dense = adjacency.to_dense()
    print("Adjacency (dense):")
    print(adjacency_dense)
    for t in range(sampler.max_walk_length):
        rw_dense = random_walk_mats[t].sparse_csr_tensor.to_dense()
        adj_power = torch.linalg.matrix_power(adjacency_dense, t)
        print(f"t={t} random walk matrix (dense):")
        print(rw_dense)
        print(f"t={t} adjacency^{t} (dense):")
        print(adj_power)
