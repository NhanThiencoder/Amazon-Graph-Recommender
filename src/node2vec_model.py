from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import random

import networkx as nx


@dataclass(frozen=True)
class Node2VecConfig:
	dimensions: int = 128
	walk_length: int = 40
	num_walks: int = 10
	window_size: int = 5
	negative_samples: int = 5
	p: float = 1.0
	q: float = 1.0
	lr: float = 0.01
	epochs: int = 3
	batch_size: int = 256
	seed: int = 42
	device: str = "cpu"


def _alias_sample(rng: random.Random, accept: List[float], alias: List[int]) -> int:
	"""Sample from an alias table."""
	i = rng.randrange(len(accept))
	if rng.random() < accept[i]:
		return i
	return alias[i]


def _build_alias_table(probs: Sequence[float]) -> Tuple[List[float], List[int]]:
	"""Build alias table for O(1) discrete sampling."""

	n = len(probs)
	if n == 0:
		return [], []

	scaled = [p * n for p in probs]
	accept = [0.0] * n
	alias = [0] * n
	small: List[int] = []
	large: List[int] = []

	for i, sp in enumerate(scaled):
		(small if sp < 1.0 else large).append(i)

	while small and large:
		s = small.pop()
		l = large.pop()
		accept[s] = scaled[s]
		alias[s] = l
		scaled[l] = scaled[l] - (1.0 - scaled[s])
		(small if scaled[l] < 1.0 else large).append(l)

	for i in large + small:
		accept[i] = 1.0
		alias[i] = i

	return accept, alias


class _BiasedRandomWalker:
	"""Node2Vec biased random walks on a NetworkX graph."""

	def __init__(self, G: nx.Graph, *, p: float, q: float, seed: int = 42):
		if G.is_directed():
			raise ValueError("This Node2Vec baseline expects an undirected graph")
		self.G = G
		self.p = float(p)
		self.q = float(q)
		self.rng = random.Random(seed)

		# Precompute neighbors for speed
		self.neighbors: Dict[Any, List[Any]] = {n: list(G.neighbors(n)) for n in G.nodes()}

		# Alias tables for transitions conditioned on previous node
		self.alias_nodes: Dict[Any, Tuple[List[float], List[int]]] = {}
		self.alias_edges: Dict[Tuple[Any, Any], Tuple[List[float], List[int], List[Any]]] = {}

		self._preprocess_transition_probs()

	def _preprocess_transition_probs(self) -> None:
		# First step (no previous node): uniform over neighbors
		for node, nbrs in self.neighbors.items():
			if not nbrs:
				self.alias_nodes[node] = ([], [])
				continue
			probs = [1.0 / len(nbrs)] * len(nbrs)
			self.alias_nodes[node] = _build_alias_table(probs)

		# Second and later steps: biased transition based on return/in-out parameters
		for src in self.G.nodes():
			for dst in self.neighbors.get(src, []):
				dst_nbrs = self.neighbors.get(dst, [])
				if not dst_nbrs:
					self.alias_edges[(src, dst)] = ([], [], [])
					continue
				unnorm: List[float] = []
				for x in dst_nbrs:
					if x == src:
						unnorm.append(1.0 / self.p)
					elif self.G.has_edge(x, src):
						unnorm.append(1.0)
					else:
						unnorm.append(1.0 / self.q)
				s = sum(unnorm) or 1.0
				probs = [u / s for u in unnorm]
				accept, alias = _build_alias_table(probs)
				self.alias_edges[(src, dst)] = (accept, alias, dst_nbrs)

	def walk(self, start_node: Any, walk_length: int) -> List[Any]:
		walk = [start_node]
		while len(walk) < walk_length:
			cur = walk[-1]
			nbrs = self.neighbors.get(cur, [])
			if not nbrs:
				break
			if len(walk) == 1:
				accept, alias = self.alias_nodes[cur]
				if not accept:
					break
				idx = _alias_sample(self.rng, accept, alias)
				walk.append(nbrs[idx])
			else:
				prev = walk[-2]
				accept, alias, dst_nbrs = self.alias_edges.get((prev, cur), ([], [], []))
				if not accept:
					break
				idx = _alias_sample(self.rng, accept, alias)
				walk.append(dst_nbrs[idx])
		return walk

	def generate_walks(self, nodes: Sequence[Any], *, num_walks: int, walk_length: int) -> List[List[Any]]:
		walks: List[List[Any]] = []
		nodes = list(nodes)
		for _ in range(num_walks):
			self.rng.shuffle(nodes)
			for n in nodes:
				walks.append(self.walk(n, walk_length))
		return walks


def _build_skipgram_pairs(walks: Iterable[List[Any]], window_size: int) -> List[Tuple[Any, Any]]:
	pairs: List[Tuple[Any, Any]] = []
	w = int(window_size)
	for walk in walks:
		for i, center in enumerate(walk):
			left = max(0, i - w)
			right = min(len(walk), i + w + 1)
			for j in range(left, right):
				if j == i:
					continue
				pairs.append((center, walk[j]))
	return pairs


def fit_node2vec_torch(
	G: nx.Graph,
	*,
	config: Node2VecConfig = Node2VecConfig(),
	nodes: Optional[Sequence[Any]] = None,
) -> Dict[Any, List[float]]:
	"""Train Node2Vec embeddings (PyTorch) and return {node: vector}.

	NOTE: This is a baseline implementation meant for experiments and fair comparison
	against your traditional graph heuristics. It is not optimized for very large graphs.
	"""

	import math

	import torch
	from torch import nn
	from torch.nn import functional as F

	if nodes is None:
		nodes = list(G.nodes())
	else:
		nodes = list(nodes)

	rng = random.Random(config.seed)

	# Index nodes
	node_to_idx: Dict[Any, int] = {n: i for i, n in enumerate(nodes)}
	idx_to_node: List[Any] = nodes
	num_nodes = len(nodes)
	if num_nodes < 2:
		raise ValueError("Need at least 2 nodes")

	# Random walks
	walker = _BiasedRandomWalker(G, p=config.p, q=config.q, seed=config.seed)
	walks = walker.generate_walks(nodes, num_walks=config.num_walks, walk_length=config.walk_length)

	# Skip-gram pairs
	pairs = _build_skipgram_pairs(walks, window_size=config.window_size)
	if not pairs:
		raise RuntimeError("No skip-gram pairs generated; graph may be too sparse")

	# Convert to index pairs
	pair_idx = [(node_to_idx[u], node_to_idx[v]) for (u, v) in pairs if u in node_to_idx and v in node_to_idx]
	rng.shuffle(pair_idx)

	device = torch.device(config.device)
	emb_in = nn.Embedding(num_nodes, config.dimensions, sparse=False).to(device)
	emb_out = nn.Embedding(num_nodes, config.dimensions, sparse=False).to(device)

	# Init
	nn.init.xavier_uniform_(emb_in.weight)
	nn.init.xavier_uniform_(emb_out.weight)

	optimizer = torch.optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=config.lr)

	# Unigram distribution for negative sampling (approx): degree^0.75
	degrees = [float(G.degree(idx_to_node[i])) for i in range(num_nodes)]
	weights = [math.pow(d if d > 0 else 1.0, 0.75) for d in degrees]
	weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
	weights_t = weights_t / weights_t.sum()

	def neg_sample(batch_size: int, k: int) -> torch.Tensor:
		# shape: (batch_size, k)
		return torch.multinomial(weights_t, num_samples=batch_size * k, replacement=True).view(batch_size, k)

	batch_size = int(config.batch_size)
	k = int(config.negative_samples)
	steps = (len(pair_idx) + batch_size - 1) // batch_size

	for _epoch in range(int(config.epochs)):
		rng.shuffle(pair_idx)
		for step in range(steps):
			batch = pair_idx[step * batch_size : (step + 1) * batch_size]
			if not batch:
				continue
			centers = torch.tensor([c for c, _ in batch], dtype=torch.long, device=device)
			contexts = torch.tensor([t for _, t in batch], dtype=torch.long, device=device)
			negs = neg_sample(len(batch), k)

			v = emb_in(centers)  # (B, D)
			u_pos = emb_out(contexts)  # (B, D)
			u_neg = emb_out(negs)  # (B, K, D)

			# Positive loss: -log sigma(v·u)
			pos_score = (v * u_pos).sum(dim=1)
			pos_loss = F.logsigmoid(pos_score).mean()

			# Negative loss: sum log sigma(-v·u_neg)
			neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)  # (B, K)
			neg_loss = F.logsigmoid(-neg_score).mean()

			loss = -(pos_loss + neg_loss)

			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()

	# Export embeddings (use input embeddings)
	vectors = emb_in.weight.detach().cpu().numpy()
	return {idx_to_node[i]: vectors[i].astype(float).tolist() for i in range(num_nodes)}

