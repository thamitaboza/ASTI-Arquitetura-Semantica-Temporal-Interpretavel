"""Árvore AVL por chave numérica com busca exaustiva por similaridade."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _validate_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (1024,) or not np.all(np.isfinite(vector)):
        raise ValueError("O vetor deve conter 1024 valores finitos.")
    if not np.any(vector):
        raise ValueError("A similaridade de cosseno não aceita vetores nulos.")
    return vector.copy()


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    # Escalar primeiro evita overflow na norma de valores muito grandes.
    scaled = vector / np.max(np.abs(vector))
    return scaled / np.linalg.norm(scaled)


@dataclass
class SemanticNode:
    """Dados semânticos e campos estruturais de um nó AVL."""

    key: float
    cluster_data: dict
    embedding: np.ndarray
    left: Optional["SemanticNode"] = None
    right: Optional["SemanticNode"] = None
    height: int = 1

    def __post_init__(self) -> None:
        self.key = float(self.key)
        if not np.isfinite(self.key):
            raise ValueError("A chave deve ser um número finito.")
        if not isinstance(self.cluster_data, dict):
            raise TypeError("cluster_data deve ser um dicionário.")
        self.embedding = _validate_vector(self.embedding)


class SemanticAVL:
    """Inserção O(log n); busca semântica O(n * 1024).

    Chaves repetidas substituem os dados do nó existente. A ordenação por
    chave não permite podar a busca por similaridade: todos os nós são lidos.
    """

    def __init__(self) -> None:
        self.root: Optional[SemanticNode] = None

    @staticmethod
    def _height(node: Optional[SemanticNode]) -> int:
        return node.height if node else 0

    def _update_height(self, node: SemanticNode) -> None:
        node.height = 1 + max(self._height(node.left), self._height(node.right))

    def _balance(self, node: SemanticNode) -> int:
        return self._height(node.left) - self._height(node.right)

    def _rotate_right(self, node: SemanticNode) -> SemanticNode:
        pivot = node.left
        assert pivot is not None
        node.left = pivot.right
        pivot.right = node
        self._update_height(node)
        self._update_height(pivot)
        return pivot

    def _rotate_left(self, node: SemanticNode) -> SemanticNode:
        pivot = node.right
        assert pivot is not None
        node.right = pivot.left
        pivot.left = node
        self._update_height(node)
        self._update_height(pivot)
        return pivot

    def insert(self, key: float, cluster_data: dict, embedding: np.ndarray) -> None:
        """Insere ou atualiza um cluster, mantendo a árvore balanceada."""
        self.root = self._insert(self.root, SemanticNode(key, cluster_data, embedding))

    def _insert(self, node: Optional[SemanticNode], new: SemanticNode) -> SemanticNode:
        if node is None:
            return new
        if new.key < node.key:
            node.left = self._insert(node.left, new)
        elif new.key > node.key:
            node.right = self._insert(node.right, new)
        else:
            node.cluster_data = new.cluster_data
            node.embedding = new.embedding
            return node

        self._update_height(node)
        balance = self._balance(node)
        if balance > 1:
            assert node.left is not None
            if self._balance(node.left) < 0:  # LR
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)  # LL
        if balance < -1:
            assert node.right is not None
            if self._balance(node.right) > 0:  # RL
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)  # RR
        return node

    def semantic_search(self, query_vector: np.ndarray) -> Optional[dict]:
        """Retorna os metadados mais similares ou None para uma árvore vazia.

        Valida a consulta mesmo em árvore vazia. Em empates, preserva o
        primeiro nó encontrado no percurso raiz, esquerda, direita.
        """
        query = _unit_vector(_validate_vector(query_vector))
        best_score = -float("inf")
        best_data = None

        def visit(node: Optional[SemanticNode]) -> None:
            nonlocal best_score, best_data
            if node is None:
                return
            score = float(np.dot(query, _unit_vector(node.embedding)))
            if score > best_score:
                best_score, best_data = score, node.cluster_data
            visit(node.left)
            visit(node.right)

        visit(self.root)
        return best_data
