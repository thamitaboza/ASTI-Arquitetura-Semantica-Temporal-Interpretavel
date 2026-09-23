"""Árvore Rubro-Negra por chave numérica com busca exaustiva por similaridade."""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


Color = Literal["RED", "BLACK"]
RED: Color = "RED"
BLACK: Color = "BLACK"


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
class SemanticNodeRB:
    """Dados semânticos e campos estruturais de um nó Rubro-Negro."""

    key: float
    cluster_data: dict
    embedding: np.ndarray
    left: Optional["SemanticNodeRB"] = None
    right: Optional["SemanticNodeRB"] = None
    parent: Optional["SemanticNodeRB"] = None
    color: Color = RED

    def __post_init__(self) -> None:
        self.key = float(self.key)
        if not np.isfinite(self.key):
            raise ValueError("A chave deve ser um número finito.")
        if not isinstance(self.cluster_data, dict):
            raise TypeError("cluster_data deve ser um dicionário.")
        self.embedding = _validate_vector(self.embedding)


class SemanticRBTree:
    """Inserção O(log n); busca semântica O(n * 1024).

    Chaves repetidas substituem os dados do nó existente. Folhas None são
    pretas. Os contadores são cumulativos: cada rotação conta uma unidade,
    e cada mudança efetiva de cor no fixup (inclusive da raiz) conta uma
    recoloração. A cor inicial de um nó novo não conta como recoloração.
    """

    def __init__(self) -> None:
        self.root: Optional[SemanticNodeRB] = None
        self.rotations = 0
        self.recolorings = 0

    @staticmethod
    def _color(node: Optional[SemanticNodeRB]) -> Color:
        return node.color if node is not None else BLACK

    def _recolor(self, node: SemanticNodeRB, color: Color) -> None:
        if node.color != color:
            node.color = color
            self.recolorings += 1

    def _rotate_left(self, node: SemanticNodeRB) -> SemanticNodeRB:
        pivot = node.right
        assert pivot is not None
        node.right = pivot.left
        if pivot.left is not None:
            pivot.left.parent = node
        pivot.parent = node.parent
        if node.parent is None:
            self.root = pivot
        elif node is node.parent.left:
            node.parent.left = pivot
        else:
            node.parent.right = pivot
        pivot.left = node
        node.parent = pivot
        self.rotations += 1
        return pivot

    def _rotate_right(self, node: SemanticNodeRB) -> SemanticNodeRB:
        pivot = node.left
        assert pivot is not None
        node.left = pivot.right
        if pivot.right is not None:
            pivot.right.parent = node
        pivot.parent = node.parent
        if node.parent is None:
            self.root = pivot
        elif node is node.parent.right:
            node.parent.right = pivot
        else:
            node.parent.left = pivot
        pivot.right = node
        node.parent = pivot
        self.rotations += 1
        return pivot

    def insert(self, key: float, cluster_data: dict, embedding: np.ndarray) -> None:
        """Insere ou atualiza um cluster, mantendo a árvore balanceada.

        Atualizações de chaves existentes não alteram os contadores.
        """
        new = SemanticNodeRB(key, cluster_data, embedding)
        parent = None
        node = self.root
        while node is not None:
            parent = node
            if new.key < node.key:
                node = node.left
            elif new.key > node.key:
                node = node.right
            else:
                node.cluster_data = new.cluster_data
                node.embedding = new.embedding
                return
        new.parent = parent
        if parent is None:
            self.root = new
        elif new.key < parent.key:
            parent.left = new
        else:
            parent.right = new
        self._insert_fixup(new)

    def _insert_fixup(self, node: SemanticNodeRB) -> None:
        while node.parent is not None and node.parent.color == RED:
            parent = node.parent
            grandparent = parent.parent
            assert grandparent is not None
            if parent is grandparent.left:
                uncle = grandparent.right
                if self._color(uncle) == RED:
                    assert uncle is not None
                    self._recolor(parent, BLACK)
                    self._recolor(uncle, BLACK)
                    self._recolor(grandparent, RED)
                    node = grandparent
                else:
                    if node is parent.right:
                        node = parent
                        self._rotate_left(node)
                        parent = node.parent
                        assert parent is not None
                    self._recolor(parent, BLACK)
                    self._recolor(grandparent, RED)
                    self._rotate_right(grandparent)
            else:
                uncle = grandparent.left
                if self._color(uncle) == RED:
                    assert uncle is not None
                    self._recolor(parent, BLACK)
                    self._recolor(uncle, BLACK)
                    self._recolor(grandparent, RED)
                    node = grandparent
                else:
                    if node is parent.left:
                        node = parent
                        self._rotate_right(node)
                        parent = node.parent
                        assert parent is not None
                    self._recolor(parent, BLACK)
                    self._recolor(grandparent, RED)
                    self._rotate_left(grandparent)
        assert self.root is not None
        self._recolor(self.root, BLACK)

    def semantic_search(self, query_vector: np.ndarray) -> Optional[dict]:
        """Retorna os metadados mais similares ou None para uma árvore vazia.

        Valida a consulta mesmo em árvore vazia. Em empates, preserva o
        primeiro nó encontrado no percurso raiz, esquerda, direita.
        """
        query = _unit_vector(_validate_vector(query_vector))
        best_score = -float("inf")
        best_data = None

        def visit(node: Optional[SemanticNodeRB]) -> None:
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
