"""Execute com python -m unittest discover -s tests."""

import unittest

import numpy as np

from src.rb_tree import BLACK, RED, SemanticNodeRB, SemanticRBTree
from src.semantic_avl import SemanticAVL


class SemanticRBTreeTests(unittest.TestCase):
    def setUp(self):
        self.vector = np.ones(1024)

    def assert_invariants(self, tree):
        if tree.root is not None:
            self.assertEqual(tree.root.color, BLACK)
            self.assertIsNone(tree.root.parent)

        def visit(node, parent, low, high):
            if node is None:
                return 1
            self.assertIs(node.parent, parent)
            self.assertLess(low, node.key)
            self.assertLess(node.key, high)
            if node.color == RED:
                for child in (node.left, node.right):
                    self.assertTrue(child is None or child.color == BLACK)
            left = visit(node.left, node, low, node.key)
            right = visit(node.right, node, node.key, high)
            self.assertEqual(left, right)
            return left + (node.color == BLACK)

        visit(tree.root, None, -float("inf"), float("inf"))

    def test_exact_counters(self):
        self.assertEqual(SemanticNodeRB(1, {}, self.vector).color, RED)
        cases = [([], 0, 0), ([1], 0, 1), ([1, 2], 0, 1),
                 ([3, 2, 1], 1, 3), ([1, 2, 3], 1, 3),
                 ([3, 1, 2], 2, 3), ([1, 3, 2], 2, 3),
                 ([2, 1, 3, 0], 0, 5), ([2, 1, 3, 4], 0, 5)]
        for keys, rotations, recolorings in cases:
            with self.subTest(keys=keys):
                tree = SemanticRBTree()
                for key in keys:
                    tree.insert(key, {}, self.vector)
                    self.assert_invariants(tree)
                self.assertEqual((tree.rotations, tree.recolorings),
                                 (rotations, recolorings))

    def test_insertion_sequences(self):
        rng = np.random.default_rng(42)
        for keys in (range(200), reversed(range(200)), rng.permutation(200)):
            tree = SemanticRBTree()
            for key in keys:
                tree.insert(key, {}, self.vector)
                self.assert_invariants(tree)

    def test_search_matches_avl_and_numpy(self):
        rng = np.random.default_rng(7)
        vectors = rng.normal(size=(50, 1024))
        tree, avl = SemanticRBTree(), SemanticAVL()
        for key, vector in enumerate(vectors):
            tree.insert(key, {"key": key}, vector)
            avl.insert(key, {"key": key}, vector)
        for query in rng.normal(size=(10, 1024)):
            expected = int(np.argmax(vectors @ query / np.linalg.norm(vectors, axis=1)))
            self.assertEqual(tree.semantic_search(query), {"key": expected})
            self.assertEqual(tree.semantic_search(query), avl.semantic_search(query))

    def test_duplicate_and_embedding_copy(self):
        tree = SemanticRBTree()
        tree.insert(1, {}, self.vector)
        counters = tree.rotations, tree.recolorings
        tree.insert(1, {"updated": True}, self.vector)
        self.vector[:] = 0
        self.assertEqual(tree.semantic_search(np.ones(1024)), {"updated": True})
        self.assertEqual((tree.rotations, tree.recolorings), counters)
        self.assert_invariants(tree)

    def test_empty_negative_and_ties(self):
        tree = SemanticRBTree()
        self.assertIsNone(tree.semantic_search(self.vector))
        tree.insert(2, {"key": 2}, -self.vector)
        tree.insert(1, {"key": 1}, -self.vector)
        self.assertEqual(tree.semantic_search(self.vector), {"key": 2})

    def test_validation_matches_avl(self):
        invalid = [np.zeros(1024), np.ones(10), np.ones((1, 1024)),
                   np.full(1024, np.nan), np.full(1024, np.inf)]
        for cls in (SemanticRBTree, SemanticAVL):
            tree = cls()
            for vector in invalid:
                with self.assertRaises(ValueError):
                    tree.insert(1, {}, vector)
                with self.assertRaises(ValueError):
                    tree.semantic_search(vector)
            for key in (float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    tree.insert(key, {}, self.vector)
            with self.assertRaises(TypeError):
                tree.insert(1, [], self.vector)
            self.assertIsNone(tree.root)
            if isinstance(tree, SemanticRBTree):
                self.assertEqual((tree.rotations, tree.recolorings), (0, 0))


if __name__ == "__main__":
    unittest.main()
