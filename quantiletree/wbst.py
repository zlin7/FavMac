import sys
from importlib import reload
import math

import ipdb

from .bst import Node, BST

class WeightedNode(Node):
    def __init__(self, val, weight=0, parent=None, left=None, right=None) -> None:
        super().__init__(val, parent, left, right)
        self.weight = weight
        self.sum = weight

    def update_sum(self):
        self.sum = self.weight + self.left.sum + self.right.sum

class WeightedBST(BST):
    def __init__(self, node_cls=WeightedNode, debug=False) -> None:
        super().__init__(node_cls, debug)
        self._eps = 1e-8

    def _check_properties(self):
        if not self.debug: return True
        # weight
        super()._check_properties()
        def check_weight(node: WeightedNode):
            if node == self.nil: return True
            if node.left == self.nil and node.right == self.nil:
                return True

            # assert node.sum == node.left.sum + node.right.sum + node.weight, f"Weight does not sum up: {node.sum} != {node.left.sum} + {node.weight} + {node.right.sum}"
            assert node.sum == node.left.sum + node.right.sum + node.weight, f"Weight does not sum up: {node.sum:.3f} != {node.left.sum:.3f} + {node.weight:.3f} + {node.right.sum:.3f} = {node.left.sum + node.weight + node.right.sum:.3f}"
            check_weight(node.left)
            check_weight(node.right)
        check_weight(self.root)
        assert self.nil.weight == 0 == self.nil.sum, "Nil should have no weight"
        # print("WBST Checks")

    def _update_parent_sum(self, node: WeightedNode):
        while node is not None and node != self.nil:
            node.update_sum()
            node = node.parent

    def query_sum(self, val, inclusive=False):
        def recurse(node):
            if node == self.nil: return 0
            if node.val < val: return node.weight + node.left.sum + recurse(node.right)
            if node.val > val: return recurse(node.left)
            return (node.weight if inclusive else 0) + recurse(node.left)
        return recurse(self.root)
        
    def query_cumu_weight(self, w, prev=True):
        def recurse(node, w):
            w = max(w, 0.) # adjust for numerical issue
            assert node.sum > w - self._eps
            if node.left.sum <= w and w < node.left.sum + node.weight:
                return node
            if node.right == self.nil and (node.left.sum <= w and w < node.left.sum + node.weight + self._eps):
                return node
            if w < node.left.sum:
                return recurse(node.left, w)
            else: # w >= node.left.sum + node.weight:
                assert node.right != self.nil
                return recurse(node.right, w - node.left.sum - node.weight)
        if w >= self.root.sum:
            node = self.maximum(self.root)
        else:
            node = recurse(self.root, w)
            if prev: node = self.predecessor(node)
        return -math.inf if node == self.nil else node.val
        
        """
        Assume the threshold to cost is:
            [(0.1, 1), (0.2, 2), (0.4, 3), (0.5, 7)]
        The mass for each threshold is
            [(0.1, 1), (0.2, 1), (0.4, 1), (0.5, 4)]
        Suppose we want to look up is for cost of 3 (i.e. 0.4)
        Suppose we want to look up cost 4 (we would need 0.4 still)
        This means the range for each threshold is
            -inf: [0, 1)
            0.1 : [1, 2)
            0.2 : [2, 3)
            0.4 : [3, 7) 
            0.5 : [7, infty) 
        We thus need to find the immediate predecesor


        """