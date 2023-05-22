import sys

import ipdb

class Node:
    def __init__(self, val, parent=None, left=None, right=None) -> None:
        self.parent = parent
        self.val = val
        self.left = left
        self.right = right

class BST:
    def __init__(self, node_cls=Node, debug=False) -> None:
        self.nil = node_cls(None)
        self.root = self.nil
        self.debug = debug

    def _check_properties(self):
        if not self.debug: return True
        def recurse(node: Node):
            if node == self.nil: return True
            assert node.val is not None
            if node.left != self.nil:
                assert node.left.val < node.val, f"left={node.left.val} > parent={node.val}"
            if node.right != self.nil:
                assert node.right.val > node.val, f"right={node.right.val} < parent={node.val}"
            recurse(node.left)
            recurse(node.right)

        recurse(self.root)
        
        assert self.nil.val is None
        assert self.nil.left is None
        assert self.nil.right is None
        # print("BST Checks")
        return True

    def __pre_order_helper(self, node):
        if node != self.nil:
            sys.stdout.write(node.val + " ")
            self.__pre_order_helper(node.left)
            self.__pre_order_helper(node.right)

    def __in_order_helper(self, node):
        if node != self.nil:
            self.__in_order_helper(node.left)
            sys.stdout.write(node.val + " ")
            self.__in_order_helper(node.right)

    def __post_order_helper(self, node):
        if node != self.nil:
            self.__post_order_helper(node.left)
            self.__post_order_helper(node.right)
            sys.stdout.write(node.val + " ")

    def __search_tree_helper(self, node, key):
        if node == self.nil or key == node.val:
            return node

        if key < node.val:
            return self.__search_tree_helper(node.left, key)
        return self.__search_tree_helper(node.right, key)

    # Pre-Order traversal
    # Node.Left Subtree.Right Subtree
    def preorder(self):
        self.__pre_order_helper(self.root)

    # In-Order traversal
    # left Subtree . Node . Right Subtree
    def inorder(self):
        self.__in_order_helper(self.root)

    # Post-Order traversal
    # Left Subtree . Right Subtree . Node
    def postorder(self):
        self.__post_order_helper(self.root)

    # search the tree for the key k
    # and return the corresponding node
    def searchTree(self, k):
        return self.__search_tree_helper(self.root, k)

    # find the node with the minimum key
    def minimum(self, node):
        while node.left != self.nil:
            node = node.left
        return node

    # find the node with the maximum key
    def maximum(self, node):
        while node.right != self.nil:
            node = node.right
        return node

    # find the successor of a given node
    def successor(self, x):
        # if the right subtree is not None,
        # the successor is the leftmost node in the
        # right subtree
        if x.right != self.nil:
            return self.minimum(x.right)

        # else it is the lowest ancestor of x whose
        # left child is also an ancestor of x.
        y = x.parent
        while y is not None and y != self.nil and x == y.right:
            x = y
            y = y.parent
        if y is None: y = self.nil
        return y

    # find the predecessor of a given node
    def predecessor(self,  x):
        # if the left subtree is not None,
        # the predecessor is the rightmost node in the 
        # left subtree
        if (x.left != self.nil):
            return self.maximum(x.left)

        y = x.parent
        while y is not None and y != self.nil and x == y.left:
            x = y
            y = y.parent
        if y is None: y = self.nil
        return y