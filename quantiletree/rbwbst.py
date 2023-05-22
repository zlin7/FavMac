from quantiletree.wbst import WeightedBST, WeightedNode

RED, BLACK, DOUBLEBLACK = 0, 1, 2

class ColorWeightedNode(WeightedNode):
    def __init__(self, val, color=BLACK, weight=0, parent=None, left=None, right=None) -> None:
        super().__init__(val, weight, parent, left, right)
        self.color = color

DEBUG = False
def debug_print(*args):
    if DEBUG:
        print(*args)
# We first assume all entries are different.
# This could be handled by the doubly linked list
class QuantileTree(WeightedBST):
    def __init__(self, node_cls=ColorWeightedNode, debug=DEBUG):
        super().__init__(node_cls, debug)

    def _check_properties(self):
        if not self.debug: return True
        super()._check_properties()
        def _check_rb(node: ColorWeightedNode):
            if node == self.nil: return True
            if node.color != RED and node.color != BLACK: return False
            return _check_rb(node.left) and _check_rb(node.right)
        assert _check_rb(self.root), "Not all nodes are red and black."
        assert self.root.color == BLACK, "Root color is wrong"
        assert self.nil.color == BLACK, "Nil color is wrong"
        def _check_rr(node: ColorWeightedNode):
            if node == self.nil: return True
            if node.color == RED:
                if node.left.color == RED or node.right.color == RED: return False
            return _check_rr(node.left) and _check_rr(node.right)
        assert _check_rr(self.root), "Some red nodes have red children"
        def _check_bd(root: ColorWeightedNode):
            def _recurse(curr: ColorWeightedNode, num_black: int):
                if curr == self.nil: return num_black, num_black
                if curr.color == BLACK: num_black += 1
                l_min, l_max = _recurse(curr.left, num_black)
                r_min, r_max = _recurse(curr.right, num_black)
                return min(l_min, r_min), max(l_max, r_max)
            min_cnt, max_cnt = _recurse(root, 0)
            return min_cnt == max_cnt
        assert _check_bd(self.root), "Paths with diff # of blacks"

    def __rb_transplant(self, u, v):
        debug_print(f"Transplant U={u.val}, V={v.val}")
        if u.parent == None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        par = v.parent = u.parent
        return par

    def __fix_delete(self, x):
        while x != self.root and x.color == BLACK:
            if x == x.parent.left:
                s = x.parent.right
                if s.color == RED:
                    # case 3.1
                    s.color = BLACK
                    x.parent.color = RED
                    self.rotate_left(x.parent)
                    s = x.parent.right
                if s.left.color == s.right.color == BLACK:
                    # case 3.2
                    s.color = RED
                    x = x.parent
                else:
                    if s.right.color == BLACK: #left is red
                        # case 3.3
                        s.left.color = BLACK
                        s.color = RED
                        self.rotate_right(s)
                        s = x.parent.right
                    # case 3.4
                    s.color = x.parent.color
                    x.parent.color = BLACK
                    s.right.color = BLACK
                    self.rotate_left(x.parent)
                    x = self.root
            else:
                s = x.parent.left
                if s.color == RED:
                    # case 3.1
                    s.color = BLACK
                    x.parent.color = RED
                    self.rotate_right(x.parent)
                    s = x.parent.left
                if s.left.color == s.right.color == BLACK:
                    # case 3.2
                    s.color = RED
                    x = x.parent
                else:
                    if s.left.color == BLACK: #right is red
                        # case 3.3
                        s.right.color = BLACK
                        s.color = RED
                        self.rotate_left(s)
                        s = x.parent.left
                    # case 3.4
                    s.color = x.parent.color
                    x.parent.color = BLACK
                    s.left.color = BLACK
                    self.rotate_right(x.parent)
                    x = self.root
        x.color = BLACK

    def delete(self, val, weight=1):
        # find the node containing key
        node = self.root
        z = self.nil
        to_del = []
        while node != self.nil:
            to_del.append(node)
            if node.val < val:
                node = node.right
            elif node.val > val:
                node = node.left
            else:
                z = node
                break
            #if node.val == val:
            #    z = node
            ##TODO: change to <?
            #node = node.right if node.val <= val else node.left

        if z == self.nil:
            raise ValueError("Couldn't find key in the tree")
            return
        if z.weight < weight:
            raise ValueError("Too much weight to subtract")
        for _ in to_del: _.sum -= weight
        if z.weight > weight:
            z.weight -= weight
            return
        assert z.weight == weight # remove the whole node

        y = z
        y_original_color = y.color
        if z.left == self.nil:
            x = z.right
            self.__rb_transplant(z, z.right)
        elif z.right == self.nil:
            x = z.left
            self.__rb_transplant(z, z.left)
        else:
            y = self.minimum(z.right)
            y_original_color = y.color
            x = y.right #TODO: What if this is nil???
            if y.parent == z: # y==y.parent.right==z.right
                x.parent = y
            else: # y == y.parent.left
                self.__rb_transplant(y, y.right)
                y.right = z.right
                y.right.parent = y

            self.__rb_transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color

        # update sum
        self._update_parent_sum(x.parent)
        if y_original_color == BLACK:
            self.__fix_delete(x)
        debug_print(f"\n Deleted {val} ({weight})")
        debug_print(self, '\n')
        self._check_properties()

    def insert(self, val, weight=1):

        new_node = ColorWeightedNode(val, weight=weight, color=RED, left=self.nil, right=self.nil)

        par = None
        curr = self.root
        while curr != self.nil:
            curr.sum += weight
            debug_print(f"{curr.sum-weight} -> {curr.sum} at {curr.val} (weight={weight})")
            par = curr
            if val < curr.val:
                curr = curr.left
            elif val > curr.val:
                curr = curr.right
            else:
                curr.weight += weight
                debug_print(f"\n Inserted {val} ({weight})")
                debug_print(self, '\n')
                return

        new_node.parent = par
        if par is None:
            self.root = new_node
        elif val < par.val:
            par.left = new_node
        else:
            par.right = new_node


        self.fix_insert(new_node)
        debug_print(f"\n Inserted {val} ({weight})")
        debug_print(self, '\n')
        self._check_properties()

    def rotate_left(self, x):
        #    x    ->    y
        #     y        x
        debug_print(f"Pre L-RORATE ({x.val}):\n", self)
        y = x.right

        new_x_sum = x.left.sum + y.left.sum + x.weight
        new_y_sum = new_x_sum + y.weight + y.right.sum

        x.right = y.left
        if y.left != self.nil:
            y.left.parent = x

        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        # print(f"{new_x_sum} = {x.left.sum + x.right.sum + x.weight}")
        # print(f"{new_y_sum} = {y.left.sum + y.right.sum + y.weight}")
        x.sum = new_x_sum
        y.sum = new_y_sum
        debug_print("Post L-RORATE:\n", self, '\n')

    def rotate_right(self, x):
        #     x    ->   y
        #    y           x
        debug_print(f"Pre R-RORATE ({x.val}):\n", self)
        y = x.left

        new_x_sum = x.right.sum + y.right.sum + x.weight
        new_y_sum = new_x_sum + y.weight + y.left.sum

        x.left = y.right
        if y.right != self.nil:
            y.right.parent = x

        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
        # print(f"{new_x_sum} = {x.left.sum + x.right.sum + x.weight}")
        # print(f"{new_y_sum} = {y.left.sum + y.right.sum + y.weight}")
        x.sum = new_x_sum
        y.sum = new_y_sum
        debug_print("Post R-RORATE:\n", self, '\n')


    def _recolor(self, parent):
        # parent is black, need to recolor
        parent.left.color = BLACK
        parent.right.color = BLACK
        parent.color = RED
        return parent

    def fix_insert(self, curr):
        while self.root != curr and curr.parent.color == RED:
            if curr.parent == curr.parent.parent.right:
                u = curr.parent.parent.left # uncle
                if u.color == RED: # recolor
                    curr = self._recolor(u.parent)
                else: # rotate
                    if curr == curr.parent.left:
                        curr = curr.parent
                        self.rotate_right(curr)
                    curr.parent.color = BLACK
                    curr.parent.parent.color = RED
                    self.rotate_left(curr.parent.parent)
                    # The subtree's root is black so we won't need to continue
            else:
                u = curr.parent.parent.right
                if u.color == RED:
                    curr = self._recolor(u.parent)
                else:
                    if curr == curr.parent.right:
                        curr = curr.parent
                        self.rotate_left(curr)
                    curr.parent.color = BLACK
                    curr.parent.parent.color = RED
                    self.rotate_right(curr.parent.parent)
        self.root.color = BLACK



    def __repr__(self):
        lines = []
        print_tree(self.root, lines)
        return '\n'.join(lines)


def print_tree(node, lines, level=0):
    if node.val is not None:
        print_tree(node.left, lines, level + 1)
        lines.append('-' * 4 * level + '> ' +
                     str(node.val) + ' ' + ('r' if node.color == RED else 'b') + " " + str(node.weight) + " " + str(node.sum))
        print_tree(node.right, lines, level + 1)

if __name__ == '__main__':
    pass