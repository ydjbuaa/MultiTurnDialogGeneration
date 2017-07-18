# -*- coding:utf-8 -*-
# tree object from stanfordnlp/treelstm


class Tree(object):
    def __init__(self):
        self.parent = None
        self.idx = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def child_enriched_structure(self):
        ret = [self.idx]
        if self.num_children > 0:
            for i in range(self.num_children):
                ret += self.children[i].child_enriched_structure()
        return ret

    def head_enriched_structure(self):
        ret = []
        if self.num_children > 0:
            for i in range(self.num_children):
                ret += self.children[i].head_enriched_structure()
        ret += [self.idx]
        return ret


def read_tree(tree_line):
    parents = list(map(int, tree_line.split()))
    trees = dict()
    root = None
    for i in range(1, len(parents) + 1):
        # if not trees[i-1] and parents[i-1]!=-1:
        if i - 1 not in trees.keys() and parents[i - 1] != -1:
            idx = i
            prev = None
            while True:
                parent = parents[idx - 1]
                if parent == -1:
                    break
                tree = Tree()
                if prev is not None:
                    tree.add_child(prev)
                trees[idx - 1] = tree
                tree.idx = idx - 1
                # if trees[parent-1] is not None:
                if parent - 1 in trees.keys():
                    trees[parent - 1].add_child(tree)
                    break
                elif parent == 0:
                    root = tree
                    break
                else:
                    prev = tree
                    idx = parent
    return root


def get_corresponding_order(order):
    order_dict = {}
    for i, idx in enumerate(order):
        order_dict[idx] = i
    return [order_dict[i] for i in range(len(order))]
    pass
# if __name__ == '__main__':
#     tree_line = "3 3 6 6 4 0 6"
#     tree = read_tree(tree_line)
#     sents = '欧盟 文化 部长 在 意大利 举行 会议'.split()
#     ces_order = []
#     ces_indices = []
#     for i in tree.child_enriched_structure():
#         ces_indices.append(i)
#         ces_order.append(sents[i])
#
#     head_order = []
#     head_indices = []
#     for i in tree.head_enriched_structure():
#         head_indices.append(i)
#         head_order.append(sents[i])
#
#     ids = list(range(len(sents)))
#     ids = [str(i) for i in ids]
#     print('    '.join(ids))
#
#     raw_order = [head_order[j] for j in get_corresponding_order(head_indices)]
#     raw_order2 = [ces_order[j] for j in get_corresponding_order(ces_indices)]
#     print(' '.join(ces_order))
#     print(' '.join(head_order))
#     print(' ')
#     print('raw order from head', ' '.join(raw_order))
#     print('raw order from child', ' '.join(raw_order2))
#     print(' '.join(sents))
#     print(head_indices)
