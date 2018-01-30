# coding: utf-8
import numpy as np


class Node:
    def __init__(self, data, dimension):
        self.data = data
        self.dimension = dimension
        self.lchild = None
        self.rchild = None


kd_tree = None
K = 5
n_data = 3  # 每个点的维度 n
m_data = 1000  # m个点
random = np.random.RandomState(666)

x_data = random.randint(-1000, 1000, (m_data, n_data))


# print(x_data)

def max_var_dim(data, exclude_dim):
    '''
    找出除父节点维度外的方差最大的维度
    :param data:
    :param exclude_dim: 父节点的维度
    :return:
    '''
    col = data.shape[1]
    max_var = 0
    max_index = 0
    for i in range(col):
        if i == exclude_dim:
            continue
        var = data[:, i].var()
        if var > max_var:
            max_var = var
            max_index = i

    return max_index


def create_kd_tree(root, data_list, exclude_dim):
    if data_list.shape[0] == 0:
        return None
        # 找到该维度的中位点
    split_dim = max_var_dim(data_list, exclude_dim)
    # print(split_dim)
    data_list = np.array(sorted(data_list, key=lambda x: x[split_dim]))  # 按照第split_dim 排序
    mid = int(data_list.shape[0] / 2)
    node = data_list[mid, :]  # 中位点
    root = Node(node, split_dim)
    # 在此维度上中位点分开的两部分进行递归
    root.lchild = create_kd_tree(root.lchild, data_list[:mid], split_dim)
    root.rchild = create_kd_tree(root.rchild, data_list[mid + 1:], split_dim)
    return root


def compute_distance(p1: Node, p2: np.array):
    return np.linalg.norm((p1.data - p2))


def compute_distance2(p1, p2: np.array):
    return np.linalg.norm((p1 - p2))


def search_nn(root: Node, point, k):
    stack = list()  # [(node, dis)]
    node_k = list()
    compare_node = root
    while compare_node:
        stack.append(compare_node)
        dim = compare_node.dimension
        dis = compute_distance(compare_node, point)
        # 不满 k个node
        if len(node_k) < k:
            node_k.append((compare_node, dis))
        else:
            node_k_max = max(node_k, key=lambda x: x[1])
            node_k_max_index = node_k.index(node_k_max)
            # 将距离最大的pop
            if dis < node_k[node_k_max_index][1]:
                node_k[node_k_max_index] = (compare_node, dis)

        # 当前维度小就向左走，大就向右走
        if point[dim] < compare_node.data[dim]:
            compare_node = compare_node.lchild
        else:
            compare_node = compare_node.rchild
    # 回溯
    while stack:
        # node_k 的中最大距离最大的(node, dis)
        node_k_max = max(node_k, key=lambda x: x[1])
        node = stack.pop()
        # 进入另一个结点
        # for node_, dis in node_k:
        #     print(node_, dis)
        # print(compute_distance(node, point))
        dim = node.dimension
        if len(node_k) < k or compute_distance(node, point) < node_k_max[1]:
        # if len(node_k) < k or abs(point[dim] - node.data[dim]) < node_k_max[1]:
            # 当前节点的dimension
            # for node_, dis in node_k:
            #     print(node_, dis)
            # print(compute_distance(node, point))
            dim = node.dimension
            if point[dim] <= node.data[dim]:
                compare_node = node.rchild
            else:
                compare_node = node.lchild
            # compare_node不是叶子结点
            if compare_node:
                stack.append(compare_node)
                # 更新 node_k
                dis = compute_distance(compare_node, point)
                if len(node_k) < k:
                    node_k.append((compare_node, dis))
                else:
                    if dis < node_k_max[1]:
                        node_k_max_index = node_k.index(node_k_max)
                        node_k[node_k_max_index] = (compare_node, dis)

    return node_k


kd_tree = create_kd_tree(kd_tree, x_data, -1)
root = kd_tree
import time
start = time.time()
result = search_nn(root, np.array([1, 3, 5]), K)
end = time.time()
print(end - start)
for node, dis in result:
    print(node, dis)

start = time.time()
min_dis = list()
for point in x_data:
    dis = compute_distance(point, np.array([1, 3, 5]))
    if len(min_dis) < K:
        min_dis.append(dis)
    else:
        min_dis = sorted(min_dis)
        if min_dis[-1] > dis:
            min_dis[-1] = dis
end = time.time()
print(end - start)
print(min_dis)
