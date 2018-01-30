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
random = np.random.RandomState(666162123)

x_data = random.randint(0, 1000, (m_data, n_data))


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
    # node_k = list()
    min_dis = compute_distance(root, point)
    NN = root
    compare_node = root
    while compare_node:
        stack.append(compare_node)
        dim = compare_node.dimension
        dis = compute_distance(compare_node, point)
        if dis < min_dis:
            min_dis = dis
            NN = compare_node

        # 当前维度小就向左走，大就向右走
        if point[dim] < compare_node.data[dim]:
            compare_node = compare_node.lchild
        else:
            compare_node = compare_node.rchild
    # 回溯
    while stack:
        node = stack.pop()
        dim = node.dimension
        if compute_distance(node, point) < min_dis:
        # if abs(point[dim] - node.data[dim]) < min_dis:
            if point[dim] <= node.data[dim]:
                temp_node = node.rchild
            else:
                temp_node = node.lchild

            if temp_node:
                stack.append(temp_node)
                dis = compute_distance(temp_node, point)
                if dis < min_dis:
                    min_dis = dis
                    NN = temp_node
    return NN, min_dis


kd_tree = create_kd_tree(kd_tree, x_data, -1)
root = kd_tree
NN, MIN = search_nn(root, np.array([1, 3, 5]), K)

print(NN, MIN)
min_dis = list()
for point in x_data:
    dis = compute_distance(point, np.array([1, 3, 5]))
    if len(min_dis) < K:
        min_dis.append(dis)
    else:
        min_dis = sorted(min_dis)
        if min_dis[-1] > dis:
            min_dis[-1] = dis

print(min_dis)