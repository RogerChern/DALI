import os
import random
import torch
import tqdm


def matrix_dist(matA, matB):
    """
    Compute affinity between m vecs in mat1 and n vecs in mat2
    matA: m x k
    matB: n x k
    return: m x n
    """
    if matA.dim() == 1:
        matA = matA[None, :]
    if matB.dim() == 1:
        matB = matB[None, :]
    assert matA.dim() == 2
    assert matB.dim() == 2
    assert matA.size(1) == matB.size(1)

    A_square = torch.norm(matA, dim=1, keepdim=True) ** 2  # m x 1
    B_square = torch.norm(matB, dim=1, keepdim=True).transpose(0, 1) ** 2  # 1 x n
    AB = matA @ matB.transpose(0, 1)  # m x n
    return A_square + B_square - 2 * AB


def test_matrix_dist(m, n):
    mat1 = torch.randn(m, 10)
    mat2 = torch.randn(n, 10)
    dist_matrix = matrix_dist(mat1, mat2)
    i, j = random.choice(range(m)), random.choice(range(n))
    lhs = dist_matrix[i, j]
    rhs = torch.norm(mat1[i] - mat2[j]) ** 2
    assert torch.allclose(lhs, rhs), f"lhs={lhs}\nrhs={rhs}" 


def matrix_dist_test_suite():
    test_matrix_dist(3, 2)
    test_matrix_dist(10, 10)
    test_matrix_dist(200, 10)
    test_matrix_dist(100000, 1)
    test_matrix_dist(1, 100000)
    print("matrix_dist is tested!")


def kcenter_greedy(feat_mat, n_centers, seeds):
    assert isinstance(feat_mat, torch.Tensor) and feat_mat.dim() == 2
    if isinstance(seeds, int):
        seeds = [seeds]
    assert isinstance(seeds, list)
    
    selection = set(seeds)

    min_dist_mat, _ = matrix_dist(feat_mat, feat_mat[seeds, :]).min(axis=1, keepdim=True)
    min_dist_mat[seeds, :] = -1e15

    for i in tqdm.tqdm(range(n_centers - len(seeds))):
        argmax_min_dist_mat = min_dist_mat.argmax().squeeze().cpu().numpy().item()
        selection.add(argmax_min_dist_mat)
        min_dist_mat[argmax_min_dist_mat] = -1e15
        increment_min_dist_mat = matrix_dist(feat_mat, feat_mat[argmax_min_dist_mat])
        increment_min_dist_mat[argmax_min_dist_mat] = -1e15
        min_dist_mat = torch.min(min_dist_mat, increment_min_dist_mat)

    return selection


def speed_test_kcenter_greedy(n, k):
    print(kcenter_greedy(torch.randn(n, 2048).cuda(), k, [0]))


def test_kcenter_greedy():
    xs, ys = torch.meshgrid(torch.arange(10), torch.arange(10))
    points = torch.cat([xs.reshape(-1, 1), ys.reshape(-1, 1)], dim=1)  # 100 x 2
    points = points[torch.randperm(100)]
    selections = kcenter_greedy(points.float(), 10, [0, 2])
    canvas = torch.zeros(size=(10, 10))
    for point in points[list(selections)]:
        canvas[point[0], point[1]] = 1
    print(canvas)


def sample_with_kcenter_greedy(feature_file):
    import pyarrow as pa
    with open(feature_file, "rb") as fin:
        features = pa.deserialize_from(fin, None)
        features = torch.from_numpy(features).float().cuda()
        selections = kcenter_greedy(features, 64100, [0])
    with open(os.path.expanduser("~/datasets/imagenet/train.lst.full")) as fin:
        file_list = []
        control_file_list = []
        for i, line in enumerate(fin):
            if i in selections:
                file_list.append(line)
            if i + 1 in selections:
                control_file_list.append(line)
    with open(os.path.expanduser("~/datasets/imagenet/train.lst.r50_oracle_1"), "w") as fout:
        for line in file_list:
            fout.write(line)
    with open(os.path.expanduser("~/datasets/imagenet/train.lst.r50_oracle_1_control"), "w") as fout:
        for line in control_file_list:
            fout.write(line)


def sample_with_kcenter_greedy_v2(feature_file):
    import json
    import pyarrow as pa
    with open(os.path.expanduser("~/datasets/imagenet/first.1000.id.json")) as fin:
        seeds = json.load(fin)
    with open(feature_file, "rb") as fin:
        features = pa.deserialize_from(fin, None)
        features = torch.from_numpy(features).float().cuda()
        selections = kcenter_greedy(features, 64100, seeds)
    with open(os.path.expanduser("~/datasets/imagenet/train.lst.full")) as fin:
        file_list = []
        control_file_list = []
        for i, line in enumerate(fin):
            if i in selections:
                file_list.append(line)
            if i + 1 in selections:
                control_file_list.append(line)
    with open(os.path.expanduser("~/datasets/imagenet/train.lst.r50_oracle_2"), "w") as fout:
        for line in file_list:
            fout.write(line)
    with open(os.path.expanduser("~/datasets/imagenet/train.lst.r50_oracle_2_control"), "w") as fout:
        for line in control_file_list:
            fout.write(line)

if __name__ == "__main__":
    # matrix_dist_test_suite()
    # speed_test_kcenter_greedy(1280000, 60000)
    # test_kcenter_greedy()
    sample_with_kcenter_greedy("exps/r50_GAP_1.3M.pa")
