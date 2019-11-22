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


def kcenter_greedy(feat_mat, n_centers):
    selection = set()

    min_dist_mat = matrix_dist(feat_mat, feat_mat[0])
    min_dist_mat[0] = -1
    selection.add(0)

    for i in tqdm.tqdm(range(n_centers - 1)):
        argmax_min_dist_mat = min_dist_mat.argmax().squeeze().cpu().numpy().item()
        selection.add(argmax_min_dist_mat)
        min_dist_mat[argmax_min_dist_mat] = -1
        increment_min_dist_mat = matrix_dist(feat_mat, feat_mat[argmax_min_dist_mat])
        increment_min_dist_mat[argmax_min_dist_mat] = -1
        min_dist_mat = torch.min(min_dist_mat, increment_min_dist_mat)

    return selection

def test_kcenter_greedy(n, k):
    print(kcenter_greedy(torch.randn(n, 2048).cuda(), k))


if __name__ == "__main__":
    # matrix_dist_test_suite()
    test_kcenter_greedy(1280000, 60000)
