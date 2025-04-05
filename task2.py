import numpy as np
from scipy.linalg import lu_factor, lu_solve
p_values = [1.0, 0.1, 0.01, 0.0001, 0.000001]

def matrix(p):
  return np.array([[p + 12, 3, 2, 0, -3, -8, 0, 0],
                  [-6, 26, 0, -3, 8, 7, -7, 7],
                  [-5, -2, -3, -4, -6, 1, 0, 7],
                  [-6, -5, -4, -2, 8, 7, 4, -8],
                  [-2, -3, 1, -2, 10, -8, 6, 0],
                  [7, -7, 2, 2, 0, -24, -6, 4],
                  [-3, -7, 0, 0, 6, 1, 13, 2],
                  [-6, -3, 4, 0, 2, -8, 6, 11]])


for p in p_values:
  A = matrix(p)
  lu, piv = lu_factor(A)

  n = A.shape[0]
  I = np.eye(n)
  A_inv = np.column_stack([lu_solve((lu, piv), I[:, i]) for i in range(n)])

  cond = np.linalg.cond(A)
  R = np.eye(n) - np.dot(A, A_inv)
  R_norm = np.linalg.norm(R)

  print(f"p = {p:.6f}, cond = {cond:.6f}, norm = {R_norm:.6e}")
