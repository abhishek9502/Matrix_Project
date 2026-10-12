import numpy as np
import time
def multiplication(A,B):
    return A@B
def transpose(A):
    return A.T
def determinant(A):
    return np.linalg.det(A)
def normalization(A):
    mini_value=np.min(A)
    maxi_value=np.max(A)
    if maxi_value==mini_value:
        return np.zeros_like(A,dtype=float)
    return (A-mini_value)/(maxi_value-mini_value)
A=np.array([[1,2],[3,4]])
B=np.array([[11,12],[13,14]])
print("Matrix_A :",end="")
print(A)
print("Matrix_B :",end="")
print(B)
print("Matrix_Multiplication :",end="")
print(multiplication(A,B))
print("Transpose of Matrix :",end="")
print(transpose(A))
print("Matrix_Determinant :",end="")
print(determinant(A))
print("Normalization of Matrix :",end="")
print(normalization(A))
print("Checking time difference between @ and dot")
rng=np.random.default_rng(42)
A_large=rng.random((500,500))
B_large=rng.random((500,500))
start=time.perf_counter()
result1=multiplication(A_large,B_large)
our_time=time.perf_counter()-start
start=time.perf_counter()
result2=np.dot(A_large,B_large)
dot_time=time.perf_counter()-start
print(f"Performance Diff :")
print(f"Ours time {our_time:.6f} seconds")
print(f"Dot time :{dot_time:.6f} seconds")
print(f"Time Difference :{abs(our_time-dot_time):.6f} seconds")
print("Results are equal:", np.allclose(result1, result2))


