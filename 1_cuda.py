import numpy as np
from timeit import default_timer as time

# from numba import vectorize
@vectorize(['float32(float32,float32)'],target='cuda')
def vectorADD(A,B):
    return A+B

def main():
	N = 10000000
	A = np.ones(N,dtype=np.float32)
	B = np.ones(N, np.float32)
	C = np.zeros(N, np.float32)

	start = time()
	C = vectorADD(A,B)
	end = time() - start 
	print("Time taken -%f" %end)
	print("First 5 in C -",C[:5])

if '__name__' == '__main__':
	main()