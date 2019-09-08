from multiprocessing import Pool
import numpy as np
import time
import matplotlib.pyplot as plt

NUM_PROCESSES = 4
sizes = []
distributed_times = []
nondistributed_times = []

def matmul(mat1, mat2):
    return np.matmul(mat1, mat2)
    
def distributed_matmul():
    start = time.time()
    with Pool(processes = NUM_PROCESSES) as pool:
        p1 = pool.apply_async(matmul, (mat1_slice1, mat2_slice1))
        p2 = pool.apply_async(matmul, (mat1_slice2, mat2_slice2))
        p3 = pool.apply_async(matmul, (mat1_slice3, mat2_slice3))
        p4 = pool.apply_async(matmul, (mat1_slice4, mat2_slice4))
        result = np.vstack((np.hstack((p1.get(), p2.get())), np.hstack((p3.get(), p4.get()))))
    end = time.time()
    distributed_time = end - start
    distributed_times.append(distributed_time)
    
if __name__ == '__main__':
    for n in range(0, 4096, 2):
        m = int(n/2)

        mat1 = np.random.rand(n, n)
        mat2 = np.random.rand(n, n)
        
        start = time.time()
        np.matmul(mat1, mat2)
        end = time.time()
        nondistributed_time = end - start
        nondistributed_times.append(nondistributed_time)
        
        mat1_slice1 = mat1[:m, :]
        mat1_slice2 = mat1[:m, :]
        mat1_slice3 = mat1[m:, :]
        mat1_slice4 = mat1[m:, :]

        mat2_slice1 = mat2[:, :m]
        mat2_slice2 = mat2[:, :m]
        mat2_slice3 = mat2[:, m:]
        mat2_slice4 = mat2[:, m:]

        distributed_matmul()

        sizes.append(n)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(sizes, nondistributed_times, c = 'b', label = 'non-distributed times')
    ax1.scatter(sizes, distributed_times, c = 'r', label = 'distributed times')
    plt.legend(loc = 'upper left');
    plt.show()

        
    


    

    

