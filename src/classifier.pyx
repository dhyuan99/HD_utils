import numpy as np

cdef class HD_Classifier:

    cdef np.ndarray memory
    cdef np.ndarray ID
    cdef int count

    cdef int n_classes
    cdef int dim
    def __cinit__(self, int n_classes, int dim):
        self.memory = np.zeros(dim)
        self.ID = np.random.randint(2, size=[n_classes, dim])
        self.count = 0

        self.n_classes = n_classes
        self.dim = dim

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def construct_memory(self, np.ndarray X, np.ndarray y):
        # X is of shape (batch_size, dim)
        # y is of shape (batch_size,)
        assert X.shape[1] == self.dim, 'Dimension does not match.'
        assert X.shape[0] == y.shape[0], 'input and target must have the same number of samples.'
        assert np.array_equal(np.unique(y) == np.arange(self.dim)), 'target must be integers from 0 to n_classes.'

        cdef int[:,:] ID_view = self.ID
        cdef int[:,:] X_view = X
        cdef Py_ssize_t[:] y_view = y

        cdef Py_ssize_t i, j
        cdef int sum_xor
        for i in range(self.dim):
            sum_xor = 0
            for j in range(X.shape[0]):
                sum_xor += X_view[i,j] * (1 - ID_view[y_view[i], j]) + (1 - X_view[i,j]) * ID_view[y_view[i], j]
            self.memory[i] += sum_xor

        self.count += X.shape[0]
