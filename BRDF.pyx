# distutils: language=c

import numpy as np
cimport numpy as cnp
cimport cython

cdef class densityRF:
    
    cdef:
        int nTrees
        int minSize
        cnp.float64_t N 
        float p
        list rf
        cnp.float64_t[:, :] initDomain
        
    def __init__(self, nTrees = 20, minSize = 5, p = 0.1):
        self.nTrees = nTrees
        self.minSize = minSize
        self.p = p
        self.rf = []
        self.N = 0
        self.initDomain = np.empty((1,2))
        
    @cython.boundscheck(False)
    @cython.wraparound(False) 
    cpdef fit(self, cnp.ndarray[cnp.float64_t, ndim=2] data):
        cdef int i
        cdef cnp.ndarray[cnp.float64_t, ndim=2] domain
        
        self.N = data.shape[1]
        self.initDomain = self.initialDomain(data)
        for i in range(self.nTrees):
            self.rf.append(self.buildTree(data, np.asarray(self.initDomain)))
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef initialDomain(self, cnp.ndarray[cnp.float64_t, ndim=2] data):
        cdef cnp.ndarray[cnp.float64_t, ndim=2] domain = np.empty((data.shape[0],2))
        domain[:,0] = np.amin(data, axis = 1)
        domain[:,1] = np.amax(data, axis = 1)
        return domain
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef buildTree(self, cnp.ndarray[cnp.float64_t, ndim=2] data, 
                        cnp.ndarray[cnp.float64_t, ndim=2] domain):
        cdef cnp.float64_t errorReduction
        cdef cnp.ndarray[cnp.float64_t, ndim=2] left
        cdef cnp.ndarray[cnp.float64_t, ndim=2] domainLeft
        cdef cnp.ndarray[cnp.float64_t, ndim=2] right
        cdef cnp.ndarray[cnp.float64_t, ndim=2] domainRight
        cdef object leftBranch, rightBranch
        cdef object test

        if  data.shape[1] < self.minSize:
            #print(domain)
            #print(self.volume(domain))
            return Leaf(data, self.volume(domain), self.N)
        else:        
            if np.random.uniform(0,1) < self.p:
                # Random split
                test = self.randomSplit(data, domain)
                left, right = self.partition(data, test)
            else:
                # Impurity criterion
                test, errorReduction = self.bestSplit(data, domain)
                if errorReduction == 0:
                    #print(domain)
                    #print(self.volume(domain))
                    return Leaf(data, self.volume(domain), self.N)
                else:
                    left, right = self.partition(data, test)
                        
            # Recursively build the left branch.
            domainLeft = np.copy(domain)
            domainLeft[test.feature, 1] = test.value
            leftBranch = self.buildTree(left, domainLeft)
        
            # Recursively build the right branch.
            domainRight = np.copy(domain)
            domainRight[test.feature, 0] = test.value
            rightBranch = self.buildTree(right, domainRight)
        
            return Node(test, leftBranch, rightBranch)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef randomSplit(self, cnp.ndarray[cnp.float64_t, ndim=2] data, 
                          cnp.ndarray[cnp.float64_t, ndim=2] domain):
        cdef int feature
        cdef cnp.float64_t value
        
        feature = np.random.randint(0, data.shape[0])
        value = np.random.uniform(domain[feature,0], domain[feature,1])
        #print(value)
        return Test(feature, value)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bestSplit(self, cnp.ndarray[cnp.float64_t, ndim=2] data, 
                        cnp.ndarray[cnp.float64_t, ndim=2] domain):
        cdef object bestTest
        cdef cnp.ndarray[cnp.int_t, ndim=1] featureBag
        cdef cnp.ndarray[cnp.float64_t, ndim=1] values
        cdef cnp.ndarray[cnp.int64_t, ndim=1] counts
        cdef cnp.float64_t bigReduction = 0
        cdef cnp.float64_t reduction
        cdef cnp.float64_t currentError, errorLeft, errorRight
        cdef cnp.float64_t vol, volLeft, volRight
        cdef cnp.float64_t val
        cdef int nFeatures, nData, nLeft, vLen
        cdef int col
        cdef size_t i
        
        vol = self.volume(domain)
        nFeatures = data.shape[0]
        nData = data.shape[1]
        currentError = -(nData**2)/(vol*self.N**2)
        #FEATURE BAGGING
        featureBag = np.random.choice(nFeatures, 
                                      size = int(np.sqrt(nFeatures)),
                                      replace = False)
        for col in featureBag:
            values, counts = np.unique(data[col,:], return_counts=True)
            vLen = values.size
            nTotal = np.sum(counts)
            nLeft = counts[0]
            for i in range(1,vLen-1):
                val = values[i]
                #Do not consider extremes of the interval to avoid cells with volume = 0
                volLeft = vol*(val - domain[col,0])/(domain[col,1] - domain[col,0])
                volRight = vol*(domain[col,1] - val)/(domain[col,1] - domain[col,0])
                errorLeft = -(nLeft**2)/(volLeft*self.N**2)
                errorRight = -(nTotal - nLeft)**2/(volRight*self.N**2)
                reduction = currentError - errorLeft - errorRight
                nLeft += counts[i]
            
                if reduction >= bigReduction:
                    bigReduction = reduction
                    bestTest = Test(col,val)
                    
        if bigReduction > 0:
            return bestTest, bigReduction
        else:
            return Test(0,0), 0
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef volume(self, cnp.ndarray[cnp.float64_t, ndim=2] domain):
        cdef cnp.float64_t vol = 1
        cdef size_t i
        
        for i in range(domain.shape[0]):
            vol *= domain[i,1] - domain[i,0]
        return vol
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef partition(self, cnp.ndarray[cnp.float64_t, ndim=2] data, object test):
        cdef cnp.ndarray[cnp.float64_t, ndim=2] left
        cdef cnp.ndarray[cnp.float64_t, ndim=2] right
        
        left = data[:, data[test.feature,:] < test.value]
        right = data[:, data[test.feature,:] >= test.value]
        return left, right
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef estimate(self, cnp.ndarray[cnp.float64_t, ndim=2] data):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] preds = np.zeros(data.shape[1])
        cdef size_t i, j
        
        for i in range(self.nTrees):
            for j in range(data.shape[1]):
                if self.outOfBox(data[:,j]):
                    preds[j] += 1 / (self.N + 1)
                else:
                    preds[j] +=  (self.N / ( self.N + 1 ))*self.densityEstimation(self.rf[i], data[:,j])
        return preds/self.nTrees
    
    cdef densityEstimation(self, object node, cnp.ndarray[cnp.float64_t, ndim=1] example):
        if isinstance(node, Leaf):
            return node.density
        if node.test.match(example):
            return self.densityEstimation(node.right, example)
        else:
            return self.densityEstimation(node.left, example)
        
    # Check whether the example is out of the initial domain
    cdef outOfBox(self, cnp.ndarray[cnp.float64_t, ndim=1] example):
        for i in range(self.initDomain.shape[0]):
            if self.initDomain[i,0] > example[i] or self.initDomain[i,1] < example[i]:
                return True
        return False

class Test:
    def __init__(self, int feature, cnp.float64_t value):
        self.feature = feature
        self.value = value

    def match(self, cnp.ndarray[cnp.float64_t, ndim=1] example):
        return example[self.feature] >= self.value

class Node:
    def __init__(self, object test, object left, object right):
        self.test = test
        self.left = left
        self.right = right
    
class Leaf:
    def __init__(self, cnp.ndarray[cnp.float64_t, ndim=2] data, cnp.float64_t volume, cnp.float64_t N):
        self.density = data.shape[1]/(volume*N)

