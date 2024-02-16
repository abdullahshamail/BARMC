class Hyperparameters(object):
    """
    Attributes:
        m: minimum cluster objects
        k: minimum length of convoy
        threshold: for 'close-enough' molecules
        epsilon: for DBSCAN
        t1: timegap for r-convoy
        t2: percentage timegap allowed
    """
    __slots__ = ('m', 'k', 'threshold', 'epsilon', 't1', 't2')

    def __init__(self, m, k, epsilon, threshold = None, t1 = None, t2 = None):
        self.m = m
        self.k = k
        self.threshold = threshold
        self.epsilon = epsilon
        self.t1 = t1 
        self.t2 = t2 

    def __repr__(self):
        return '<%r %r m=%r, k=%r, threshold=%r, epsilon=%r, t1=%r, t2=%r>' % (self.__class__.__name__, id(self), self.m, self.k, self.threshold, self.epsilon, self.t1, self.t2)
