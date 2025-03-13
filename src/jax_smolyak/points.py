import numpy as np


class GaussHermite :

    def __init__(self, m, a) :
        self.m = m
        self.a = a
        self.is_nested = False

    def __call__(self, n) :
        nodes = np.polynomial.hermite.hermgauss(n + 1)[0]
        return self.scale(nodes, self.m, self.a)

    def scale(self, x, m=None, a=None) :
        """
        Affine transformation x -> m + a * x
        x : scalar or array or list of shape (n, d) or (d,)
        m : scalar
        a : positive scalar
        """
        if m is None : m = self.m
        if a is None : a = self.a
        assert a > 0
        return m + a*x

    def scale_back(self, x) :
        return (x - self.m) / self.a

    def get_random(self) :
        return self.scale(np.random.randn())


class Leja :

    nodes = np.array([0, 1, -1, 1/np.sqrt(2), -1/np.sqrt(2)])

    def __init__(self, domain=None) :
        self.domain = domain
        self.is_nested = True

    @classmethod
    def ensure_nodes(cls, n) :
        k = cls.nodes.shape[0]
        if n >= k :
            cls.nodes = np.append(cls.nodes, np.empty((n+1-k,)))
            for j in range(k, n+1) :
                if j % 2 == 0 :
                    cls.nodes[j] = -cls.nodes[j-1]
                else :
                    cls.nodes[j] = np.sqrt((cls.nodes[int((j+1)/2)] + 1) / 2)

    def __call__(self, n) :
        self.ensure_nodes(n)
        if self.domain is None :
            return self.nodes[:n+1]
        return self.scale(self.nodes[:n+1], [-1,1], self.domain)

    def scale(self, x, d1=None, d2=None) :
        """
        Affine transformation from the interval d1 to the interval d2 applied to point x.
        x : scalar or array or list of shape (n, d) or (d,)
        d1, d2 : arrays or lists of shape (d, 2) or (2,)
        """
        if d1 is None : d1 = [-1,1]
        if d2 is None : d2 = self.domain

        # ensure d1, d2 have shape (d, 2)
        d1, d2 = np.array(d1), np.array(d2)
        assert d1.shape == d2.shape
        d = 1 if len(d1.shape) == 1 else d1.shape[0]
        if len(d1.shape) == 1 :
            d1 = d1.reshape((1,2))
            d2 = d2.reshape((1,2))
        for i in range(d) :
            assert d1[i,0] < d1[i,1]
            assert d2[i,0] < d2[i,1]

        # ensure x has shape (n, d)
        x = np.array(x)
        x_shape = x.shape
        if x_shape == () :
            x = np.array([[x]])
        else :
            x = np.array(x)
            if len(x.shape) == 1 :
                if x.shape[0] == d :
                    x = x.reshape((1,len(x)))
                else :
                    x = x.reshape((len(x),1))

        # ensure x in d1
        for i in range(d) :
            for j in range(x.shape[0]) :
                assert (x[j,i] >= d1[i,0] or np.isclose(x[j,i], d1[i,0])),\
                        f'Assertion failed with\n x[{j},{i}] ({x[j,i]})\n d1[{i},0] ({d1[i,0]})'
                assert (x[j,i] <= d1[i,1] or np.isclose(x[j,i], d1[i,1])),\
                        f'Assertion failed with\n x[{j},{i}] ({x[j,i]})\n d1[i,1] ({d1[i,1]})'

        # check
        assert len(x.shape) == len(d1.shape) == len(d2.shape) == 2
        assert x.shape[1] == d1.shape[0] == d2.shape[0]
        assert d1.shape[1] == d2.shape[1] == 2

        # scale
        for i in range(d) :
            x[:,i] = (x[:,i] - d1[i,0]) / (d1[i,1] - d1[i,0])
            x[:,i] = x[:,i] * (d2[i,1] - d2[i,0]) + d2[i,0]

        # ensure x in d2
        #for i in range(d) :
        #    assert (x[:,i] >= d2[i,0]).all(), f"Assertion failed with\n x[:,i] ({x[:,i]})\n d2[i,0] ({d2[i,0]})"
        #    assert (x[:,i] <= d2[i,1]).all(), f"Assertion failed with\n x[:,i] ({x[:,i]})\n d2[i,1] ({d2[i,1]})"

        # return in original shape
        return x.reshape(x_shape)

    def scale_back(self, x) :
        return self.scale(x, d1=self.domain, d2=[-1,1])

    def get_random(self) :
        if self.domain is None :
            return np.random.uniform(-1, 1)
        return np.random.uniform(self.domain[0], self.domain[1])


class Multi :

    def __init__(self, gs) :
        self.gs = gs
        self.d = len(self.gs)
        self.is_nested = gs[0].is_nested

    def __getitem__(self, i) :
        return self.gs[i]

    def get_zero(self) :
        return np.array([g(0)[0] for g in self.gs])

    def get_random(self, n=0) :
        if n == 0 : return np.array([g.get_random() for g in self.gs])
        return np.array([[g.get_random() for g in self.gs] for _ in range(n)])

    def scale(self, x) :
        return np.array([g.scale(xi) for g, xi in zip(self.gs, x)])

    def scale_back(self, x) :
        return np.array([g.scale_back(xi) for g, xi in zip(self.gs, x)])


class GaussHermiteMulti(Multi) :

    def __init__(self, mlist, alist):
        Multi.__init__(self, [GaussHermite(m,a) for m,a in zip(mlist, alist)])

    def print(self) :
        print(f'\tGauss Hermite in d = {len(self.gs)}')
        for g in self.gs :
            print(f'\t\t m = {g.m}, a = {g.a}')


class LejaMulti(Multi) :

    def __init__(self, *, domains=None, d=None) :
        if domains is not None :
            Multi.__init__(self, [Leja(domain) for domain in domains])
        elif d is not None :
            Multi.__init__(self, [Leja() for _ in range(d)])
        else : raise

    def print(self) :
        print(f'\tLeja in d = {len(self.gs)}')
        for g in self.gs :
            print(f'\t\t domain = {g.domain}')
