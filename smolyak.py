import sys
import numpy as np
import itertools as it
from copy import deepcopy

sys.path.append('../')
from interpolation.indices import indexset_sparse, abs_e_sparse
from interpolation.tensorproduct import TensorProductBarycentricInterpolator
from interpolation.points import test_gens


class SmolyakBarycentricInterpolator :

    def __init__(self, gens, k, l, f=None) :
        self.k = k
        self.operators = []
        self.coefficients = []
        self.is_nested = gens.is_nested
        kmap = lambda j : k[j]
        i = indexset_sparse(kmap, l, cutoff=len(k))
        for nu in i :
            c = np.sum([(-1)**e for e in abs_e_sparse(kmap, l, nu=nu, cutoff=len(k))])
            if c != 0 :
                self.operators.append(TensorProductBarycentricInterpolator(gens, nu, len(k)))
                self.coefficients.append(c)
        if self.is_nested :
            self.n = len(i)
        else :
            self.n = int(np.sum([np.prod(o.F.shape) for o in self.operators]))
        self.n_f_evals = 0
        if f is not None :
            self.set_F(f)

    def set_F(self, f, F=None, i=None) :
        if F is None : F = {}
        if self.is_nested :
            for o in self.operators :
                for idx in it.product(*(range(d+1) for d in o.degrees)) :
                    ridx = o.reduced_index(idx)
                    if idx not in F.keys() :
                        o.set_x(ridx)
                        #F[idx] = {'x' : deepcopy(o.x), 'Fx' : None}
                        F[idx] = f(o.x)
                        self.n_f_evals += 1
                    if i is None :
                        #continue
                        o.F[ridx] = F[idx]
                    else :
                        o.F[ridx] = F[idx][i]
        else :
            for o in self.operators :
                Fo = F.get(o.degrees, {})
                for idx in it.product(*(range(d+1) for d in o.degrees)) :
                    ridx = o.reduced_index(idx)
                    if idx not in Fo.keys() :
                        o.set_x(ridx)
                        Fo[idx] = f(o.x)
                        self.n_f_evals += 1
                    if i is None :
                        o.F[ridx] = Fo[idx]
                    else :
                        o.F[ridx] = Fo[idx][i]
                F[o.degrees] = Fo
        return F

    def __call__(self, x) :
        r = 0
        for c, o in zip(self.coefficients, self.operators) :
            r += c*o(x)
        return r

    def get_max_degrees(self) :
        max_degrees = list(self.operators[0].degrees)
        for o in self.operators[1:] :
            for i in range(len(max_degrees)) :
                max_degrees[i] = max(o.degrees[i], max_degrees[i])
        return max_degrees

    def gen_test_f(self) :
        class test_f :
            def __init__(self, parent, n) :
                self.coeffs = np.random.rand(n)*2 - 1
                self.coeffs /= len(self.coeffs)
                idxs = np.random.randint(low=0, high=len(parent.operators), size=n)
                self.fs = [parent.operators[i].gen_test_f() for i in idxs]
            def __call__(self, x) :
                res = 0
                for c, fi in zip(self.coeffs, self.fs) :
                    res += c * fi(x)
                return res
        return test_f(self, 1)


class MultivariateSmolyakBarycentricInterpolator :

    def __init__(self, *, g, k, l, f=None) :
        self.components = [SmolyakBarycentricInterpolator(g, k, li) for li in l]
        self.n = max(c.n for c in self.components)
        self.F = None
        if f is not None :
            self.set_F(f=f)

    def set_F(self, *, f, F=None) :
        assert self.F is None
        if F is None : F = {}
        for i, c in enumerate(self.components) :
            F = c.set_F(f, F, i)
        self.F = F

        return F

    def __call__(self, x) :
        res = np.array([c(x) for c in self.components]).T
        assert res.shape[res.ndim-1] == len(self.components)
        return res

    def print(self) :
        for i,c in enumerate(self.components) :
            print('i = {}'.format(i))
            for o in c.operators :
                print('\t', o.degrees)

    def gen_test_f(self) :
        class test_f :
            def __init__(self, fs) :
                self.fs = fs
            def __call__(self, x) :
                return np.array([fi(x) for fi in self.fs])
        fs = [c.gen_test_f() for c in self.components]
        return test_f(fs)


if __name__ == '__main__' :

    #NOTE: This test code serves only the purpose of ensuring the interface works. For convergence tests, see notebook.

    for g in test_gens(10,3) :

        k = sorted(np.random.randint(low=1, high=10, size=g.d))
        k /= k[0]
        print(f'Testing with d = {g.d}, k = {k}')
        #g.print()

        if True :
            print('\n\tTEST SmolyakBarycentricInterpolator')

            ip = SmolyakBarycentricInterpolator(g, k, 2)
            f = ip.gen_test_f()
            ip.set_F(f)

            for n in range(5) :
                x = g.get_random()
                print(f'\t\t ip(x) = {ip(x)}, f(x) = {f(x)}')
                assert np.isclose(ip(x), f(x)), \
                   f'Assertion failed with\n x = {x}\n f(x) = {f(x)}\n ip(x) = {ip(x)}'

            print('\tSUCCESSFUL\n')

        if False :
            d2 = np.random.randint(low=1, high=5)
            k2 = sorted(np.random.randint(low=1, high=10, size=d2), reverse=True)

            print('\tTEST MultivariateSmolyakBarycentricInterpolator')
            print(f'\t    Testing with d2 = {d2}, k2 = {k2}')

            ip = MultivariateSmolyakBarycentricInterpolator(g, k, k2)
            f = ip.gen_test_f()
            ip.set_F(f)

            for n in range(5) :
                x = g.get_random()
                print(f'\t\t ip(x) = {ip(x)}, f(x) = {f(x)}')
                assert np.isclose(ip(x), f(x)).all(), \
                   f'Assertion failed with\n x = {x}\n f(x) = {f(x)}\n ip(x) = {ip(x)} @ n = {n}'

            print('\tSUCCESSFUL\n')
