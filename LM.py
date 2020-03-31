import torch as th
from tqdm import tqdm


class PoseFunctionBase:
    def __init__(self, x, params_shape):
        self.x = x
        self.params = th.rand(params_shape)

    def value(self):
        pass

    def jacobian(self):
        pass

    def evaluate(self):
        pass

    def update_params(self, deltas):
        assert self.params.shape == deltas.shape, "Shapes of parameters and updates are different."

        self.params = self.params + deltas
        pass

    @property
    def f(self):
        return self.value()

    @property
    def J(self):
        return self.jacobian()


class DampingFunction:
    def __init__(self, lam_min, lam_max, D, sigma):
        self.lam_max = lam_max
        self.lam_min = lam_min
        self.D = D
        self.sigma = sigma

    def Qlam(self, r0, r1):
        value = self.lam_min + (self.lam_max - self.lam_min)/(1 + self.D*th.exp(-self.sigma*(r1 - r0)))
        return value

    def Qx(self, dx, r1, r0):
        value = dx/(1 + th.exp(-(r1 - r0)))
        return value


class DiffLM:
    def __init__(self, y, function, decision_function, tol=1e-7, max_iter=1000):
        self.decision_function = decision_function
        self.y = y
        self.function = function
        self.tol = tol
        self.max_iter = max_iter

    @property
    def r(self):
        return self.y - self.function.f

    @property
    def r_abs(self):
        r = self.r
        return r.T@r

    def step(self, lam):
        J = self.function.J
        r = self.r
        rhs = J.T@r
        lhs = J.T@J + lam*th.eye(J.shape[-1]).type_as(J)
        deltas = th.inverse(lhs)@rhs
        return deltas

    def optimize(self, verbose=0):
        lam = self.decision_function.lam_max
        r0 = self.r_abs
        r1 = r0.clone()
        
        iterations = range(self.max_iter)
        if verbose == 1:
            iterations = tqdm(iterations)
        for i in range(self.max_iter):
            deltas = self.step(lam)
            step = self.decision_function.Qx(deltas, r0, r1)
            self.function.update_params(step)

            if deltas.norm() < self.tol:
                return self.function

            r0 = r1.clone()
            r1 = self.r_abs
            lam = self.decision_function.Qlam(r0, r1)
            if verbose >= 2:
                print('Deltas: ', deltas)
                print('r_0: ', r0)
                print('r_1: ', r1)
                print('\n')
        print('Did not converge with required tolerance.')
        return self.function