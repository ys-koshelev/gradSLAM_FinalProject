import torch as th
import matplotlib.pyplot as plt
from LM import PoseFunctionBase, DiffLM, DampingFunction


class ExpFunction(PoseFunctionBase):
    def __init__(self, x, init_params=None):
        self.x = x
        if init_params is None:
            self.params = th.rand(4)
        else:
            self.params = init_params

    def value(self):
        v = self.params[0] * th.exp(self.params[1] * self.x + self.params[2]) + self.params[3]
        return v

    def jacobian(self):
        J = th.stack([th.exp(self.params[1] * self.x + self.params[2]),
                      self.params[0] * self.x * th.exp(self.params[1] * self.x + self.params[2]),
                      self.params[0] * th.exp(self.params[1] * self.x + self.params[2]),
                      th.ones_like(self.x)], dim=1)
        return J

    def evaluate(self, x):
        return self.params[0] * th.exp(self.params[1] * x + self.params[2]) + self.params[3]


# Prepare noisy data to fit
x = th.linspace(0, 1, 25)
true_params = th.Tensor([1.5, 1.5, 0, 10])
y = true_params[0] * th.exp(true_params[1] * x + true_params[2]) + true_params[3] + th.rand_like(x) / 5
y = y.abs()
y.requires_grad = True

# Fit using implemented differentiable LM
s = ExpFunction(x, init_params=th.ones(4))
d = DampingFunction(lam_min=0.1, lam_max=1, D=1, sigma=1e-5)
solver = DiffLM(y=y, function=s, decision_function=d, tol=1e-3, max_iter=1000)
f = solver.optimize()

# Check the gradients
l = th.nn.functional.mse_loss(f.params, true_params)
with th.autograd.set_detect_anomaly(True):
    l.backward()
print('Gradients of y:\n', y.grad)

# Plot results
x_fit = th.linspace(0, 1, 1000)
y_fit = f.evaluate(x_fit)
plt.plot(x, y.detach(), '.', label='Input noisy data')
plt.plot(x_fit.detach(), y_fit.detach(), label='Fitted curve')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('f(x) = a exp(bx + c) + d + N(0, 1/5)')
plt.legend()
plt.show()