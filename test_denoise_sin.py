import torch as th
import matplotlib.pyplot as plt
from LM import PoseFunctionBase, DiffLM, DampingFunction


class SinFunction(PoseFunctionBase):
    def __init__(self, x, init_params=None):
        self.x = x
        if init_params is None:
            self.params = th.rand(4)
        else:
            self.params = init_params

    def value(self):
        return self.params[0]*th.sin(self.params[1]*self.x + self.params[2]) + self.params[3]

    def jacobian(self):
        J = th.stack([th.sin(self.params[1]*self.x + self.params[2]),
                      self.params[0]*self.x*th.cos(self.params[1]*self.x + self.params[2]),
                      self.params[0]*th.cos(self.params[1]*self.x + self.params[2]),
                      th.ones_like(self.x)], dim=1)
        return J

    def evaluate(self, x):
        return self.params[0] * th.sin(self.params[1] * x + self.params[2]) + self.params[3]


# Prepare noisy data to fit
x = th.linspace(0, 1, 25)
true_params = th.Tensor([1.5, 1.5, 0, 0.1])
y_gt = true_params[0] * th.sin(true_params[1] * x + true_params[2]) + true_params[3]
y = y_gt + th.rand_like(x) / 5
y = y.abs()
y.requires_grad = True
y = th.nn.Parameter(y)
opt = th.optim.SGD([y], lr=0.01)

# Performing denoising
ys = [y.data.clone()]
for i in range(1000):
    s = SinFunction(x, init_params=th.ones(4))
    d = DampingFunction(lam_min=0.1, lam_max=10, D=1, sigma=1e-3)
    solver = DiffLM(y=y, function=s, decision_function=d, tol=1e-3, max_iter=500)
    f = solver.optimize()
    l = th.nn.functional.l1_loss(f.params, true_params)
    opt.zero_grad()
    l.backward()
    opt.step()
    ys.append(y.data.clone())

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
s_true = SinFunction(x, init_params=true_params)
x_fit = th.linspace(0, 1, 1000)
y_fit = f.evaluate(x_fit)
y_fit_true = s.evaluate(x_fit)
ax[0].plot(x, ys[0], '.', label='Before denoising')
ax[0].plot(x, ys[-1], '.', label='After denoising')
ax[0].plot(x_fit.detach(), y_fit_true.detach(), label='True curve')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y = f(x)')
ax[0].set_title('f(x) = a sin(bx + c) + d + N(0, 1/5)')
ax[0].legend()
deltas = [th.nn.functional.mse_loss(y_cur, y_gt) for y_cur in ys]
ax[1].plot(deltas)
ax[1].set_title('MSE error between restored and GT data')
ax[1].set_xlabel('Number of GD steps')
ax[1].set_ylabel('$MSE(y, y_{gt})$')
plt.show()