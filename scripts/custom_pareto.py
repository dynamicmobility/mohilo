import matplotlib.pyplot as plt
import numpy as np

def f1(x):
    return -(x - 4) ** 2
    # return np.sin(x)

def f2(x):
    return -(x - 1) ** 2

fig, axs = plt.subplots(ncols=3, figsize=(12, 4))
L, H, N = 0, 6, 100
x = np.linspace(L, H, N)
norm_vary = (x - L) / (H - L)
cols = np.vstack([norm_vary, 1 - norm_vary, np.zeros_like(norm_vary)]).T
print(cols.shape)
ax_f1, ax_f2, ax_pareto = axs

ax_f1.scatter(x, f1(x), label='f1', c=cols)
ax_f1.set_title('Objective 1')
ax_f1.set_xlabel('x')
ax_f1.set_ylabel('f1(x)')

ax_f2.scatter(x, f2(x), label='f2', c=cols)
ax_f2.set_title('Objective 2')
ax_f2.set_xlabel('x')
ax_f2.set_ylabel('f2(x)')

ax_pareto.scatter(f1(x), f2(x), label='Collection', c=cols)
ax_pareto.set_title('Pareto Front')
ax_pareto.set_xlabel('f1(x)')
ax_pareto.set_ylabel('f2(x)')

fig.tight_layout()
fig.savefig('pareto_front.pdf')