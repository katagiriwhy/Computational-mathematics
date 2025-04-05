import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

x = np.array([-1.0, -0.9, -0.8, -0.7, -0.6, -0.5])
f_x = np.array([0.5440, -0.4121, -0.9894, -0.6570, 0.2794, 0.9589])

spline = CubicSpline(x, f_x)

def func_to_solve(x):
    return spline(x) - 1.8 * x**2

e = 0.0001

print(f"f(-1) = {func_to_solve(-1)}")
print(f"f(-0.5) = {func_to_solve(-0.5)}")

def mybisect(f, a, b):
  if f(a) * f(b) > 0:
    return
  f_a = f(a)
  while(abs(b - a) > e):
    c = (a + b) / 2
    f_c = f(c)
    if (f_a <= 0 and f_c <= 0) or (f_a >= 0 and f_c >= 0):
      a = c
    else:
      b = c
  return c
root = mybisect(func_to_solve, -1, -0.5)

print(f"Найденный корень: {root}")

xx = np.linspace(-1, -0.5, 100)
yy_spline = spline(xx)
yy_quad = 1.8 * xx**2

plt.plot(xx, yy_spline, label="Cubic Spline", color="blue")
plt.plot(xx, yy_quad, label="1.8 * x^2", color="red")
plt.axhline(0, color="black", linestyle="--")
plt.scatter([root], [spline(root)], color="green", label="Найденный корень", zorder=3)

plt.legend()
plt.show()
