import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import solve_ivp
from scipy.optimize import bisect
from scipy.optimize import brentq

def myfunc(x):
  return np.exp(-x) / (x**3)

def system(x, Y):
  y, z = Y
  dydx = z
  dzdx = (y**2) - 1
  return [dydx, dzdx]

def shooting_func(primary_y):
  solution = solve_ivp(system, [0,1], [0, primary_y],method='RK45', t_eval=[1])
  return solution.y[0][-1] - 1.01

result, error = quad(myfunc, 0.1, 0.2)
A = 1.2
B = 0.046
B *= result

primary_y = brentq(shooting_func, A, B, xtol=1e-10, rtol=1e-10)

sol = solve_ivp(system, [0,1], [0, primary_y],method='RK45', t_eval=np.linspace(0,1,100))

df_rkf45 = pd.DataFrame({"t": sol.t, "x1 (RKF45)": sol.y[0]})

print("Таблица значений для метода RKF45 при y(1) = 1.01:")
print(df_rkf45.to_string(index=False))

sol_high_precision = solve_ivp(system, [0, 1], [0, primary_y], method='LSODA', t_eval=np.linspace(0, 1, 100))

err = np.abs(sol.y[0] - sol_high_precision.y[0])
max_err = np.max(err)
max_err += error
print(f"max error:{max_err:.6f}")

plt.plot(sol.t, sol.y[0], label="RKF45")
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()