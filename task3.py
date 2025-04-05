import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

h_int = 0.0075
eps = 0.0001
t_span = (0, 0.15)
y0 = [3, -1]

def system(t, y):
  x1, x2 = y
  dx1_dt = -430*x1 -12000*x2 + np.exp(-10 * t)
  dx2_dt = x1 + np.log(1 + 100 * t ** 2)
  return [dx1_dt, dx2_dt]

rkf45 = solve_ivp(system, t_span, y0, method="RK45", rtol = eps, atol = eps)

def advanced_Euler(system, y0, h, t_span):
  t0, t_end = t_span
  y_values = [y0]
  t_values = np.arange(t0, t_end + h, h)
  for i in range(len(t_values) - 1):
    t_n = t_values[i]
    y_n = y_values[-1]
    k1 = np.array(system(t_n, y_n))
    k2 = np.array(system(t_n + (h / 2), y_n + (h / 2) * k1))
    y_next = y_n + h * k2
    y_values.append(y_next)
  return t_values, np.array(y_values)

t_euler, y_euler = advanced_Euler(system, y0, h_int, t_span)

#table for x1,x2

df_rkf45 = pd.DataFrame({"t": rkf45.t, "x1 (RKF45)": rkf45.y[0], "x2 (RKF45)": rkf45.y[1]})

df_euler = pd.DataFrame({"t": t_euler, "x1 (Euler)": y_euler[:, 0], "x2 (Euler)": y_euler[:, 1]})

print("Таблица значений для метода RKF45:")
print(df_rkf45.to_string(index=False))

print("\nТаблица значений для метода Эйлера:")
print(df_euler.to_string(index=False))

A = np.array([[-430, -12000], [1, 0]])
eigenvalues = np.linalg.eigvals(A)
h_critical = 2 / np.max(np.abs(eigenvalues))

print(f"\nКритический шаг для устойчивости метода Эйлера: {h_critical:.6f}")

my_h = 0.0045
my_t_euler, my_y_euler = advanced_Euler(system, y0, my_h, t_span)

my_df_euler = pd.DataFrame({"t": my_t_euler, "x1 (myEuler)": my_y_euler[:, 0], "x2 (myEuler)": my_y_euler[:, 1]})
print("\nТаблица значений для метода Эйлера с моим шагом:")
print(my_df_euler.to_string(index=False))

plt.figure(figsize=(12, 6))

plt.subplot(2,2,1)
plt.plot(rkf45.t, rkf45.y[0], "--g", label="RKF45")
plt.plot(my_t_euler,my_y_euler[:,0], ":r", label="Euler")
plt.xlabel("t", fontsize=18, color='blue')
plt.ylabel("x1(t)", fontsize=18, color='pink')
plt.title(r"Plot for $\frac{dx_1} {dt}$", fontsize= 18)
plt.grid()
plt.legend()


plt.subplot(2,2,2)
plt.plot(my_t_euler, my_y_euler[:,1],"r:",label="Euler")
plt.plot(rkf45.t, rkf45.y[1],"g--", label="RKF45")
plt.xlabel("t", fontsize=18, color='blue')
plt.ylabel("x2(t)", fontsize=18, color='pink')
plt.title(r"Plot for $\frac{dx_2} {dt}$", fontsize= 18)
plt.legend()
plt.grid()

plt.show()