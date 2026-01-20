
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.signal import argrelextrema

# -------- Function 1D --------
def f1(x):
    return np.sin(5*x)*np.exp(-x**2) - 0.1*x**3 + np.cos(2*x)

x = np.linspace(-4, 4, 4000)
y = f1(x)

mins = argrelextrema(y, np.less)[0]
maxs = argrelextrema(y, np.greater)[0]

gmin_idx = np.argmin(y)
gmax_idx = np.argmax(y)

plt.figure(figsize=(10,6))
plt.plot(x, y, label="f(x)")
plt.scatter(x[mins], y[mins], c="blue", marker="v", label="Local minima")
plt.scatter(x[maxs], y[maxs], c="orange", marker="^", label="Local maxima")
plt.scatter(x[gmin_idx], y[gmin_idx], c="red", marker="*", s=200, label="Global min")
plt.scatter(x[gmax_idx], y[gmax_idx], c="green", marker="*", s=200, label="Global max")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Function of one variable with extrema")
plt.legend()
plt.grid()
plt.savefig("outputs_f1.png")
plt.close()

# -------- Global optimization comparison --------
methods = {}
res_de = differential_evolution(f1, [(-4,4)])
methods["Differential Evolution"] = res_de.fun

# -------- Function 2D --------
def f2(v):
    x, y = v
    return np.sin(x) + 2*np.sin(y)

res2 = minimize(f2, x0=[0,0], bounds=[(-2*np.pi,2*np.pi),(-2*np.pi,2*np.pi)])
xmin, ymin = res2.x

X, Y = np.meshgrid(np.linspace(-2*np.pi,2*np.pi,200),
                   np.linspace(-2*np.pi,2*np.pi,200))
Z = np.sin(X) + 2*np.sin(Y)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9)
ax.scatter(xmin, ymin, f2([xmin,ymin]), c="red", s=60)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x,y)")
ax.set_title("3D surface with minimum")
plt.savefig("outputs_f2_surface.png")
plt.close()

print("Lab 7 finished successfully.")
