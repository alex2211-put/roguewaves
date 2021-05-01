# KdV 1D
# Based on https://wikiwaves.org/Numerical_Solution_of_the_KdV

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# utils
def range_with_last(start, end, step):
    return np.arange(start, end + (step / 2), step)

sech = lambda x: 1 / np.cosh(x)


# params
N = 256
x = np.linspace(-10, 10, N)
delta_x = x[1] - x[0]
delta_k = 2 * np.pi / (N * delta_x)

k = np.concatenate((range_with_last(0, (N/2) * delta_k, delta_k),
                    range_with_last(-((N/2)-1) * delta_k, -delta_k, delta_k)))

delta_t = 1/(N**2)
tmax = 1
nmax = round(tmax/delta_t)


# initial condition
u1 = 8*np.square(sech(2*(x+8)))
u2 = 2*np.square(sech(x+1))
u = u1 + u2

# c = 16
# u = np.square(c * (sech(np.sqrt(c) / 2 * (x + 8))) / 2)


# solve
u_t = []
U = np.fft.fft(u)
for n in range(nmax):
    U1 = np.exp(1j * (k**3) * delta_t) * U
    U = U1 - delta_t * (3j * k * np.fft.fft((np.real(np.fft.ifft(U))**2)))
    u_t.append(np.real(np.fft.ifft(U)))


# plot
fig = plt.figure()
ax = plt.axes()
line, = ax.plot([], [])


def init():
    global line
    line.set_data(x, u_t[0])
    return line,


def animate(i):
    global x, u_t, line
    line.set_data(x, u_t[i])
    return line,


plt.xlim(-10, 10)
plt.ylim(0, 80)
u_t = u_t[::100]
animate(1)
anim = animation.FuncAnimation(
    fig, animate, frames=len(u_t), interval=1, init_func=init)

plt.show()
