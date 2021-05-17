import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random as rnd

def range_with_last(start, end, step):
    return np.arange(start, end + (step / 2), step)

# params
N = 256//2
x = np.linspace(-10, 10, N)
delta_t = 1/(N**2)
t_max = 10

delta_x = x[1] - x[0]
delta_k = 2 * np.pi / (N * delta_x)
k = 1


# initial condition
height = np.loadtxt('data/north_atlantic/height_m.dat')[:100]
period = np.loadtxt('data/north_atlantic/period_m.dat')[:100]

result = []
u = np.ones_like(x)
for h, l in zip(height, period):
    phi = 2 * rnd.random() * np.pi
    k1 = 2 * np.pi * 0.001 / l
    u += h * np.sin(k1*x + phi)

u += abs(min(u))
u = abs(u)
u = u/max(u)
u = u * 2
u += 10

psi = u

# solver
psi_t = []

for n in range(int(t_max//delta_t)):
    a = 0.01
    psi = np.fft.ifft(np.exp(-1j * k ** 2 * delta_t/2) *
                        np.fft.fft(np.exp(1j*delta_t*(psi ** 2 - x*np.linalg.norm(a)))*psi))
    psi_t.append(np.absolute(psi))
result.append(max(abs(psi)))

fig = plt.figure()
ax = plt.axes()
line, = ax.plot([], [])

def init():
    global line
    line.set_data(x, psi_t[0])
    return line,


def animate(i):
    global x, psi_t, line
    line.set_data(x, psi_t[i])
    return line,


plt.xlim(-10, 10)
plt.ylim(0, 16)
animate(1)
anim = animation.FuncAnimation(
    fig, animate, frames=len(psi_t), interval=1, init_func=init)

plt.show()
