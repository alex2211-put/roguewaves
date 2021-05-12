import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# params
N = 2048
x = np.linspace(-10, 10, N)
k = 10 ** 8
delta_t = 0.05

# initial condition
psi0_1 = 1
psi0_2 = 0
psi = psi0_1 + 1j*psi0_2
r1 = np.random.uniform(-1, 1, 10).T
r2 = np.random.uniform(-1, 1, 10).T
a1 = 0.2 * r1
a2 = 0.2 * r2
a = a1 + a2


# solver
psi_t = []
for n in range(N):
    psi = np.fft.ifft(np.exp(-1j * k ** 2 * delta_t/2) *
                      np.fft.fft(np.exp(1j*delta_t*(psi ** 2 - x*np.linalg.norm(a)))*psi))
    psi_t.append(np.absolute(psi))


# plot
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
plt.ylim(0, 5)
animate(1)
anim = animation.FuncAnimation(
    fig, animate, frames=len(psi_t), interval=1, init_func=init)

plt.show()
