import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# params
N = 2048
x = np.linspace(-1000, 1000, N)
k = 2500*0.001
delta_t = 0.01

# initial condition
psi = np.ones(2048)
psi_t = []
m = 0
# solver
for i in range(N):
    r1 = np.random.uniform(-1, 1, 10).T
    r2 = np.random.uniform(-1, 1, 10).T
    a = 0.2*r1 + 0.2*r2
    psi_linear = np.exp(1j*delta_t*(np.abs(psi) ** 2 - x * np.linalg.norm(a)))*psi
    psi = np.fft.ifft((psi_linear + (np.exp(-1j * k ** 2 * delta_t / 2) * np.fft.fft(psi_linear))))
    psi_t.append(np.real(psi))
    mux = max(np.real(psi))
    if mux > m:
        m = mux
print(m)
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


plt.xlim(-500, 500)
plt.ylim(-10, 10)
animate(1)
anim = animation.FuncAnimation(
    fig, animate, frames=len(psi_t), interval=1, init_func=init)

plt.show()