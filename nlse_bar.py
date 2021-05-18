import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# params
N = 2048

#k = 2500*0.001
delta_t = 0.01

height = np.loadtxt('height_m.dat')[:536]
period = np.loadtxt('period_m.dat')[:536]

result = []
counter = 0
for h, l in zip(height, period):
    k = 2 * np.pi * 0.001 / l
    x = np.linspace(-1000, 1000, N)
    # initial condition
    psi = np.ones(2048)*h
    psi_t = []
    m = 0
    # solver
    for i in range(N):
        r1 = np.random.uniform(-1, 1, 10).T
        r2 = np.random.uniform(-1, 1, 10).T
        a = 0.2*r1 + 0.2*r2
        #print(np.abs(psi) ** 2 - x * np.linalg.norm(a))
        psi_linear = np.exp(1j*delta_t*(np.abs(psi) ** 2 - x * np.linalg.norm(a)))*psi
        psi = np.fft.ifft((psi_linear + (np.exp(-1j * k ** 2 * delta_t / 2) * np.fft.fft(psi_linear))))
        psi_t.append(np.real(psi))
        mux = max(np.real(psi))
        if mux > m:
            m = mux
    counter += 1
    print(counter)
    result.append(m)
print(sum(result)/len(result))
weights = np.ones_like(result)/float(len(result))
plt.hist(result, bins=25, weights=weights)
plt.xlabel('Максимальная высота возникающей волны, м')
plt.show()