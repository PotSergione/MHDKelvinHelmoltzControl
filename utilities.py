import numpy as np
import matplotlib.pyplot as plt
import time



def image_initialiser(rows=2, cols=1):
    fig, (ax, ax1)  = plt.subplots(nrows=rows, ncols=cols, figsize=(6,  5))
    ax1.set(xlim=[0, 10], ylim=[0, 1.1], xlabel='time [s]', ylabel='intensity')
    x = np.linspace(0, 10, 100)
    strat = ax1.plot(x[0], 0)[0]
    ax.set_title('ω')
    ax.axis('off')
    ax1.grid()

    return fig, ax, strat, x


def plotter(matrix, fig, ax, ax1, x, control):
    probes_z = np.linspace(0, 64, 16, endpoint=False).astype(int)
    probes_x = np.linspace(42, 86, 12, endpoint=True).astype(int)#[46, 50, 54, 58, 60, 62, 64, 66, 68, 72, 76, 80]
    index = np.ix_(probes_x, probes_z)
    X, Z = np.meshgrid(probes_x, probes_z)
    ax.scatter(X, Z, c='yellow', marker='o', s=15, label='probes')
    ax.imshow(matrix.T,  cmap='seismic', aspect='auto')
    ax.axis('off')
    ax1.set_xdata(x[:len(control)])
    ax1.set_ydata(control)

    fig.show()
    plt.pause(0.01)


def stats_printer(env, start):
    print("--------------------------------------------")
    print("------------------STATS---------------------")
    print("--------------------------------------------")
    print(f"    Total reward: {np.sum(env.rewards)}")
    print(f"    Total iterations: {env.iteration}")
    print(f"    Epsilon: {env.eps}")
    print(f"    Phase: {env.phase}")
    print(f"    Maximum control at time: {np.argmax(env.controls)}")
    print(f"    Maximum control: {np.max(env.controls)}")
    print(f"    Total  power control: {np.sum(env.controls)}")
    print(f"    Total time: {time.time() - start:.2f} seconds")
    print("--------------------------------------------")
    print("------------------DONE----------------------")
    print("--------------------------------------------")