import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    for i in range(5000):
        print('step: ', i)
        rand_x = np.random.uniform(0.0, 1.0)
        arc_angle = 70
        radius = 1.6
        radius_cos = radius * np.cos(np.deg2rad(arc_angle / 2))
        if rand_x < radius - radius_cos:
            ymax = np.sqrt(radius ** 2 - (rand_x - radius) ** 2)
        else:
            ymax = (radius - rand_x) * np.tan(np.deg2rad(arc_angle / 2))
        rand_y = np.random.uniform(-ymax, ymax)
        plt.scatter(rand_x, rand_y)
    plt.xlim(-0.5, 1.8)
    plt.ylim(-1.5, 1.5)
    plt.show()
