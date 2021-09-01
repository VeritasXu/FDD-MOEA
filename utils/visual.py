import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visual_2d(idx, line, label='IGD'):
    fig = plt.figure(figsize=(4, 4))
    plt.plot(idx, line, '-^', label = label)
    plt.show()


def visual_3d(found_pf, pf):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45, 45)

    # fp1, fp2, fp3 = zip(*[(a, b, c) for a, b, c in
    #                       zip(found_pf[:, 0], found_pf[:, 1], found_pf[:, 2])
    #                       if 0 < a < 1.2 and 0 < b < 1.2 and 0 < c < 1.2])
    fp1, fp2, fp3 = found_pf[:, 0], found_pf[:, 1], found_pf[:, 2]

    ax.scatter3D(fp1, fp2, fp3, cmap='Blues', label='found')
    ax.scatter3D(pf[:, 0], pf[:, 1], pf[:, 2], cmap='Reds', label='real')
    # ax.set_xlim(0, 1.1)
    # ax.set_ylim(0, 1.1)
    # ax.set_zlim(0, 1.1)
    plt.tight_layout()
    plt.legend()
    plt.show()
    # plt.savefig('./results/dtlz2.pdf', dpi=500)