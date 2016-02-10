from ler import LER
from mpl_toolkits.mplot3d import Axes3D
from optparse import OptionParser
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt


def demo(k):
    X, t = make_swiss_roll(noise=1)

    le = SpectralEmbedding(n_components=2, n_neighbors=k)
    le_X = le.fit_transform(X)

    ler = LER(n_components=2, n_neighbors=k, affinity='rbf')
    ler_X = ler.fit_transform(X, t)

    _, axes = plt.subplots(nrows=1, ncols=3, figsize=plt.figaspect(0.33))
    axes[0].set_axis_off()
    axes[0] = plt.subplot(131, projection='3d')
    axes[0].scatter(*X.T, c=t, s=50)
    axes[0].set_title('Swiss Roll')
    axes[1].scatter(*le_X.T, c=t, s=50)
    axes[1].set_title('LE Embedding')
    axes[2].scatter(*ler_X.T, c=t, s=50)
    axes[2].set_title('LER Embedding')
    plt.show()


if __name__ == "__main__":
    op = OptionParser()
    op.add_option('--n_neighbors', type=int, metavar='k', default=10,
                  help='# of neighbors for LE & LER [7]')
    opts, args = op.parse_args()
    demo(opts.n_neighbors)
