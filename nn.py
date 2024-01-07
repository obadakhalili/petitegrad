from petitegrad.tensor import Tensor
from mnist import MNIST
import numpy as np

# np.random.seed(1337)


class PetiteNet:
    def __init__(self, input_size, output_size):
        h1_size = 128
        h2_size = 64
        self.w1 = Tensor(np.random.randn(input_size, h1_size))
        self.b1 = Tensor(np.random.randn(h1_size))
        self.w2 = Tensor(np.random.randn(h1_size, h2_size))
        self.b2 = Tensor(np.random.randn(h2_size))
        self.w3 = Tensor(np.random.randn(h2_size, output_size))
        self.b3 = Tensor(np.random.randn(output_size))

    def forward(self, X):
        return (
            X.mul(self.w1)
            .add(self.b1)
            .relu()
            .mul(self.w2)
            .add(self.b2)
            .relu()
            .mul(self.w3)
            .add(self.b3)
        )


def fetch_mnist():
    mndata = MNIST("./data")
    mndata.gz = True
    mndata.load_training()
    mndata.load_testing()

    X_train = np.array(mndata.train_images)
    y_train = np.eye(10)[mndata.train_labels]
    X_test = np.array(mndata.test_images)
    y_test = np.eye(10)[mndata.test_labels]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    print("Fetching MNIST data...")
    X_train, y_train, X_test, y_test = fetch_mnist()

    input_size = X_train.data.shape[1]
    output_size = 10

    batch_size = 64
    X_batch = Tensor(X_train[:batch_size])
    y_batch = Tensor(y_train[:batch_size])

    net = PetiteNet(input_size, output_size)
    output = net.forward(X_batch)
    loss = output.categorical_crossentropy(y_batch)
