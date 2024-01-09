from petitegrad.tensor import Tensor
from mnist import MNIST
import numpy as np


class PetiteNet:
    def __init__(self, input_size, output_size):
        h1_size = 10
        h2_size = 64

        self.w1 = Tensor(np.random.randn(input_size, h1_size))
        self.b1 = Tensor(np.random.rand(h1_size))
        self.w2 = Tensor(np.random.randn(h1_size, h2_size))
        self.b2 = Tensor(np.random.rand(h2_size))
        self.w3 = Tensor(np.random.randn(h2_size, output_size))
        self.b3 = Tensor(np.random.rand(output_size))

        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def forward(self, X):
        assert isinstance(X, Tensor)
        assert X.data.ndim == 2

        return (
            X.matmul(self.w1)
            .add(self.b1)
            .relu()
            .matmul(self.w2)
            .add(self.b2)
            .relu()
            .matmul(self.w3)
            .add(self.b3)
            .sigmoid()
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

    X_test = Tensor(X_test)

    input_size = X_train.data.shape[1]
    output_size = 10
    net = PetiteNet(input_size, output_size)

    batch_size = 64
    for epoch in range(1000):
        batches_indices = np.arange(len(X_train))
        np.random.shuffle(batches_indices)

        for batch_indices in [
            batches_indices[batch_idx : batch_idx + batch_size]
            for batch_idx in range(0, len(batches_indices), batch_size)
        ]:
            X_batch = Tensor(X_train[batch_indices])
            y_batch = Tensor(y_train[batch_indices])

            output = net.forward(X_batch)
            loss = output.mean_squared_error(y_batch)

            loss.backward()

            for p in net.params:
                p.data -= 0.01 * p.grad

        if epoch % 10 == 0:
            predictions = net.forward(X_test)
            accuracy = (predictions.data.argmax(axis=1) == y_test.argmax(axis=1)).mean()
            print(f"Epoch {epoch}: accuracy = {accuracy}")
