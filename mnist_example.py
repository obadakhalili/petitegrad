from petitegrad import Tensor
from mnist import MNIST
import numpy as np

np.random.seed(42)


class PetiteNet:
    def __init__(self, input_size, output_size):
        h1_size = 100
        h2_size = 50

        self.w1 = Tensor(np.random.randn(input_size, h1_size), requires_grad=True)
        self.b1 = Tensor(np.random.rand(h1_size), requires_grad=True)
        self.w2 = Tensor(np.random.randn(h1_size, h2_size), requires_grad=True)
        self.b2 = Tensor(np.random.rand(h2_size), requires_grad=True)
        self.w3 = Tensor(np.random.randn(h2_size, output_size), requires_grad=True)
        self.b3 = Tensor(np.random.rand(output_size), requires_grad=True)

        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def __call__(self, X):
        assert isinstance(X, Tensor)
        assert X.data.ndim == 2
        assert X.data.shape[1] == self.w1.data.shape[0]

        # TODO: using any non-linearity other than sigmoid makes the score plummet!
        return (
            X.mm(self.w1)
            .add(self.b1)
            .sigmoid()
            .mm(self.w2)
            .add(self.b2)
            .sigmoid()
            .mm(self.w3)
            .add(self.b3)
            .sigmoid()
        )


def fetch_mnist():
    mndata = MNIST("./data/mnist")
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

    input_size = X_train.shape[1]
    output_size = 10
    net = PetiteNet(input_size, output_size)

    batch_size = 64
    for epoch in range(1, 51):
        batches_indices = np.random.permutation(len(X_train))

        # TODO: using only a sample in each epoch should work like in other frameworks, without having to use the full dataset to get a good score
        for batch_indices in [
            batches_indices[batch_idx : batch_idx + batch_size]
            for batch_idx in range(0, len(batches_indices), batch_size)
        ]:
            X_batch = Tensor(X_train[batch_indices])
            y_batch = Tensor(y_train[batch_indices])

            output = net(X_batch)
            loss = output.mse(y_batch)

            loss.backward()

            for p in net.params:
                # TODO: using the below syntax to update the data isn't good
                p.data -= 0.01 * p.grad
                p.reset_grad()

        predictions = net(X_test).data
        accuracy = (predictions.argmax(axis=1) == y_test.argmax(axis=1)).mean()
        print(f"Epoch {epoch}: accuracy = {accuracy}")
