import numpy as np


class Tensor:
    def __init__(self, nparray):
        assert isinstance(nparray, np.ndarray)
        self.data = nparray

    def transpose(self):
        return Tensor(self.data.transpose())

    def mul(self, tensor):
        assert isinstance(tensor, Tensor)
        return Tensor(self.data @ tensor.data)

    def add(self, tensor):
        assert isinstance(tensor, Tensor)
        return Tensor(self.data + tensor.data)

    def relu(self):
        return Tensor(np.maximum(self.data, 0))

    def sigmoid(self):
        return Tensor(1 / (1 + np.exp(-self.data)))

    def softmax(self):
        exps = np.exp(self.data)
        return Tensor(exps / np.sum(exps, axis=1, keepdims=True))

    def categorical_crossentropy(self, tensor):
        assert isinstance(tensor, Tensor)
        probs = self.softmax()
        return Tensor(
            -np.sum(tensor.data * np.log(probs.data), axis=1).mean(keepdims=True)
        )

    def backward(self):
        pass
