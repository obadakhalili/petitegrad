import numpy as np


class Tensor:
    def __init__(self, data, grad_fn=None, src=None):
        assert isinstance(data, np.ndarray)
        assert grad_fn is None or callable(grad_fn)
        assert src is None or isinstance(src, list)
        if src is not None:
            assert all(isinstance(t, Tensor) for t in src)

        self.data = data
        self.grad = None
        self._grad_fn = grad_fn
        self._src = src

    def matmul(self, t):
        assert isinstance(t, Tensor)
        assert self.data.ndim == t.data.ndim == 2
        assert self.data.shape[1] == t.data.shape[0]

        def matmul_grad():
            # TODO: not sure if this is correct (dividing by batch size). I mean abstraction wise
            self.grad = (out.grad @ t.data.T) / self.data.shape[0]
            t.grad = (self.data.T @ out.grad) / self.data.shape[0]

        out = Tensor(self.data @ t.data, grad_fn=matmul_grad, src=[self, t])

        return out

    # NOTE: consider whether operands should be validated
    def add(self, t):
        assert isinstance(t, Tensor)

        def add_grad():
            self.grad = np.ones_like(self.data) * out.grad
            t.grad = np.ones_like(t.data) * out.grad

            # TODO: not sure this is correct. I mean abstraction wise
            if self.data.shape != t.data.shape:
                if self.data.ndim > t.data.ndim:
                    t.grad = t.grad.sum(axis=0)
                else:
                    self.grad = self.grad.sum(axis=0)

        out = Tensor(self.data + t.data, grad_fn=add_grad, src=[self, t])

        return out

    def relu(self):
        def relu_grad():
            self.grad = (self.data > 0) * out.grad

        out = Tensor(np.maximum(self.data, 0), grad_fn=relu_grad, src=[self])

        return out

    def sigmoid(self):
        def sigmoid_grad():
            self.grad = out.grad * out.data * (1 - out.data)

        out = Tensor(1 / (1 + np.exp(-self.data)), grad_fn=sigmoid_grad, src=[self])

        return out

    # NOTE: consider whether operands should be validated
    def mean_squared_error(self, y):
        assert isinstance(y, Tensor)

        def mse_grad():
            self.grad = (self.data - y.data) * out.grad
            y.grad = (y.data - self.data) * out.grad

        out = Tensor(
            (np.square(self.data - y.data) / 2).sum(axis=1).mean(keepdims=True),
            grad_fn=mse_grad,
            src=[self, y],
        )

        return out

    def backward(self):
        if self._grad_fn is None:
            return

        if self.grad is None:
            self.grad = 1

        self._grad_fn()

        if self._src is not None:
            for t in self._src:
                t.backward()
