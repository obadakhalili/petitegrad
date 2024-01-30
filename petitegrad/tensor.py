import numpy as np


class Tensor:
    def __init__(self, data, grad_fn=None, src=None):
        if isinstance(data, (int, float)):
            data = np.array(data, dtype=np.float32)

        if isinstance(data, np.ScalarType) and (
            np.issubdtype(data.dtype, np.integer)
            or np.issubdtype(data.dtype, np.floating)
        ):
            data = np.array(data, dtype=np.float32)

        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32)

        assert isinstance(data, np.ndarray) and np.issubdtype(
            data.dtype, np.floating
        ), "data must be a scalar or numpy array of floats or integers"
        assert grad_fn is None or callable(grad_fn), "grad_fn must be callable"
        assert src is None or isinstance(src, list), "src must be a list of tensors"
        if src is not None:
            assert all(
                isinstance(t, Tensor) for t in src
            ), "src must be a list of tensors"

        self._data = data
        self._grad = np.zeros_like(data, dtype=data.dtype)
        self._grad_fn = grad_fn
        self._src = src

    @property
    def data(self):
        return self._data

    @property
    def grad(self):
        return self._grad

    def zero_grad(self):
        self._grad = np.zeros_like(self._data, dtype=self._data.dtype)

    # TODO: support higher-order tensors in `.dot` method
    # TODO: implement a single `.dot` method that supports all cases allowed by numpy, and remove `.mv` and `.mm`. simiar to tinygrad's `.dot` method

    def dot(self, t):
        assert isinstance(t, Tensor), "dot requires a tensor"
        assert self.data.ndim == 1 and t.data.ndim == 1, "dot requires 1D tensors"

        def grad_fn():
            self._grad += t.data * out.grad
            t._grad += self.data * out.grad

        out = Tensor(np.dot(self.data, t.data), grad_fn=grad_fn, src=[self, t])

        return out

    def mv(self, t):
        assert isinstance(t, Tensor), "mv requires a tensor"
        assert (self.data.ndim == 1 and t.data.ndim == 2) or (
            self.data.ndim == 2 and t.data.ndim == 1
        ), "mv requires 1D and 2D tensors"

        def grad_fn():
            mat, vec = (self, t) if self.data.ndim == 2 else (t, self)

            # TODO: figure out a more straightforward way to find the gradients
            if self is mat:
                mat._grad += vec.data * out.grad.reshape(-1, 1)
            else:
                mat._grad += vec.data.reshape(-1, 1) * out.grad
            vec._grad += np.dot(out.grad, mat.data if mat is self else mat.data.T)

        out = Tensor(np.dot(self.data, t.data), grad_fn=grad_fn, src=[self, t])

        return out

    def mm(self, t):
        assert isinstance(t, Tensor), "mm requires a tensor"
        assert self.data.ndim == 2 and t.data.ndim == 2, "mm requires 2D tensors"

        def grad_fn():
            self._grad += np.matmul(out.grad, t.data.T)
            t._grad += np.matmul(self.data.T, out.grad)

        out = Tensor(np.matmul(self.data, t.data), grad_fn=grad_fn, src=[self, t])

        return out

    def mul(self, t):
        if isinstance(t, (int, float)):
            t = Tensor(t)

        assert isinstance(t, Tensor), "mul requires a tensor or scalar"

        def grad_fn():
            a, b = (self, t) if self.data.ndim >= t.data.ndim else (t, self)

            a._grad += b.data * out.grad
            b._grad += (a.data * out.grad).sum(
                axis=tuple(i for i in range(a.data.ndim - b.data.ndim))
            )

        out = Tensor(np.multiply(self.data, t.data), grad_fn=grad_fn, src=[self, t])

        return out

    def add(self, t):
        if isinstance(t, (int, float)):
            t = Tensor(t)

        assert isinstance(t, Tensor), "add requires a tensor or scalar"

        def grad_fn():
            a, b = (self, t) if self.data.ndim >= t.data.ndim else (t, self)

            a._grad += out.grad
            b._grad += out.grad.sum(
                axis=tuple(i for i in range(a.data.ndim - b.data.ndim))
            )

        out = Tensor(np.add(self.data, t.data), grad_fn=grad_fn, src=[self, t])

        return out

    def relu(self):
        def grad_fn():
            self._grad += (self.data > 0) * out.grad

        out = Tensor(np.maximum(self.data, 0), grad_fn=grad_fn, src=[self])

        return out

    def sigmoid(self):
        def grad_fn():
            self._grad += (1 - out.data) * out.data * out.grad

        out = Tensor(1 / (1 + np.exp(-self.data)), grad_fn=grad_fn, src=[self])

        return out

    def tanh(self):
        def grad_fn():
            self._grad += (1 - out.data**2) * out.grad

        out = Tensor(np.tanh(self.data), grad_fn=grad_fn, src=[self])

        return out

    def mse(self, t):
        assert isinstance(t, Tensor), "mse requires a tensor"

        # TODO: this shouldn't be necessary. such methods should be implemented using tensor methods (such as `Tensor.sub`, and `Tensor.square`) instead of numpy methods,
        # which will take care of broadcasting and their gradients.
        # the same is true for other higher-order tensor methods
        assert self.data.shape == t.data.shape, "mse requires tensors of the same shape"

        def grad_fn():
            self._grad += (self.data - t.data) * out.grad
            t._grad += (t.data - self.data) * out.grad

        out = Tensor(
            np.square(self.data - t.data).mean(), grad_fn=grad_fn, src=[self, t]
        )

        return out

    # TODO: should support numpy arguments, such as `axis`
    def sum(self):
        def grad_fn():
            self._grad += np.ones_like(self._data) * out._grad

        out = Tensor(np.sum(self._data), grad_fn=grad_fn, src=[self])

        return out

    def backward(self):
        assert self.data.shape == (), "backward only supported for scalar outputs"

        def backward(t):
            if t._grad_fn is None:
                return

            t._grad_fn()

            if t._src is not None:
                for t in t._src:
                    backward(t)

        self._grad = np.ones_like(self._data)

        backward(self)
