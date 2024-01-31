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

    def add(self, t):
        if isinstance(t, (int, float)):
            t = Tensor(t)

        assert isinstance(t, Tensor), "add requires a tensor or scalar"

        def grad_fn():
            self._grad += np.sum(
                out.grad,
                axis=tuple(
                    i
                    for i, axis in enumerate(
                        (1,) * max(t.data.ndim - self.data.ndim, 0) + self.data.shape
                    )
                    if axis == 1
                ),
            ).reshape(self.data.shape)

            t._grad += np.sum(
                out.grad,
                axis=tuple(
                    i
                    for i, axis in enumerate(
                        (1,) * max(self.data.ndim - t.data.ndim, 0) + t.data.shape
                    )
                    if axis == 1
                ),
            ).reshape(t.data.shape)

        out = Tensor(np.add(self.data, t.data), grad_fn=grad_fn, src=[self, t])

        return out

    def mul(self, t):
        if isinstance(t, (int, float)):
            t = Tensor(t)

        assert isinstance(t, Tensor), "mul requires a tensor or scalar"

        def grad_fn():
            self._grad += np.sum(
                t.data * out.grad,
                axis=tuple(
                    i
                    for i, axis in enumerate(
                        (1,) * max(t.data.ndim - self.data.ndim, 0) + self.data.shape
                    )
                    if axis == 1
                ),
            ).reshape(self.data.shape)

            t._grad += np.sum(
                self.data * out.grad,
                axis=tuple(
                    i
                    for i, axis in enumerate(
                        (1,) * max(self.data.ndim - t.data.ndim, 0) + t.data.shape
                    )
                    if axis == 1
                ),
            ).reshape(t.data.shape)

        out = Tensor(np.multiply(self.data, t.data), grad_fn=grad_fn, src=[self, t])

        return out

    def sum(self, axis=None):
        assert axis is None or isinstance(
            axis, (int, tuple)
        ), "axis must be an integer or tuple of integers"

        if axis is None:
            axis = tuple(range(self.data.ndim))
        elif isinstance(axis, int):
            axis = (axis,)

        # TODO: support negative axes
        assert all(
            isinstance(i, int) and 0 <= i < self.data.ndim for i in axis
        ), "axes must be a tuple of integers between 0 and the number of dimensions of the tensor"

        def grad_fn():
            out_grad = out.grad.reshape(
                tuple(
                    1 if i in axis else self.data.shape[i]
                    for i in range(self.data.ndim)
                )
            )
            self._grad += np.ones_like(self._data) * out_grad

        out = Tensor(np.sum(self._data, axis=axis), grad_fn=grad_fn, src=[self])

        return out

    def reshape(self, shape):
        assert isinstance(shape, tuple), "shape must be a tuple"

        def grad_fn():
            self._grad += out.grad.reshape(self.data.shape)

        out = Tensor(self.data.reshape(shape), grad_fn=grad_fn, src=[self])

        return out

    # TODO: implement a single `.dot` method that implements vector-matrix multiplication,
    #       replaces `.mv`, `.mm`, and vector dot product (similar to tinygrad's `.dot` method),
    #       and supports matrix multiplication for >2D tensors

    def dot(self, t):
        assert isinstance(t, Tensor), "dot requires a tensor"
        assert self.data.ndim == 1 and t.data.ndim == 1, "dot requires 1D tensors"
        assert self.data.shape == t.data.shape, "dot requires tensors of the same shape"

        return self.mul(t).sum()

    def mv(self, t):
        assert isinstance(t, Tensor), "mv requires a tensor"
        assert (
            self.data.ndim == 2 and t.data.ndim == 1
        ), "mv requires a matrix and vector"
        assert self.data.shape[1] == t.data.shape[0], "mv requires compatible shapes"

        return self.mul(t).sum(axis=1)

    def mm(self, t):
        assert isinstance(t, Tensor), "mm requires a tensor"
        assert self.data.ndim == 2 and t.data.ndim == 2, "mm requires 2D tensors"
        assert self.data.shape[1] == t.data.shape[0], "mm requires compatible shapes"

        a = self.reshape((self.data.shape[0], 1, self.data.shape[1]))
        b = t.reshape((1, t.data.shape[1], t.data.shape[0]))

        return a.mul(b).sum(axis=2)

    # TODO: `.sigmoid` and `.mse` should be implemented using tensor methods instead of numpy methods

    def sigmoid(self):
        def grad_fn():
            self._grad += (1 - out.data) * out.data * out.grad

        out = Tensor(1 / (1 + np.exp(-self.data)), grad_fn=grad_fn, src=[self])

        return out

    def mse(self, t):
        assert isinstance(t, Tensor), "mse requires a tensor"
        # TODO: this shouldn't be necessary when such ops are implemented using tensor methods
        assert self.data.shape == t.data.shape, "mse requires tensors of the same shape"

        def grad_fn():
            self._grad += (self.data - t.data) * out.grad
            t._grad += (t.data - self.data) * out.grad

        out = Tensor(
            np.square(self.data - t.data).mean(), grad_fn=grad_fn, src=[self, t]
        )

        return out
