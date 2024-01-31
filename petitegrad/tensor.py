import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False, grad_fn=None, src=None):
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
        assert isinstance(requires_grad, bool), "requires_grad must be a boolean"
        assert grad_fn is None or callable(grad_fn), "grad_fn must be callable"
        assert src is None or isinstance(src, list), "src must be a list of tensors"
        if src is not None:
            assert all(
                isinstance(t, Tensor) for t in src
            ), "src must be a list of tensors"

        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = grad_fn
        self._src = src

    def reset_grad(self):
        self.grad = None

    def backward(self):
        assert self.data.shape == (), "backward only supported for scalar outputs"

        def backward(t, t_grad):
            if t.requires_grad:
                if t.grad is None:
                    t.grad = t_grad
                else:
                    t.grad += t_grad

            if t._grad_fn is None:
                return

            src_grads = t._grad_fn(t_grad)

            if t._src is not None:
                for t, t_grad in zip(t._src, src_grads):
                    backward(t, t_grad)

        self_grad = np.ones_like(self.data)

        backward(self, self_grad)

    def add(self, t):
        if isinstance(t, (int, float)):
            t = Tensor(t)

        assert isinstance(t, Tensor), "add requires a tensor or scalar"

        def grad_fn(out_grad):
            self_grad = np.sum(
                out_grad,
                axis=tuple(
                    i
                    for i, axis in enumerate(
                        (1,) * max(t.data.ndim - self.data.ndim, 0) + self.data.shape
                    )
                    if axis == 1
                ),
            ).reshape(self.data.shape)

            t_grad = np.sum(
                out_grad,
                axis=tuple(
                    i
                    for i, axis in enumerate(
                        (1,) * max(self.data.ndim - t.data.ndim, 0) + t.data.shape
                    )
                    if axis == 1
                ),
            ).reshape(t.data.shape)

            return [self_grad, t_grad]

        out = Tensor(np.add(self.data, t.data), grad_fn=grad_fn, src=[self, t])

        return out

    def mul(self, t):
        if isinstance(t, (int, float)):
            t = Tensor(t)

        assert isinstance(t, Tensor), "mul requires a tensor or scalar"

        def grad_fn(out_grad):
            self_grad = np.sum(
                t.data * out_grad,
                axis=tuple(
                    i
                    for i, axis in enumerate(
                        (1,) * max(t.data.ndim - self.data.ndim, 0) + self.data.shape
                    )
                    if axis == 1
                ),
            ).reshape(self.data.shape)

            t_grad = np.sum(
                self.data * out_grad,
                axis=tuple(
                    i
                    for i, axis in enumerate(
                        (1,) * max(self.data.ndim - t.data.ndim, 0) + t.data.shape
                    )
                    if axis == 1
                ),
            ).reshape(t.data.shape)

            return [self_grad, t_grad]

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

        def grad_fn(out_grad):
            out_grad_reshape = out_grad.reshape(
                tuple(
                    1 if i in axis else self.data.shape[i]
                    for i in range(self.data.ndim)
                )
            )
            self_grad = np.ones_like(self.data) * out_grad_reshape

            return [self_grad]

        out = Tensor(np.sum(self.data, axis=axis), grad_fn=grad_fn, src=[self])

        return out

    def reshape(self, shape):
        assert isinstance(shape, tuple), "shape must be a tuple"

        def grad_fn(out_grad):
            self_grad = out_grad.reshape(self.data.shape)
            return [self_grad]

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
        def grad_fn(out_grad):
            self_grad = (1 - out.data) * out.data * out_grad
            return [self_grad]

        out = Tensor(1 / (1 + np.exp(-self.data)), grad_fn=grad_fn, src=[self])

        return out

    def mse(self, t):
        assert isinstance(t, Tensor), "mse requires a tensor"
        # TODO: this shouldn't be necessary when such ops are implemented using tensor methods
        assert self.data.shape == t.data.shape, "mse requires tensors of the same shape"

        def grad_fn(out_grad):
            self_grad = (self.data - t.data) * out_grad
            t_grad = (t.data - self.data) * out_grad

            return [self_grad, t_grad]

        out = Tensor(
            np.square(self.data - t.data).mean(), grad_fn=grad_fn, src=[self, t]
        )

        return out
