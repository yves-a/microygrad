class Value:
    def __init__(self, data, _children=(), op=None, label=None):
        self.data = data
        self._prev = set(_children)
        self._op = op
        self._backward = lambda: None
        self.grad = 0.0
        self.label = label

    def __repr__(self):
        return f"Value({self.data})"

    def __neg__(self):
        return self * -1

    def __mul__(self, other):
        other = Value(other, label=other) if not isinstance(other, Value) else other

        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __add__(self, other):
        other = Value(other, label=other) if not isinstance(other, Value) else other

        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, exponent):
        out = Value(self.data**exponent, (self,), f"^{exponent}")

        def _backward():
            self.grad += (exponent * (self.data ** (exponent - 1))) * out.grad

        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * (other ** (-1))

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "e")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        nodes = []
        visited = set()

        def topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topo(child)
                nodes.append(v)

        topo(self)
        self.grad = 1.0

        for n in reversed(nodes):
            n._backward()
