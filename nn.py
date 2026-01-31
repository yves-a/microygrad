class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):

    def __init__(self, num_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((x_i * w_i for x_i, w_i in zip(x, self.w)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):

    def __init__(self, num_inputs, num_outputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_outputs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        outs = []
        for n in self.neurons:
            outs.extend(n.parameters())
        return outs


class MLP(Module):

    def __init__(self, num_inputs, num_outputs):
        sz = [num_inputs] + num_outputs
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(num_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        outs = []
        for layer in self.layers:
            outs.extend(layer.parameters())
        return outs
