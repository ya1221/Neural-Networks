import torch
from torch import nn

in_dim = 2  # Input dim
out_dim = 1  # Output dim
temp = 0.001  # for sigmoid


class BTU(torch.nn.Module):
    def __init__(self, T = temp):
        super(BTU, self).__init__()
        self.T = T

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-input / self.T))


class Network(nn.Module):
    def __init__(self, k, bypass):
        super().__init__()
        self.bypass = bypass
        self.hidden = nn.Linear(in_dim, k)
        if self.bypass:
            self.output = nn.Linear(k + in_dim, out_dim)
        else:
            self.output = nn.Linear(k, out_dim)
        self.BTU = BTU(temp)

    def set_weights(self, w, b, layer_name):
        if not hasattr(self, layer_name):
            raise ValueError(f"Layer '{layer_name}' does not exist in the model.")

        layer = getattr(self, layer_name)

        if w.shape != layer.weight.shape:
            raise ValueError(f"Shape mismatch for weights: expected {layer.weight.shape}, got {w.shape}")
        if b.shape != layer.bias.shape:
            raise ValueError(f"Shape mismatch for bias: expected {layer.bias.shape}, got {b.shape}")

        with torch.no_grad():
            layer.weight.copy_(w)
            layer.bias.copy_(b)

    def forward(self, input):
        z1 = self.hidden(input)
        y1 = self.BTU(z1)
        if self.bypass:
            y1_concat = torch.cat((input, y1), 1)
            z2 = self.output(y1_concat)
        else:
            z2 = self.output(y1)
        return self.BTU(z2)


def loss(x, y, model_format):
    squared_deltas = torch.square(model_format(x) - y)  ## SSE
    return torch.sum(squared_deltas).numpy()


def main():
    xor_train = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    while True:
        bypass = False
        work = True
        choose = input("Please choose the number of hidden neuron (1/2/4), 0 for end: ")
        if choose == "0":
            print("Have a nice day!")
            break
        elif choose == "1":
            bypass = True
            model = Network(int(choose), bypass)
            hidden_weights = torch.tensor([[1., 1.]])
            hidden_bias = torch.tensor([-1.5])
            output_weights = torch.tensor([[1., 1., -2.]])
            output_bias = torch.tensor([-0.5])
        elif choose == "2":
            model = Network(int(choose), bypass)
            hidden_weights = torch.tensor([[-1., -1.], [1., 1.]])
            hidden_bias = torch.tensor([1.5, -0.5])
            output_weights = torch.tensor([[1., 1.]])
            output_bias = torch.tensor([-1.5])
        elif choose == "4":
            model = Network(int(choose), bypass)
            hidden_weights = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
            hidden_bias = torch.tensor([0.5, -0.5, -0.5, -1.5])
            output_weights = torch.tensor([[0., 1., 1., 0.]])
            output_bias = torch.tensor([-0.5])
        else:
            print("Wrong input.\n")
            work = False

        if work:
            model.set_weights(hidden_weights, hidden_bias, 'hidden')
            model.set_weights(output_weights, output_bias, 'output')
            model(xor_train)

            for name, param in model.named_parameters():
                print(f"Layer: {name} -> Values: {param.data.numpy()}")

            t = torch.FloatTensor([[0], [1], [1], [0]])
            with torch.no_grad():
                print(f"\nThe loss is: {loss(xor_train, t, model)}")

            print("\nTruth Table:")
            for d in xor_train:
                print(f"Input (x): {d.numpy()} -> Output (y): {model(d.unsqueeze(0)).item():.4f}")
            print()


if __name__ == "__main__":
    main()
