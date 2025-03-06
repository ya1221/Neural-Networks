import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

dim = 2
num_hidden = [1, 2, 4]
out_dim = 1
l_rate = [0.01, 0.1]
bypass = [True, False]
num_epocs = 40000
MinLossVal = 0.0001
MaxLossVal = 0.2
MaxSuccRun = 10

x_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=True, dtype=torch.float32)
x_val = torch.cat(
    (x_train, torch.tensor([[1, 0.1], [1, 0.9], [0.9, 0.9], [0.1, 0.9]], requires_grad=True, dtype=torch.float32)),
    dim=0)
t_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
t_val = torch.cat((t_train, torch.tensor([[1], [0], [0], [1]], requires_grad=True, dtype=torch.float32)), dim=0)


class BTU(torch.nn.Module):
    def __init__(self, T=0.2, inplace: bool = False):
        super(BTU, self).__init__()
        self.T = T

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-input / self.T))


class XOR_Net_Model(nn.Module):
    def __init__(self, num_hidden, bypass=True):
        super().__init__()
        self.bypass = bypass
        self.hidden = nn.Linear(dim, num_hidden)
        if self.bypass:
            self.output = nn.Linear(num_hidden + dim, out_dim)
        else:
            self.output = nn.Linear(num_hidden, out_dim)
        self.BTU = BTU(0.5)

    def forward(self, input, get):
        z1 = self.hidden(input)
        y1 = self.BTU(z1)
        if get:
            return y1
        if self.bypass:
            y1_concat = torch.cat((input, y1), 1)
            z2 = self.output(y1_concat)
        else:
            z2 = self.output(y1)
        return self.BTU(z2)


def Loss(out, t_train):
    return -torch.sum(t_train * torch.log(out) + (1.0 - t_train) * torch.log(1.0 - out)) / out.size()[
        0]  # Cross Entropy loss function


def train(model, x_train, t_train, optimizer):
    y_pred = model(x_train, False)
    loss = Loss(y_pred, t_train)

    # zero gradients berfore running the backward pass
    optimizer.zero_grad()

    # backward pass to compute the gradient of loss
    # backprop + accumulate
    loss.backward()

    # update params
    optimizer.step()
    return loss


def check_val(loss, arr_loss, numEp):
    arr_loss.append(loss)
    if numEp >= 10:
        arr_loss.pop(0)
    if loss < MaxLossVal and (arr_loss[0] - loss) < MinLossVal:
        return True
    return False


def make_values(l_rate, num_hidden, bypass):
    values = []
    for i in l_rate:
        for j in num_hidden:
            for b in bypass:
                values.append([i, j, b])
    values.append([0.01, 1, True])  # exper 4
    return values


def one_run(arr_one_exper, model, optimizer):
    arr_loss = []
    for i in range(num_epocs):
        train_loss = train(model, x_train, t_train, optimizer)
        val_loss = Loss(model(x_val, False), t_val)
        if check_val(val_loss, arr_loss, i + 1):
            arr_one_exper[0].append(i + 1)
            arr_one_exper[1].append(val_loss.item())
            arr_one_exper[2].append(train_loss.item())
            return 1
    return 0


def one_exper(values):
    fail_run = 0
    succ_run = 0
    arr_one_exper = [[], [], []]  # [[nums of epocs], [val losses], [train losses]]
    while succ_run != MaxSuccRun:
        model = XOR_Net_Model(values[1], values[2])
        optimizer = torch.optim.SGD(model.parameters(), lr=values[0])
        if one_run(arr_one_exper, model, optimizer):
            succ_run += 1
        else:
            fail_run += 1
        if values[1] == 1:
            print("Truth Table (section 6) - run number " + str(succ_run + fail_run) + ":\n x1   x2   Output")
            results = (torch.cat((x_train, model(x_train, True)), dim=1)).tolist()
            for row in results:
                print(row)  # print each row as a array
    arr_one_exper.append(fail_run)
    return arr_one_exper


def all_exper():
    arr_all_exper = []
    values = make_values(l_rate, num_hidden[1:], bypass)
    for i in values:
        arr_one_exper = one_exper(i)
        arr_all_exper.append([[np.mean(arr_one_exper[0]), np.std(arr_one_exper[0])],
                              [np.mean(arr_one_exper[2]), np.std(arr_one_exper[2])],
                              [np.mean(arr_one_exper[1]), np.std(arr_one_exper[1])], arr_one_exper[3]])
    print_all(values, arr_all_exper)
    make_graph(num_hidden, meanEpocsNumHidden(values, arr_all_exper), "Num Hidden", "Mean Epocs",
               "Mean Epocs Num Hidden")
    make_graph(bypass, meanEpocsIsBridge(values, arr_all_exper), "Bypass", "Mean Epocs", "Mean Epocs Is Bridge")
    make_graph(l_rate, stdEpocsLearningRate(values, arr_all_exper), "Learning Rate", "Std Epocs",
               "Std Epocs Learning Rate")


def print_all(values, arr_all_exper):
    for i in range(len(values)):
        print("Experiment number " + str(i) + " - Learning rate: " + str(values[i][0]) + ", Hidden: " + str(
            values[i][1]) + ", Bypass: " + str(values[i][2]) +
              "\nMean epochs: " + str(arr_all_exper[i][0][0]) + ", Standard deviation epochs: " + str(
            arr_all_exper[i][0][1]) +
              "\nMean train loss: " + str(arr_all_exper[i][1][0]) + ", Standard deviation train loss: " + str(
            arr_all_exper[i][1][1]) +
              "\nMean validation loss: " + str(arr_all_exper[i][2][0]) + ", Standard deviation validation loss: " + str(
            arr_all_exper[i][2][1]) +
              "\nNum of fail runs: " + str(arr_all_exper[i][3]) + "\n\n")


def meanEpocsNumHidden(values, arr_all_exper):
    mean_epocs_num_hidden = [0, 0, 0]
    countNH = [0, 0, 0]
    for i in range(len(arr_all_exper)):
        mean_epocs_num_hidden[num_hidden.index(values[i][1])] += arr_all_exper[i][0][0]
        countNH[num_hidden.index(values[i][1])] += 1
    for s in range(len(mean_epocs_num_hidden)):
        mean_epocs_num_hidden[s] /= countNH[s]
    return mean_epocs_num_hidden


def meanEpocsIsBridge(values, arr_all_exper):
    mean_epocs_is_bridge = [0, 0]
    countB = [0, 0, 0]
    for i in range(len(arr_all_exper)):
        mean_epocs_is_bridge[bypass.index(values[i][2])] += arr_all_exper[i][0][0]
        countB[bypass.index(values[i][2])] += 1
    for s in range(len(mean_epocs_is_bridge)):
        mean_epocs_is_bridge[s] /= countB[s]
    return mean_epocs_is_bridge


def stdEpocsLearningRate(values, arr_all_exper):
    std_epocs_learning_rate = [0, 0]
    countB = [0, 0, 0]
    for i in range(len(arr_all_exper)):
        std_epocs_learning_rate[l_rate.index(values[i][0])] += arr_all_exper[i][0][0] ** 2
        countB[l_rate.index(values[i][0])] += 1
    for s in range(len(std_epocs_learning_rate)):
        std_epocs_learning_rate[s] /= countB[s]
        std_epocs_learning_rate[s] **= 0.5
    return std_epocs_learning_rate


def make_graph(x, y, nameX, nameY, nameGraph):
    plt.plot(x, y, marker='o', color='b', label='Line Plot')
    # Add labels and title
    plt.xlabel(nameX)
    plt.ylabel(nameY)
    plt.title(nameGraph)
    plt.legend()  # Add a legend
    plt.show()  # Display the plot


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    all_exper()


if __name__ == "__main__":
    main()