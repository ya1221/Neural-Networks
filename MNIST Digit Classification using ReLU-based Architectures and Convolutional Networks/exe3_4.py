import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch import nn
from torchvision import datasets
import torchvision.transforms as transforms
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 50
data_dir = '/files/'


def prepare_dataloaders(batch_size):
    image_normalize = lambda x: x / 255.

    train_loader = torch.utils.data.DataLoader(datasets.MNIST(data_dir,
                                                              train=True,
                                                              download=True,
                                                              transform=transforms.Compose([transforms.ToTensor(),
                                                                                            transforms.Lambda(
                                                                                                image_normalize)])),
                                               batch_size=batch_size,
                                               shuffle=True)

    train_size = int(0.9 * len(train_loader.dataset))  # 90% ל-train
    val_size = len(train_loader.dataset) - train_size  # 10% ל-validation

    train_dataset, val_dataset = random_split(train_loader.dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST(data_dir,
                                                             train=False,
                                                             download=True,
                                                             transform=transforms.Compose([transforms.ToTensor(),
                                                                                           transforms.Lambda(
                                                                                               image_normalize)])),
                                              batch_size=batch_size,
                                              shuffle=True)

    return train_loader, val_loader, test_loader


def init_conv2d_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        m.weight = nn.Parameter(torch.abs(m.weight))
        m.bias.data.fill_(0.01)


lossF = torch.nn.NLLLoss()
num_epochs = 100
epsilon = 1e-8
num_classes = 10
batch_limit = 13000  # Maximum number of mini-batches to process
target_accuracy = 0.99


def count_weights_and_bias(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validation_accuracy(model, val_loader):
    predicted, t = predict(model, val_loader)
    return (t == predicted).sum().float() / len(t)


def train_model(model, num_model, train_loader, val_loader):
    start = time.time()
    start5 = 0
    num_iter5 = 0
    if num_model >= 4:
        val = True  # found validation accuracy = 0.99
        maxV = 0
        start5 = time.time()
    mini_batches_count = 0
    for epoch in range(num_epochs):
        # print("Epoch ", epoch + 1, ":")
        for images, labels in train_loader:
            # Move data to the correct device
            images, labels = images.to(device), labels.to(device)
            # Calculate loss
            loss = lossF(model(images), labels)
            # Zero gradients before running the backward pass
            optimizer.zero_grad()
            # Compute gradients
            loss.backward()
            # Update weights
            optimizer.step()

            if num_model >= 4 and val:
                num_iter5 += 1
                model.eval()
                with torch.no_grad():
                    c = validation_accuracy(model, val_loader).item()
                    if c > maxV:
                        maxV = c
                    if c >= target_accuracy:
                        print(
                            f"Reached {target_accuracy} accuracy: Iteration {num_iter5}, Time: {time.time() - start5}")
                        val = False

            # Update the mini-batch counter
            mini_batches_count += 1
            # Stop training after processing 13,000 mini-batches
            if mini_batches_count >= batch_limit:
                print("Reached the batch limit of 13,000. Stopping training.")
                break
        if mini_batches_count >= batch_limit:
            break
    if num_model >= 4 and val:
        print(f"Validation did not reach the value {target_accuracy}, maximum validation value: {maxV}")
    return time.time() - start


def score(Y, T):
    # Initialize confusion matrix components
    tp = torch.zeros(num_classes, device=Y.device)
    fp = torch.zeros(num_classes, device=Y.device)
    fn = torch.zeros(num_classes, device=Y.device)
    tn = torch.zeros(num_classes, device=Y.device)

    # Compute confusion matrix
    for i in range(num_classes):
        tp[i] = torch.sum((T == i) & (Y == i)).float()
        fp[i] = torch.sum((T != i) & (Y == i)).float()
        fn[i] = torch.sum((T == i) & (Y != i)).float()
        tn[i] = torch.sum((T != i) & (Y != i)).float()

    # Calculate metrics
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    accuracy = (tp + tn) / (tp + fp + fn + tn + epsilon)
    balanced_accuracy = 0.5 * (recall + (tn / (tn + fp + epsilon)))
    mean_balanced_accuracy = balanced_accuracy.mean()

    # Print metrics for each class
    for i in range(num_classes):
        print(
            f"Class {i} - Accuracy: {accuracy[i]:.3f}, Precision: {precision[i]:.3f}, Recall: {recall[i]:.3f}, F1-Score: {f1_score[i]:.3f}, Balanced Accuracy: {balanced_accuracy[i]:.3f}")

    print(f"Mean Balanced Accuracy: {mean_balanced_accuracy:.3f}")
    return mean_balanced_accuracy


def predict(model, loader):
    model.eval()  # Set the model to evaluation mode
    Y = []
    T = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            Y.append(torch.argmax(model(images.to(device)), dim=1))
            T.append(labels)
    Y = torch.cat(Y)
    T = torch.cat(T)
    return Y, T


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def image_conv(model, batch, layer, channel, after_activation):
    model[0].register_forward_hook(get_activation('conv1'))
    model[1].register_forward_hook(get_activation('conv1_Relu'))
    model[3].register_forward_hook(get_activation('conv2'))
    model[4].register_forward_hook(get_activation('conv2_Relu'))
    model(batch)
    plt.figure()
    title = dict[layer + (after_activation == True)]
    plt.title(title)
    plt.imshow(activation[title][0][channel].cpu())
    plt.show()


def plot_heatmaps_by_groups(filters, layer_name):
    group_size = 8
    num_filters = filters.shape[0]
    num_groups = (num_filters + group_size - 1) // group_size

    for group_idx in range(num_groups):
        plt.figure(figsize=(12, 8))
        start_idx = group_idx * group_size
        end_idx = min((group_idx + 1) * group_size, num_filters)
        rows = (group_size + 2) // 3
        cols = 3
        for i in range(start_idx, end_idx):
            filter_weights = filters[i, 0].cpu()
            ax = plt.subplot(rows, cols, i - start_idx + 1)
            im = ax.imshow(filter_weights, cmap="coolwarm", interpolation="nearest")
            plt.title(f"Filter {i + 1}", fontsize=10)
            plt.axis('off')
        cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])
        plt.colorbar(im, cax=cbar_ax)
        plt.suptitle(f"{layer_name} Filters Group {group_idx + 1}", fontsize=16, y=0.95)
        plt.tight_layout(rect=[0, 0, 0.9, 0.9])
        plt.show()


def plot_histograms_by_groups(filters, layer_name):
    group_size = 8
    num_filters = filters.shape[0]
    num_groups = (num_filters + group_size - 1) // group_size

    for group_idx in range(num_groups):
        plt.figure(figsize=(12, 8))
        start_idx = group_idx * group_size
        end_idx = min((group_idx + 1) * group_size, num_filters)
        rows = (group_size + 2) // 3
        cols = 3
        for i in range(start_idx, end_idx):
            filter_weights = filters[i, 0].detach().cpu().numpy().flatten()
            plt.subplot(rows, cols, i - start_idx + 1)
            plt.hist(filter_weights, bins=30, color='blue', alpha=0.7)
            plt.title(f"Filter {i + 1}", fontsize=10)
            plt.xlabel("Weight Value", fontsize=8)
            plt.ylabel("Frequency", fontsize=8)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
        plt.suptitle(f"{layer_name} Filters Group {group_idx + 1}", fontsize=16, y=1)
        plt.subplots_adjust(hspace=0.5, wspace=0.4)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def plot_statistics_by_groups(filters, layer_name):
    group_size = 8
    num_filters = filters.shape[0]
    num_groups = (num_filters + group_size - 1) // group_size

    for group_idx in range(num_groups):
        plt.figure(figsize=(14, 10))
        start_idx = group_idx * group_size
        end_idx = min((group_idx + 1) * group_size, num_filters)
        rows = (group_size + 2) // 3
        cols = 3
        for i in range(start_idx, end_idx):
            filter_weights = filters[i, 0].detach().cpu().numpy().flatten()
            mean = np.mean(filter_weights)
            std = np.std(filter_weights)
            max_val = np.max(filter_weights)
            min_val = np.min(filter_weights)
            stats = [mean, std, max_val, min_val]
            labels = ["Mean", "Std", "Max", "Min"]

            ax = plt.subplot(rows, cols, i - start_idx + 1)
            ax.bar(labels, stats, color=["blue", "orange", "green", "red"], alpha=0.7)
            plt.title(f"Filter {i + 1}", fontsize=10)
            plt.ylabel("Value", fontsize=8)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
        plt.suptitle(f"{layer_name} Filters Group {group_idx + 1}", fontsize=16, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(hspace=0.5, wspace=0.4)
        plt.show()


def plot_lineplots_by_groups(filters, layer_name):
    group_size = 8
    num_filters = filters.shape[0]
    num_groups = (num_filters + group_size - 1) // group_size

    for group_idx in range(num_groups):
        plt.figure(figsize=(12, 8))
        start_idx = group_idx * group_size
        end_idx = min((group_idx + 1) * group_size, num_filters)
        rows = (group_size + 2) // 3
        cols = 3
        for i in range(start_idx, end_idx):
            filter_weights = filters[i, 0].cpu().flatten()
            x = np.arange(len(filter_weights))
            ax = plt.subplot(rows, cols, i - start_idx + 1)
            ax.plot(x, filter_weights, color="blue", alpha=0.7, linewidth=1)
            plt.title(f"Filter {i + 1}", fontsize=10)
            plt.xlabel("Index", fontsize=8)
            plt.ylabel("Weight Value", fontsize=8)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
        plt.suptitle(f"{layer_name} Filters Group {group_idx + 1}", fontsize=16, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(hspace=0.5, wspace=0.4)
        plt.show()


def plot_weights_as_table(filters, layer_name):
    group_size = 8
    num_filters = filters.shape[0]
    num_groups = (num_filters + group_size - 1) // group_size

    for group_idx in range(num_groups):
        rows = 4
        plt.figure(figsize=(15, 5))
        start_idx = group_idx * group_size
        end_idx = min((group_idx + 1) * group_size, num_filters)

        for i in range(start_idx, end_idx):
            ax = plt.subplot(rows, 3, i - start_idx + 1)
            filter_weights = filters[i].detach().cpu().numpy()
            ax.axis('off')

            table_data = np.round(filter_weights[0, :, :], 2)
            ax.table(
                cellText=table_data,
                loc='center',
                cellLoc='center',
                colWidths=[0.1] * table_data.shape[1]
            )
            ax.set_title(f"Filter {i + 1}", fontsize=10)

        plt.suptitle(f"{layer_name} Filters Group {group_idx + 1}", fontsize=16, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.subplots_adjust(hspace=0.5, wspace=0.4)
        plt.show()


# models
model1 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10),
    nn.LogSoftmax(dim=1)
)

model2 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 200),
    nn.ReLU(),
    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Linear(200, 10),
    nn.LogSoftmax(dim=1)
)

model3 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Flatten(),
    nn.Linear(in_features=32 * 14 * 14, out_features=1024),
    nn.ReLU(),
    nn.Linear(in_features=1024, out_features=10),
    nn.LogSoftmax(dim=1)
)

model4 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Flatten(),
    nn.Linear(in_features=64 * 7 * 7, out_features=1024),
    nn.ReLU(),
    nn.Linear(in_features=1024, out_features=10),
    nn.LogSoftmax(dim=1)
)

Dropout_rate = 0.5
model5 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Flatten(),
    nn.Dropout(p=Dropout_rate),
    nn.Linear(in_features=64 * 7 * 7, out_features=1024),
    nn.ReLU(),
    nn.Linear(in_features=1024, out_features=10),
    nn.LogSoftmax(dim=1)
)

models = [model1, model2, model3, model4, model5, copy.deepcopy(model5)]
batch_size5 = 100
lr = 1e-4
means_balanced_accuracy = []
test_loaders = []
train_loader, val_loader, test_loader = prepare_dataloaders(batch_size)
test_loaders.append(test_loader)

for i in models[2:]:
    i.apply(init_conv2d_weights)

for num, (model) in enumerate(models):
    print(f"Model number: {num + 1}\n")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if num == 5:
        train_loader, val_loader, test_loader = prepare_dataloaders(batch_size5)
        test_loaders.append(test_loader)
        print(f"Batch size: {batch_size5}")
    else:
        print(f"Batch size: {batch_size}")

    print(
        f"Time model: {train_model(model, num, train_loader, val_loader)}, Count Weights: {count_weights_and_bias(model)}")
    Y_T = [[0, 0], [0, 0], [0, 0]]
    for i, (loder) in enumerate([train_loader, val_loader, test_loader]):
        Y_T[i][0], Y_T[i][1] = predict(model, loder)

    # Print results for all datasets
    for i in [["train", Y_T[0][0], Y_T[0][1]], ["val", Y_T[1][0], Y_T[1][1]], ["test", Y_T[2][0], Y_T[2][1]]]:
        print(f"\n{i[0]} - Accuracy per class:")
        value = score(i[1], i[2])
        if i[0] == "test" and num >= 2:
            means_balanced_accuracy.append(value)

dict = {0: 'conv1', 1: 'conv1_Relu', 3: 'conv2', 4: 'conv2_Relu'}
index_model = 2 + means_balanced_accuracy.index(max(means_balanced_accuracy))
model_high_balanced_accuracy = models[index_model]
print(f"The index of the model with the highest balanced accuracy is: {index_model + 1}")
num_channel = 4
model_high_balanced_accuracy = model_high_balanced_accuracy.to(device)

layer = [0]
if index_model > 2:
    layer.append(3)
for i in layer:
    f = model_high_balanced_accuracy[i].weight.data
    plot_lineplots_by_groups(f, layer_name=dict[i])
    plot_heatmaps_by_groups(f, layer_name=dict[i])
    plot_weights_as_table(f, layer_name=dict[i])
    plot_histograms_by_groups(f, layer_name=dict[i])
    plot_statistics_by_groups(f, layer_name=dict[i])

with torch.no_grad():
    for count, (images, _) in enumerate(test_loaders[0 if index_model < 5 else 1]):
        if count == 4:
            break
        images = images.to(device)
        image_conv(model_high_balanced_accuracy, images, 0, num_channel, False)
        image_conv(model_high_balanced_accuracy, images, 0, num_channel, True)
        if index_model > 2:
            image_conv(model_high_balanced_accuracy, images, 3, num_channel, False)
            image_conv(model_high_balanced_accuracy, images, 3, num_channel, True)
        plt.figure()
        plt.title("Source image")
        plt.imshow(images[0].squeeze(0).cpu().numpy())
        plt.show()