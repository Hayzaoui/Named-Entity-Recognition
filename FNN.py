from gensim import downloader
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch import save

GLOVE_PATH = 'glove-twitter-200'


class FNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) # Linear function 1: 212 --> 100
        self.relu1 = nn.LeakyReLU() # Non-linearity 1

        self.fc2 = nn.Linear(hidden_dim, output_dim) # Linear function 2: 100 --> 2
        self.relu2 = nn.LeakyReLU() # Non-linearity 2

        # add here we did sort of encoder. Now times to decode and then to predict

        self.fc3 = nn.Linear(output_dim, 100) # Linear function 3: 2 --> 100
        self.relu3 = nn.LeakyReLU() # Non-linearity 3

        self.fc4 = nn.Linear(100, 200) # Linear function 4: 100 --> 200
        self.relu4 = nn.LeakyReLU() # Non-linearity 4

        # now we finish the sort of decoder and we encode to predict

        self.fc5 = nn.Linear(200, 100) # Linear function 5: 200 --> 100
        self.relu5 = nn.LeakyReLU()  # Non-linearity 5

        self.fc6 = nn.Linear(100, output_dim) # Linear function 6: 100 --> 2
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.fc1(x)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        out = self.relu3(out)

        out = self.fc4(out)
        out = self.relu4(out)

        out = self.fc5(out)
        out = self.relu5(out)

        out = self.sigmoid(self.fc6(out))

        return out


def train(model, train_loader, test_loader, num_epochs, device, optimizer, criterion, num_labels):
    iter = 0
    TP, TN, FP, FN = 0.0, 0.0, 0.0, 0.0
    f1_score = 0
    for epoch in range(num_epochs):
        for i, (words, labels) in enumerate(train_loader):
            # Load images with gradient accumulation capabilities
            words = words.view(-1, 212).requires_grad_().to(device)
            labels = torch.nn.functional.one_hot(labels.to(torch.int64), 2)

            labels = labels.float()
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(words)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            iter += 1

            if iter % 2000 == 0:
                model.eval()
                correct = 0
                total = 0
                for words, labels in test_loader:
                    words = words.view(-1, 212).requires_grad_().to(device)

                    with torch.no_grad():
                        outputs = model(words)
                    _, predicted = torch.max(outputs.data, 1)

                    total += labels.size(0)
                    if torch.cuda.is_available():

                        for i in range(len(predicted)):
                            if predicted.cpu()[i] == 1 and labels[i] == 1:
                                TP += 1
                            if predicted.cpu()[i] == 0 and labels.cpu()[i] == 1:
                                FN += 1
                            if predicted.cpu()[i] == 1 and labels.cpu()[i] == 0:
                                FP += 1
                            if predicted.cpu()[i] == 0 and labels.cpu()[i] == 0:
                                TN += 1
                    else:
                        for i in range(len(predicted)):
                            if predicted[i] == 1 and labels[i] == 1:
                                TP += 1
                            if predicted[i] == 0 and labels[i] == 1:
                                FN += 1
                            if predicted[i] == 1 and labels[i] == 0:
                                FP += 1
                            if predicted[i] == 0 and labels[i] == 0:
                                TN += 1

                recall = float(TP / (TP + FN))
                precision = float(TP / (TP + FP))
                f1_score = 2 * precision * recall / (precision + recall)
    return f1_score


def predict(model, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data_loaders = {"train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True),
    # "test": DataLoader(data_sets["test"], batch_size=batch_size, shuffle=False)}
    labels, preds = [], []
    model.to(device)
    model.eval()
    print(data_loader)
    for words in data_loader:
        words = words[0].view(-1, 212).requires_grad_().to(device)
        with torch.no_grad():
            outputs = model(words)

        _, predicted = torch.max(outputs.data, 1)
        preds += predicted.view(-1).tolist()

    return preds

def test(model, data_sets):
    TP, TN, FP, FN = 0.0, 0.0, 0.0, 0.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labels, preds = [], []
    model.to(device)
    model.eval()
    for words, labs in data_sets:
        words = words.view(-1, 212).requires_grad_().to(device)
        with torch.no_grad():
            outputs = model(words)

        pred = outputs.argmax(dim=-1).clone().detach().cpu()
        labels += labs.cpu().view(-1).tolist()
        preds += pred.view(-1).tolist()
        for i in range(len(preds)):
            if preds[i] != 0 and labels[i] != 0:
                TP += 1
            if preds[i] == 0 and labels[i] != 0:
                FN += 1
            if preds[i] != 0 and labels[i] == 0:
                FP += 1
            if preds[i] == 0 and labels[i] == 0:
                TN += 1

    recall = float(TP / (TP + FN))
    precision = float(TP / (TP + FP))
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score


def model2(vecs, vecs_dev, tags_dict, tags, tags_dev, tags_dict_dev):
    batch_size = 300
    n_iters = 10000

    input_dim = 212
    output_dim = 2
    hidden_dim = 100
    learning_rate = 0.3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FNN(input_dim, hidden_dim, output_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.to(device)

    tensor_x_train = torch.Tensor(np.array(vecs))  # transform to torch tensor
    tensor_y_train = torch.Tensor(np.array(tags))

    tensor_x_test = torch.Tensor(np.array(vecs_dev))  # transform to torch tensor
    tensor_y_test = torch.Tensor(np.array(tags_dev))


    train_dataset = TensorDataset(tensor_x_train, tensor_y_train)  # create your datset
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)  # create your dataloader

    test_dataset = TensorDataset(tensor_x_test, tensor_y_test)  # create your datset
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)  # create your dataloader
    num_epochs = n_iters / (len(train_dataset) / batch_size)
    num_epochs = int(num_epochs)


    f1_score_test_on_train = train(model, train_loader, test_loader, num_epochs, device, optimizer, criterion,
                                   2)
    f1_score_test = test(model, test_loader)
    save(model, "model2.plk")
    print(f1_score_test)
