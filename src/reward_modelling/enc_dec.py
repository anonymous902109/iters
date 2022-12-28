from torch import nn
from torch.optim import RMSprop


class EncDecNet(nn.Module):

    def __init__(self, input_size, enc_size):
        super(EncDecNet, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, enc_size)
        self.fc3 = nn.Linear(enc_size, 128)
        self.fc4 = nn.Linear(128, input_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    def encode(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class EncoderDecoder():

    def __init__(self, input_size, enc_size):
        self.input_size = input_size
        self.enc_size = enc_size

        self.net = EncDecNet(input_size, enc_size)

        self.criterion = nn.MSELoss()
        self.optimizer = RMSprop(self.net.parameters(), lr=0.001)

    def train(self, train_dataloader, test_dataloader):
        print('Training trajectory encoder...')
        for i in range(20):
            total_loss = 0.0
            for x, y in train_dataloader:
                self.optimizer.zero_grad()

                output = self.net.float()(x.float())

                loss = self.criterion(output.float(), x.float())  # difference to input

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print('Epoch = {}. Training loss = {}'.format(i, total_loss/len(train_dataloader)))

        self.evaluate(test_dataloader)

    def encode(self, x):
        self.net.eval()
        # return self.net.float().encode(x.float())
        return x

    def evaluate(self, dataloader):
        self.net.eval()

        total_loss = 0.0

        for x, y in dataloader:
            output = self.net.float()(x.float())
            loss = self.criterion(x.float(), output.float())

            total_loss += loss.item()

        print('Mean squared loss on test dataset = {}'.format(total_loss/len(dataloader)))