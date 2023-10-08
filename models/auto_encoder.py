import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


class SimpleAutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, input_dim),
            nn.Sigmoid()  # MinMax scaling 된 데이터를 재구성하기 위함
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AutoEncoderModel(object):

    def __init__(self, model):
        self.model = model
        pass

    def train(self, X):
        # PyTorch용 데이터 로더 생성
        tensor_X = torch.Tensor(X)
        dataset = TensorDataset(tensor_X, tensor_X)  # AutoEncoder는 입력 == 출력이므로
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        autoencoder = self.model
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

        epochs = 100
        for epoch in range(epochs):
            for batch_x, _ in loader:  # _ 는 사용되지 않는 target입니다.
                outputs = autoencoder(batch_x)
                loss = criterion(outputs, batch_x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        return autoencoder
