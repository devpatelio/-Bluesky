import data_preprocessing
import CNN_implementation

class LSTM_audio(torch.nn.Module):
    def __init__ (self, audio_dim, hidden_dim):
        super(LSTM_audio, self).__init__()
        self.lstm = nn.LSTM(audio_dim, hidden_dim, batch_first=True) #, bidirectional=True x2 shape
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(hidden_dim, audio_dim)
    def forward(self, x):
        x = self.dropout(x)
#         print((x.view(1, len(x), -1)).shape)
        print(((x.view(1, len(x), -1)).view(-1, 20)).shape)
        lstm_out, (h_t, c_t) = self.lstm((x.view(1, len(x), -1))) ##lstm output
        model_out = self.output(((x.view(1, len(x), -1)).view(-1, 20))) ##linear layer -> 20 values -> 1 hot vector encoding
        output_pred = F.log_softmax(model_out, dim=0) ##softmax
        return output_pred, (h_t, c_t)




import torch.optim as optim
AUDIO_DIM = 20
HIDDEN_DIM = 20
lstm_model = LSTM_audio(AUDIO_DIM, HIDDEN_DIM)
optimizer_lstm = optim.SGD(lstm_model.parameters(), lr=0.1)
loss_lstm = nn.MSELoss()

running_loss_lstm = []

epochs = 50
for epoch in range(0, epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        lstm_model.zero_grad()
        lstm_model.train()
        inputs, labels = data
#         print(inputs.shape)
        optimizer_lstm.zero_grad()
        outputs, states = lstm_model(inputs)
        print(outputs[1])
        print(labels[1].shape)
        labels = labels.type(torch.FloatTensor)
        losses = nn.MSELoss()(outputs, labels)
        losses += losses.item()
        running_loss_lstm.append(losses.item())
        loss_lstm.backward()
        optimizer_lstm.step()
        
    print(f"Epoch: {epoch} Training loss: {running_loss/1}")  


print('Finished Training')