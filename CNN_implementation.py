import data_preprocessing
import LSTM_implementation

class Net(nn.Module):
    def __init__ (self):
        super(Net, self).__init__() 
        self.fc1 = nn.Linear(20, 300)
        self.fc2 = nn.Linear(300, 700)
        self.fc3 = nn.Linear(700, 500)
        self.fc4 = nn.Linear(500, 300)
        self.fc5 = nn.Linear(300, 200)
        self.fc6 = nn.Linear(200, 100)
        self.output = nn.Linear(100, 20)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.output(x)
        x = self.softmax(x)
        return x
net = Net()   

import torch.optim as optim
optimizer_nn = optim.SGD(net.parameters(), lr=0.001)
criterion_nn = nn.MSELoss()
losses_nn = []

epochs = 50
for epoch in range(0, epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer_nn.zero_grad()
        outputs = net(inputs)
        labels = labels.type(torch.FloatTensor)
        loss = criterion_nn(outputs, labels)
        loss.backward()
        optimizer_nn.step()

        running_loss += loss.item()
        losses_nn.append(loss.item())
        
    print(f"Epoch: {epoch} Training loss: {running_loss/1}")  


print('Finished Training')







base_model = label_encoder_dict.copy()
# print(basemodel)
acc_dict = {}
for i in categories:
    key = i
    acc_dict.setdefault(key, [base_model[key].tolist()])



correct_count, all_count = 0, 0
output_probabilities = []

##STILL NEED TO MAKE CONFUSION MATRIX GRID HERE
    

for i, data in enumerate(trainloader, 0):
    inputs, labels = iter(data)
    with torch.no_grad():
        probs = net(inputs.float())
    ps = torch.exp(probs)
    ps = ps.tolist()
    labels = labels.tolist()
    output_probabilities.append(ps)
    m_pred = max(ps)
    m_true = max(labels)
    if (ps.index(m_pred) == labels.index(m_true)):
        correct_count += 1
    all_count += 1
    
print(correct_count/all_count * 100, '%')



print(acc_dict)


plt.plot(losses)
plt.title('loss vs epochs')
plt.xlabel('epochs')
plt.ylabel('loss')

keys = []
values = []
for key, value in label_encoder_dict.items():
    keys.append(key)


nb_classes = 20
confusion_matrix = np.zeros((nb_classes, nb_classes))
for i, (inputs, classes) in enumerate(testloader):
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        for t in classes.view(-1):
            for p in preds.view(-1):
                confusion_matrix[t.long(), p.long()] += 1

                
import seaborn as sns

plt.figure(figsize=(15,10))

class_names = keys
df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
plt.ylabel('True label')
plt.xlabel('Predicted label')
("")



