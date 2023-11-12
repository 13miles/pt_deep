import torch
import torch.nn as nn
import string
import random

total_epochs = 1000
chunk_len = 10

all_characters = string.printable
n_characters = len(all_characters)

hidden_size = 100
batch_size = 1
num_layers = 1
embedding_size = 70
learning_rate = 0.002

sentence = ("if you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea.")

def random_chunk():
    start_index = random.randint(0, len(sentence) - chunk_len)
    end_index = start_index + chunk_len + 1
    return sentence[start_index : end_index]

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return tensor

def random_train_set():
    chunk = random_chunk()
    input = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return input, target

class RNNet(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.out_size = output_size
        self.num_layers = num_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.rnn = nn.GRU(self.embedding_size, self.hidden_size, self.num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden

    def forward(self, input, hidden):
        x = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(x, hidden)
        y = self.fc(output.view(batch_size, -1))
        return y, hidden

model = RNNet(input_size=n_characters,
              embedding_size=embedding_size,
              hidden_size=hidden_size,
              output_size=n_characters,
              num_layers=num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

def test_func():
    start_str = 'b'
    input = char_tensor(start_str)
    hidden = model.init_hidden()
    generated_text = []

    for i in range(200):
        output, hidden = model(input, hidden)

        out_dist = output.data.view(-1).div(0.8).exp()
        top_i = torch.multinomial(out_dist, 1)[0]
        predicted_char = all_characters[top_i]
        generated_text.append(predicted_char)
        input = char_tensor(predicted_char)

    return ''.join(generated_text)

for i in range(total_epochs):
    input, label = random_train_set()
    loss = torch.tensor([0]).type(torch.FloatTensor)
    hidden = model.init_hidden()
    optimizer.zero_grad()
    generated_text = []

    for j in range(chunk_len - 1):
        x_train = input[j]
        y_train = label[j].unsqueeze(0).type(torch.LongTensor)
        hypothesis, hidden = model(x_train, hidden)
        loss += loss_func(hypothesis, y_train)

        out_dist = hypothesis.data.view(-1).div(0.8).exp()
        top_i = torch.multinomial(out_dist, 1)[0]
        predicted_char = all_characters[top_i]
        generated_text.append(predicted_char)

    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        generated_text = ''.join(generated_text)
        print(f'epoch [{i}/{total_epochs}], loss: {loss.item():.4f}')
        print(f'Generated Text: {generated_text}')

result_text = test_func()
print(result_text)
