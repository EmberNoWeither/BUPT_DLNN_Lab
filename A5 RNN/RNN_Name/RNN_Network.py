import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_of_all_kinds):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(num_of_all_kinds + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(num_of_all_kinds + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size).to('cuda')


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_of_all_kinds, LSTM_nums = 3, bidirectional=False) -> None:
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size+num_of_all_kinds, hidden_size, LSTM_nums, bidirectional=bidirectional)
        self.hidden_size = hidden_size
        self.lstm_layers = LSTM_nums
        self.bidirectional = bidirectional
        if bidirectional:
            self.__class__.__name__ = 'BiLSTM'
            self.liner_layers = nn.Linear(hidden_size * 2, output_size)
        else:
            self.liner_layers = nn.Linear(hidden_size, output_size)
            
        self.softmax = nn.LogSoftmax(dim=1)
            
    def initHidden(self):
        if self.bidirectional:
            return (torch.zeros(self.lstm_layers*2,  self.hidden_size).to('cuda'), torch.zeros(self.lstm_layers*2, self.hidden_size).to('cuda'))
        else: 
            return (torch.zeros(self.lstm_layers,  self.hidden_size).to('cuda'), torch.zeros(self.lstm_layers, self.hidden_size).to('cuda'))
    
    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input), 1)
        output, (h_n, c_n) = self.lstm(input_combined, hidden)
        output = self.liner_layers(output)
        output = self.softmax(output)
        
        return output, (h_n, c_n)
    
    

class Seq2Seq(nn.Module):
    def __init__(self, input_size, encoder_hidden_size, decoder_hidden_size, output_size, num_of_all_kinds, LSTM_nums = 2) -> None:
        super().__init__()
        
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.lstm_layers = LSTM_nums
        self.encoder = nn.LSTM(input_size+num_of_all_kinds, encoder_hidden_size, LSTM_nums, bidirectional=True)
        self.decoder = nn.LSTM(encoder_hidden_size*2 + input_size + num_of_all_kinds, decoder_hidden_size, LSTM_nums)
        self.linear_layer = nn.Linear(decoder_hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def encoder_initHidden(self):
        return (torch.zeros(self.lstm_layers * 2,  self.encoder_hidden_size).to('cuda'), torch.zeros(self.lstm_layers * 2, self.encoder_hidden_size).to('cuda'))
    
    def decoder_initHidden(self):
        return (torch.zeros(self.lstm_layers,  self.decoder_hidden_size).to('cuda'), torch.zeros(self.lstm_layers, self.decoder_hidden_size).to('cuda'))
        
    def encoder_forward(self, category, input, hidden):
        input_combined = torch.cat((category, input), 1)
        output, (h_n, c_n) = self.encoder(input_combined, hidden)
        return output, (h_n, c_n)
    
    def decoder_forward(self, category, input, hidden, encoder_output):
        input_combined = torch.cat((category, input, encoder_output), 1)
        output, (h_n, c_n) = self.decoder(input_combined, hidden)
        output = self.linear_layer(output)
        output = self.softmax(output)
        return output, (h_n, c_n)