import torch 
import matplotlib.pyplot as plt
from RNN_Network import RNN, LSTM, Seq2Seq
from datasets_prepare import train_datasets_make, num_of_all_letters, num_of_all_kinds, all_kinds, all_letters, completion_train_datasets_make
from tqdm import tqdm
import torch.nn as nn
import sys
from adjustText import adjust_text 

mode = sys.argv[1]  # Train, Test, Completion, CompletionTest
model_type = sys.argv[2] # RNN, LSTM, BiLSTM, Seq2Seq

if model_type == 'RNN':
    rnn = RNN(num_of_all_letters, 128, num_of_all_letters, num_of_all_kinds).to('cuda')
elif model_type == 'LSTM':
    rnn = LSTM(num_of_all_letters, 128, num_of_all_letters, num_of_all_kinds, LSTM_nums=1).to('cuda')
elif model_type == 'BiLSTM':
    rnn = LSTM(num_of_all_letters, 128, num_of_all_letters, num_of_all_kinds, bidirectional=True, LSTM_nums=1).to('cuda')
elif model_type == 'Seq2Seq':
    rnn = Seq2Seq(num_of_all_letters, 128, 256, num_of_all_letters, num_of_all_kinds).to('cuda')
else:
    assert "Model Type Error!"
    exit()

rnn.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr = 5e-4)

def train(kind_tensor, input_name_tensor, target_name_tensor):
    hidden = rnn.initHidden()
    optimizer.zero_grad()
    target_name_tensor.unsqueeze_(-1)
    loss = torch.tensor(0.0, requires_grad=True).to('cuda')
    
    for i in range(input_name_tensor.size(0)):
        output, hidden = rnn(kind_tensor, input_name_tensor[i], hidden)
        loss = loss + criterion(output, target_name_tensor[i])
    loss.backward()
    optimizer.step()
    return output, loss.item()/input_name_tensor.size(0)


def completion_train(kind_tensor, input_name_tensor, target_name_tensor):
    optimizer.zero_grad()
    target_name_tensor.unsqueeze_(-1)
    loss = torch.tensor(0.0, requires_grad=True).to('cuda')
    
    encoder_hidden = rnn.encoder_initHidden()
    decoder_hidden = rnn.decoder_initHidden()
    
    # 编码器先获取全局信息
    # 判断是否为补全任务
    iscompletion = False
    for i in range(input_name_tensor.size(0)):
        if input_name_tensor[i][0][all_letters.find('.')] == 1:
            iscompletion = True # 存在缺项则为补全任务，需要提取全局信息
            
    encoder_output = torch.zeros(1, rnn.encoder_hidden_size * 2).to('cuda')
    if iscompletion:
        for i in range(input_name_tensor.size(0)):
            encoder_output, encoder_hidden = rnn.encoder_forward(kind_tensor, input_name_tensor[i], encoder_hidden)
        
    # 解码器根据编码器编码信息和输入信息进行补全训练或进行普通训练
    for i in range(input_name_tensor.size(0)):
        output, decoder_hidden = rnn.decoder_forward(kind_tensor, input_name_tensor[i], decoder_hidden, encoder_output)
        loss = loss + criterion(output, target_name_tensor[i])


    loss.backward()
    optimizer.step()
    return output, loss.item()/input_name_tensor.size(0)


def completion_predict(kind, first='A'):
    with torch.no_grad():
        encoder_hidden = rnn.encoder_initHidden()
        decoder_hidden = rnn.decoder_initHidden()
        
        kind_tensor = torch.zeros(1, num_of_all_kinds).to('cuda')
        kind_tensor[0][all_kinds.index(kind)] = 1
        input = torch.zeros(len(first), 1, num_of_all_letters).to('cuda')
        predict_name = first
        
        iscompletion = False
        completion_locations = []
        encoder_output = torch.zeros(1, rnn.encoder_hidden_size * 2).to('cuda')
        for i in range(len(first)):
            input[i][0][all_letters.find(first[i])] = 1
            if first[i] == '.':     # 判断是否为补全任务
                iscompletion = True
                completion_locations.append(i)
                
        if iscompletion:
            for i in range(len(first)):
                encoder_output, encoder_hidden = rnn.encoder_forward(kind_tensor, input[i], encoder_hidden)
        
        words_prob = []
        words = []

        for j in range(len(first)):
            output, decoder_hidden = rnn.decoder_forward(kind_tensor, input[j], decoder_hidden, encoder_output)
            tv, ti = output.topk(5)
            words_prob.append(torch.exp(tv)[0].tolist())
            print(torch.exp(tv)[0])
            word_5 = []
            for k in range(5):
                print(all_letters[ti[0][k]], end = ' ')
                word_5.append(all_letters[ti[0][k]])
            words.append(word_5)
            print('\n')
            if j in completion_locations:
                tv, ti = output.topk(1)
                t = ti[0][0]
                if(t == num_of_all_letters - 1):
                    is_end = True
                    s1 = list(predict_name)
                    s1[j] = ''
                    predict_name = ''.join(s1)
                    break
                s1 = list(predict_name)
                s1[j] = all_letters[t]
                predict_name = ''.join(s1)
                
        plt.figure()
        plt.plot(list(range(1, len(words_prob)+1)), words_prob,linestyle='', marker='o')
        texts = []
        for x, y, word in zip(list(range(1, len(words_prob)+1)), words_prob, words):
            for w, prob in zip(word, y):
                texts.append(plt.text(x, prob, w))
                
        adjust_text(texts, )
        
        plt.show()
        plt.savefig('./completion_words_map.png')
                    
        return predict_name


if __name__ == '__main__':
    
    if mode == 'Train':
        losses = 0
        L_loss = []
        Train_Steps = []
        train_steps = 100001
        
        with tqdm(total=train_steps) as t:
            for i in range(train_steps):
                kind_tensor, input_name_tensor, target_name_tensor = train_datasets_make()
                output, loss = completion_train(kind_tensor, input_name_tensor, target_name_tensor)
                losses += loss
                
                if i % 500 == 0 and i != 0:
                    L_loss.append(round(losses/500.0, 5))
                    Train_Steps.append(i)
                    losses = 0
                    
                if i % 100000 == 0:
                    torch.save(rnn.state_dict(), rnn.__class__.__name__+'%i'%i+"_.pth")
                
                t.set_postfix(train_loss='%.4f'%loss)
                t.update(1)
                
        torch.save(rnn.state_dict(), rnn.__class__.__name__+"_last_.pth")

        plt.figure()
        plt.plot(Train_Steps, L_loss)
        plt.ylabel("Loss")
        plt.show()
        plt.savefig(rnn.__class__.__name__+'_Loss.png')
        
    elif mode == 'Completion':
        losses = 0
        L_loss = []
        Train_Steps = []
        train_steps = 100001
        
        with tqdm(total=train_steps) as t:
            for i in range(train_steps):
                kind_tensor, input_name_tensor, target_name_tensor = completion_train_datasets_make()
                output, loss = completion_train(kind_tensor, input_name_tensor, target_name_tensor)
                losses += loss
                
                if i % 500 == 0 and i != 0:
                    L_loss.append(round(losses/500.0, 5))
                    Train_Steps.append(i)
                    losses = 0
                    
                if i % 100000 == 0:
                    torch.save(rnn.state_dict(), rnn.__class__.__name__+'%i'%i+"_.pth")
                
                t.set_postfix(train_loss='%.4f'%loss)
                t.update(1)
                
        torch.save(rnn.state_dict(), rnn.__class__.__name__+"_last_.pth")

        plt.figure()
        plt.plot(Train_Steps, L_loss)
        plt.ylabel("Loss")
        plt.show()
        plt.savefig(rnn.__class__.__name__+'_Loss.png')
        
    elif mode == 'CompletionTest':
        model_path = sys.argv[3]
        kinds = sys.argv[4]
        first_input = sys.argv[5]
        rnn.load_state_dict(torch.load(model_path))
        
        predict_name = completion_predict(kinds, first_input)
        print(predict_name)

        
    elif mode == 'Test':
        model_path = sys.argv[3]
        kinds = sys.argv[4]
        first_input = sys.argv[5]
        rnn.load_state_dict(torch.load(model_path))
        # rnn = torch.load(model_path).to('cuda')
        words_prob = []
        words = []
        def predict(kind, first='A'):
            with torch.no_grad():
                kind_tensor = torch.zeros(1, num_of_all_kinds).to('cuda')
                kind_tensor[0][all_kinds.index(kind)] = 1
                input = torch.zeros(len(first), 1, num_of_all_letters).to('cuda')
                for i in range(len(first)):
                    input[i][0][all_letters.find(first[i])] = 1
                hidden = rnn.initHidden()
                predict_name = first
                
                for i in range(7):
                    if i==0:
                        for j in range(len(first)):
                            output, hidden = rnn(kind_tensor, input[j], hidden)
                    output, hidden = rnn(kind_tensor, input[0], hidden)
                    tv, ti = output.topk(5)
                    words_prob.append(torch.exp(tv)[0].tolist())
                    print(torch.exp(tv)[0])
                    word_5 = []
                    for k in range(5):
                        print(all_letters[ti[0][k]], end = ' ')
                        word_5.append(all_letters[ti[0][k]])
                    words.append(word_5)
                    print('\n')
                        
                    tv, ti = output.topk(1)
                    t = ti[0][0]
                    if(t == num_of_all_letters - 1):
                        break
                    else:
                        predict_name += all_letters[t]
                        
                    input = torch.zeros(len(first), 1, num_of_all_letters).to('cuda')
                    input[0][0][t] = 1
                return predict_name
            
        predict_name = predict(kinds, first_input)
        
        plt.figure()
        plt.plot(list(range(1, len(words_prob)+1)), words_prob,linestyle='', marker='o')
        texts = []
        for x, y, word in zip(list(range(1, len(words_prob)+1)), words_prob, words):
            for w, prob in zip(word, y):
                texts.append(plt.text(x, prob, w))
                
        adjust_text(texts, ),#箭头样式 lw= 1,#线宽 color='red')#箭头颜色 )
        
        plt.show()
        plt.savefig('./'+rnn.__class__.__name__+'words_map.png')
        
        print(predict_name)
