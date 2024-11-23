import torch
import glob
import random
import string
import os
import copy

origin_datasets = glob.glob('./datasets/*.txt')

all_kinds = []
all_kinds_names = {}
for data in origin_datasets:
    kind = os.path.splitext(os.path.basename(data))[0]
    all_kinds.append(kind)
    one_kind_names = open(data, encoding='utf-8').read().strip().split('\n')
    all_kinds_names[kind] = one_kind_names

num_of_all_kinds = len(all_kinds)
all_letters = string.ascii_letters + " .,;'-"
num_of_all_letters = len(all_letters)
    
def train_datasets_make():
    kind = random.choice(all_kinds)
    name = random.choice(all_kinds_names[kind])

    kind_tensor = torch.zeros(1, num_of_all_kinds)
    kind_tensor[0][all_kinds.index(kind)] = 1

    input_name_tensor = torch.zeros(len(name), 1, num_of_all_letters)
    for i in range(len(name)):
        letter = name[i]
        input_name_tensor[i][0][all_letters.find(letter)] = 1

    letter_indexes = [all_letters.find(name[j]) for j in range(1, len(name))]
    letter_indexes.append(num_of_all_letters - 1)
    target_name_tensor = torch.LongTensor(letter_indexes)

    return kind_tensor.to('cuda'), input_name_tensor.to('cuda'), target_name_tensor.to('cuda')


def completion_train_datasets_make():
    kind = random.choice(all_kinds)
    name = random.choice(all_kinds_names[kind])
    target_name = copy.deepcopy(name)
    
    name_length = len(name)
    if_comple = int(random.uniform(0, 2))
    
    if if_comple == 1:
        get_times = name_length / 2
        get_locations = random.sample(range(name_length), int(get_times))
        for i in get_locations:
            s = list(name)
            s[i] = '.'
            name = ''.join(s)

    kind_tensor = torch.zeros(1, num_of_all_kinds)
    kind_tensor[0][all_kinds.index(kind)] = 1

    input_name_tensor = torch.zeros(len(name), 1, num_of_all_letters)
    for i in range(len(name)):
        letter = name[i]
        input_name_tensor[i][0][all_letters.find(letter)] = 1

    letter_indexes = [all_letters.find(target_name[j]) for j in range(0, len(target_name))]
    letter_indexes.append(num_of_all_letters - 1)
    target_name_tensor = torch.LongTensor(letter_indexes)

    return kind_tensor.to('cuda'), input_name_tensor.to('cuda'), target_name_tensor.to('cuda')

