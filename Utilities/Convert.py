import torch
from torch import Tensor


def indexTensor(names: list, max_len: int, allowed_chars: list):
    tensor = torch.zeros(max_len, len(names)).type(torch.LongTensor)
    for i, name in enumerate(names):
        for j, letter in enumerate(name):
            index = allowed_chars.index(letter)

            if index < 0:
                raise Exception(f'{names[j][i]} is not a char in {allowed_chars}')

            tensor[j][i] = index
    return tensor

def zipcodeTensor(zip_codes:list, zip_code_length: int):
    tensor = torch.zeros(zip_code_length, len(zip_codes)).type(torch.LongTensor)

    for i, name in enumerate(zip_codes):
        zip_code = str(zip_codes[i].item())
        
        for j in range(zip_code_length):
            tensor[j][i] = int(zip_code[j])
    
    return tensor

def lengthTestTensor(lengths:list):
    tensor = torch.zeros(len(lengths)).type(torch.LongTensor)
    for i, length in enumerate(lengths):
        tensor[i] = length[i]
    
    return tensor

def targetsTensor(names: list, max_len: int, allowed_chars: list):
    batch_sz = len(names)
    ret = torch.zeros(max_len, batch_sz).type(torch.LongTensor)
    for i in range(max_len):
        for j in range(batch_sz):
            index = allowed_chars.index(names[j][i])

            if index < 0:
                raise Exception(f'{names[j][i]} is not a char in {allowed_chars}')

            ret[i][j] = index
    return ret
