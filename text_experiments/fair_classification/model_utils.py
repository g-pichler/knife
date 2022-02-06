import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm
import itertools


class Classifier(torch.nn.Module):
    '''
    General Classifier with Leaky Relu as activation function
    '''

    def __init__(self, input_dim, output_dim, use_complex_classifier=False):
        super(Classifier, self).__init__()
        if use_complex_classifier:
            self.net = nn.Sequential(nn.Linear(input_dim, input_dim), nn.LeakyReLU(), nn.Linear(input_dim, input_dim),
                                     nn.LeakyReLU(), nn.Linear(input_dim, input_dim),
                                     nn.LeakyReLU(), nn.Linear(input_dim, output_dim),
                                     nn.LogSoftmax())
        else:
            self.net = nn.Sequential(nn.Linear(input_dim, output_dim), nn.LogSoftmax())

    def forward(self, input):
        input = torch.sum(input, dim=0)
        return self.net(input)

    def weights_init(self):
        for module in self.net:
            try:
                torch.nn.init.xavier_uniform(module.weight.data)
            except:
                print(module)


class ClassifierGamma(torch.nn.Module):
    '''
    General Classifier with Leaky Relu as activation function
    '''

    def __init__(self, input_dim, output_dim):
        super(ClassifierGamma, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, input_dim), nn.LeakyReLU(), nn.Linear(input_dim, input_dim),
                                 torch.nn.Dropout(p=0.1, inplace=False),
                                 nn.LeakyReLU(), nn.Linear(input_dim, output_dim),
                                 nn.LogSoftmax())

    def forward(self, input):
        input = torch.sum(input, dim=0)
        return self.net(input)

    def weights_init(self):
        for module in self.net:
            try:
                torch.nn.init.xavier_uniform(module.weight.data)
            except:
                print(module)


# MINE : https://github.com/sungyubkim/MINE-Mutual-Information-Neural-Estimation-/blob/master/GAN_MINE.ipynb
# Paper https://arxiv.org/pdf/1806.04498.pdf
def update_target(ma_net, net, update_rate=1e-1):
    # update moving average network parameters using network
    for ma_net_param, net_param in zip(ma_net.parameters(), net.parameters()):
        ma_net_param.data.copy_((1.0 - update_rate) \
                                * ma_net_param.data + update_rate * net_param.data)


# control which parameters are frozen / free for optimization
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def comput_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return torch.tensor(total_norm)


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


class EncoderRNN(nn.Module):
    def __init__(self, args, input_size, hidden_size, number_of_layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.args = args

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=number_of_layers,
                          dropout=args.dropout)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(2 * self.args.number_of_layers, self.args.batch_size, self.hidden_size,
                           device=self.args.device)


class DecoderRNN(nn.Module):
    def __init__(self, args, hidden_size, output_size, number_of_layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.args = args

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=2 * number_of_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.args.batch_size, self.hidden_size, device=self.args.device)


class EMA():
    '''inspired from http://www.programmersought.com/article/28492072406/'''

    def __init__(self, model, args, decay=0.99):
        self.model = model
        self.args = args
        self.decay = decay
        self.shadow = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data.to(self.args.device) + self.decay * self.shadow[name].to(
                    self.args.device)
                self.shadow[name] = new_average.clone()


def corrupt_input(model, input_tensor):
    ''' pqs de premiere sequcnce pas de pad '''
    pad_indices = (input_tensor == model.args.tokenizer.pad_token_id) + (
            input_tensor == model.args.tokenizer.sep_token_id) + (
                          input_tensor == model.args.tokenizer.cls_token_id)
    pad_indices.to(model.args.device)
    ### MASK TOKEN ###
    """
    special_tokens_mask = self.args.tokenizer.mask_token_id
    input_tensor_to_corrupt = input_tensor.clone()
    probability_matrix = torch.full(input_tensor_to_corrupt.shape, self.args.noise_p).to(self.args.device)
    masked_indices = torch.bernoulli(probability_matrix).byte() & ~pad_indices
    input_tensor_to_corrupt[masked_indices] = torch.tensor(special_tokens_mask).to(self.args.device)"""

    ### RANDOMLY CHANGE SOME TOKENS ###
    input_tensor_to_corrupt = input_tensor.clone()
    probability_matrix = torch.full(input_tensor_to_corrupt.shape, model.args.noise_p).to(model.args.device)
    random_indices = torch.bernoulli(probability_matrix).byte() & (~pad_indices).byte()
    random_words = torch.randint(len(model.args.tokenizer), input_tensor.shape, dtype=torch.long).to(
        model.args.device)
    input_tensor_to_corrupt[random_indices] = random_words[random_indices]

    ### RANDOMLY SWAP ORDER OF TOKENS ###
    # BUILD THE CORRECT MATRIX
    corrupt_batch = True if random.random() < model.args.noise_p else False
    input_tensor_corrupted_ = []
    for b in range(model.args.batch_size):
        lengths = torch.sum(input_tensor[b, :] != model.args.tokenizer.pad_token_id)
        perms = list(itertools.permutations(list(range(1, lengths - 1)), 2))  # don't permut beo or eos
        random_perm = random.choice(perms)
        index_ = list(range(model.args.max_length))
        if corrupt_batch:
            try:
                index_[random_perm[0]], index_[random_perm[1]] = index_[random_perm[1]], index_[random_perm[0]]
            except:
                print("Corruption Error Empty")
        input_tensor_corrupted_.append(
            torch.index_select(input_tensor_to_corrupt[b, :].unsqueeze(0), 1,
                               torch.LongTensor(index_).to(model.args.device)))

    input_tensor_corrupted = torch.cat(input_tensor_corrupted_, dim=0)

    return input_tensor_corrupted
