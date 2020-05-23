import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def get_batch(X_train, y_train, seq_length):
    '''Return a training batch with certain number of X and y pairs.'''
    X = X_train
    y = torch.from_numpy(y_train).long()
    for i in range(0, len(y), seq_length):
        id_stop = i+seq_length if i+seq_length < len(y) else len(y)
        yield([torch.from_numpy(X[i:id_stop].toarray().astype(np.float32)),
               y[i:id_stop]])

# 使用训练好的模型训练文本
def sample_chars(rnn, X_seed, h_prev, chars, length=200):
    '''Generate text using trained model'''
    X_next = X_seed
    results = []
    with torch.no_grad():
        for i in range(length):
            y_score, h_prev = rnn(X_next.view(1,1,-1), h_prev)
            y_prob = nn.Softmax(0)(y_score.view(-1)).detach().cpu().numpy()
            y_pred = np.random.choice(chars,1, p=y_prob).item()
            results.append(y_pred)
            X_next = torch.zeros_like(X_seed)
            X_next[chars.index(y_pred)] = 1
    return ''.join(results)


def train(X_batch, y_batch, rnn, optimizer, loss_fn):
    h_prev = rnn.initHidden()
    # print(h_prev)
    optimizer.zero_grad()
    batch_loss = torch.tensor(0, dtype=torch.float)

    for i in range(len(X_batch)):
        y_score, h_prev = rnn(X_batch[i].view(1, 1, -1), h_prev)
        loss = loss_fn(y_score.view(1, -1), y_batch[i].view(1)).cpu()
        batch_loss += loss
    batch_loss.backward()
    optimizer.step()
    return y_score, batch_loss / len(X_batch)

def plot_loss(loss):
    plt.figure(1)
    plt.plot(loss)
    plt.title("LOSS")
    plt.savefig("source/loss.png", dpi=300)
    plt.show()
