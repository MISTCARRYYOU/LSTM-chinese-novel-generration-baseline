from util import *
from scipy.sparse import csr_matrix
import pickle
from Model import nn_LSTM
from torch.optim.lr_scheduler import LambdaLR
import os

root = 'source/'

with open(root + 'X_train.pickle', 'rb') as handle:
    X_train = pickle.load(handle)
with open(root + 'y_train.pickle', 'rb') as handle:
    y_train = pickle.load(handle)
with open(root + 'chars.pickle', 'rb') as handle:
    chars = pickle.load(handle)
with open(root + 'vocab_size.pickle', 'rb') as handle:
    vocab_size = pickle.load(handle)


print("数据集大小：", X_train.shape, y_train.shape)


hidden_size = 256
seq_length = 25
EPOCHS = 25


rnn = nn_LSTM(vocab_size, hidden_size, vocab_size)

# 看是否有训练差不多的模型
if os.path.exists(root + 'save_model.pth'):
    try:
        checkpoint = torch.load(root + 'save_model.pth')
        rnn.load_state_dict(checkpoint)
        print("finetuning!")
    except RuntimeError:
        print("新模型将覆盖原模型！")


# 开始训练过程
rnn = rnn.cuda()  # 使用gpu
loss_fn = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam([{'params': rnn.parameters(), 'initial_lr': 0.005}], lr=0.005)

# scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))    # 训练模型
all_losses = []
print_every = 500
with open(root + 'loss.txt', "a", encoding="utf-8") as file:
    for epoch in range(EPOCHS):
        # print("当前学习率：", optimizer.param_groups[0]['lr'])
        for batch in get_batch(X_train, y_train, seq_length):
            X_batch, y_batch = batch
            # print(X_batch, y_batch)
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
            _, batch_loss = train(X_batch, y_batch, rnn, optimizer, loss_fn)
            batch_loss = batch_loss.cpu()
            all_losses.append(batch_loss.item())
            file.write("{} {}\n".format(epoch, batch_loss.item()))
            if len(all_losses) % print_every == 1:
                print("epoch:", epoch+1, f'----\nRunning Avg Loss:{np.mean(all_losses[-print_every:])} at iter: {len(all_losses)}\n----')
                print(sample_chars(rnn, X_batch[0], rnn.initHidden(), chars, 200))
        # scheduler.step()
        # 每过一个epoch保存一次模型
        torch.save(rnn.state_dict(), root + 'save_model.pth')
print("训练完成")
plot_loss(all_losses)
print(sample_chars(rnn, X_batch[0], rnn.initHidden(), chars, 200))
# for i in rnn.state_dict():
#     print(i)



