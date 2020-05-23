from scipy.sparse import csr_matrix
import numpy as np
import pickle
root = 'source/'
# 选择小说
with open(root+'鸳鸯刀.txt', "r", encoding="utf-8") as file:
    data = file.readlines()
data=''.join(data)

# Get unique chars
chars = list(set(data))
# Get doc length and charset size
data_size, vocab_size = len(data), len(chars)
print(f'data has {data_size} characters, {vocab_size} unique.')
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}

X_train = csr_matrix((len(data), len(chars)), dtype=np.int)
char_id = np.array([chars.index(c) for c in data])
X_train[np.arange(len(data)), char_id] = 1
y_train = np.roll(char_id,-1)

print(X_train)
print(type(y_train))

with open(root + 'X_train.pickle', 'wb') as handle:
    pickle.dump(X_train, handle, protocol=2)

with open(root + 'y_train.pickle', 'wb') as handle:
    pickle.dump(y_train, handle, protocol=2)

with open(root + 'chars.pickle', 'wb') as handle:
    pickle.dump(chars, handle, protocol=2)

with open(root + 'vocab_size.pickle', 'wb') as handle:
    pickle.dump(vocab_size, handle, protocol=2)
