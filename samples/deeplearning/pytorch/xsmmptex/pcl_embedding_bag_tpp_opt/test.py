
import torch
import torch.nn as nn
import pcl_embedding_bag

torch.manual_seed(999)
#E = nn.EmbeddingBag(10, 5, mode="sum", sparse=True)
embedding_sum = nn.EmbeddingBag(10, 3, mode='sum', sparse=True)
# a batch of 2 samples of 4 indices each
input = torch.LongTensor([1,2,4,5,4,3,2,9])
offsets = torch.LongTensor([0,4])
x = embedding_sum(input, offsets)

print(x)

x = torch.mean(x)
x.backward()
print(embedding_sum.weight.grad.data.coalesce())
