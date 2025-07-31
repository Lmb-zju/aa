import torch

# # vector x vector
# tensor1 = torch.randn(3)
# tensor2 = torch.randn(3)
# print(torch.matmul(tensor1, tensor2).size())
# print(torch.matmul(tensor1, tensor2).shape)
#
# torch.Size([])


# matrix x vector
# tensor1 = torch.randn(3, 4)
# tensor2 = torch.randn(4)
# a = torch.matmul(tensor1, tensor2).size()
# print(a)

# torch.Size([3])


# batched matrix x broadcasted vector
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4)
a = torch.matmul(tensor1, tensor2).size()
print(a)

# torch.Size([10, 3])


# # batched matrix x batched matrix
# tensor1 = torch.randn(10, 3, 4)
# tensor2 = torch.randn(10, 4, 5)
# torch.matmul(tensor1, tensor2).size()
#
# torch.Size([10, 3, 5])


# # batched matrix x broadcasted matrix
# tensor1 = torch.randn(10, 3, 4)
# tensor2 = torch.randn(4, 5)
# torch.matmul(tensor1, tensor2).size()
#
# torch.Size([10, 3, 5])
