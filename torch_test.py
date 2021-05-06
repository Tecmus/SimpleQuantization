import torch
arr= [0.1,0.2,1.5,3.4]
t=torch.Tensor(arr) 
# t=torch.round(t)
# x=torch.clamp(t,1,2)
# print(x.type())
# num_bits=0
# max_bound = torch.tensor((2.0**(num_bits - 1 + int(True))) - 1.0)
# print(max_bound.type())
# print(max_bound)
# min_bound=0
# inputs=t
# x=torch.clamp((inputs ).round_(), min_bound, max_bound)
# print(x)
# print(x.type())
# print(t.type())

y=torch.arange(40).view(4,1,2,5)
print(torch.min(y))
print(y.size())
x=torch.tensor([1,2,3,4]).view(-1,1)
print(x.size())
z=y/x
print(z)

torch.quantize_per_channel()

# thresh=torch.tensor(-128,dtype=torch.int8)
# t=t.mul(0.5).round().clamp(thresh, -thresh)
# print(t)
# print(t)
