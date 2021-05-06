import torch 
import torch.nn as nn
import torch.nn.functional as F
import pdb
    
def Q(value,scale):
    out =  torch.clamp(torch.round(value/scale),-127,128)
    return out

def DQ(value,scale):
    return scale*value

class FakeQuantFunc(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx,input_tensor,scale):
        # ctx.save_for_backward(input, weight, bias)
        out=Q(input_tensor,scale)
        out=DQ(out,scale)
        
        return out
    @staticmethod
    def backward(ctx,gradient_output):        
        # input, weight, bias = ctx.saved_tensors

        return gradient_output, None
    
"""
    input:fp32
    output:int8
"""
class  QuantizationLayer(nn.Module):
    def _init_weights(self,input_features,out_features):
        weights=torch.empty(out_features,input_features)
        nn.init.uniform_(weights)
        weights = nn.Parameter(weights)
        return weights
    
    def _init_bias(self,out_features):
        bias = nn.Parameter(torch.zeros(out_features))
        return bias
    
    def __init__(self,in_features,out_features):
        super(QuantizationLayer,self).__init__()   
        self._weights=self._init_weights(in_features,out_features)
        self._bias=self._init_bias(out_features)
        self.fake_quant = FakeQuantFunc.apply
    def get_scale(self,input_tensor,mode):
        if mode=='per_tensor':
            min_value=torch.min(input_tensor)
            max_value= torch.max(input_tensor)
            max_bound = max(min_value,max_value)
        elif mode=='per_channel':
            pass
        else:
            raise("mode error ")
        print('max_bound',max_bound)
        
        scale = max_bound/127 # [-128,127]
        print('scale',scale)
        return scale
    
        
    def forward(self,input_tensor):
        if self.training : 
            print('enter training')

            scale=self.get_scale(input_tensor,'per_tensor')
            quant_out=self.fake_quant(input_tensor,scale)
            self._weights.data=self.fake_quant(self._weights,scale)
            
            # self._bias.data=self.fake_quant(self._bias,scale)
            
            return F.linear(quant_out,self._weights,self._bias)
        else:
            print('enter inference')
            scale=self.get_scale(input_tensor,'per_tensor')
            quant_input =Q(input_tensor,scale)
            self._weights.data =Q(self._weights.data,scale)
            self._weights.data.to(torch.int8)
            # print('out_weight',self._weights)
            # self._bias.data =Q(self._bias.data,scale)
            
            return F.linear(quant_input,self._weights,self._bias)*scale  #DQ
    
   

# class QuantMnist(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.quant_layer = QuantizationLayer(4,3)
        
#     def forward(self, x):
#         x = self.quant_layer(x)
#         return x

# model=M()

# # input_tensor=torch.arange(50).view(-1,10)
# input_tensor=torch.rand(5,4)*10-5
# # print(input_tensor)
# # model.eval()
# y=model(input_tensor)
# print('out y \t',y)
# print('out y size \t',y.size())
# print(model)
