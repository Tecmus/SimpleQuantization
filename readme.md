# Pytorch的量化实现

## 量化：
量化就是把高精度的数值在低精度下进行表示，如通过如下映射可以将fp32范围在int8(非连续区间[-128,127] )下表示。

量化可以使现在深层模型如bert、reset-net等计算量比较大的模型可以通过量化实现在不改变模型结构且保持于原始模型的准度相当的情况下，减少模型计算量，压缩模型体积的效果。

## 量化与反量化的实现

### 量化
对称量化，就是映射前后两个范围的零点对齐。

y = round(clamp(x,threshold,-threshold)/scale)

scale= threshold - (-threshold)/127-128= threshold/127

```python
def Q(value,scale):
    out =  torch.clamp(torch.round(value/scale),-127,128)
    return out
```

非对称量化，不要零点对齐，映射的时候只需要加上对应的零点的偏移量即可。
### 反量化
```python
def DQ(value,scale):
    return scale*value
```
量化的逆操作

y = scale*x

由于在后续的矩阵运算中会有多余的展开项从而增加计算复杂度，因此在实践中一般会采用对称量化。

## 量化的应用
量化实际使用主要分为三种方式，

1.训练后量化 (post-training quantization PTQ)

2.动态量化 (dynamic quantization)

3.训练感知量化(qutization-aware-training QAT)

## QAT相关：

本文主要介绍QAT,相比于训练后量化和动态量化，QAT有较高的准确度.

QAT基本思路:

在**训练过程**中加入伪量化节点，使模型参数适应在量化下的表示，
在**预测过程**中将伪量化节点拆开，将计算密集算子(如mat_mul)放入量化和放量化操作之间，这样的变化使恒等变化，这样就能使计算密集算子在低精度下进行，达到加速的效果。

如图：
  
实现:

```python
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
```

伪量化节点的实现:
前向是一个Q-DQ操作,
在实现具体算子时，需要考虑反向传播的值，

通过torch.utils.autograd.Function
```python 
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
```

## Layer实现
和nn.layer初始化部分相同只不过是在预测时和训练时要进行区分。

验证：
  torch官方也实现了训练时感知量化，

对比测试:
  

tensornboard
