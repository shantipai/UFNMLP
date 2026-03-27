import numbers
from einops import rearrange
import torch
import torch.nn.functional as F
class SAttention(torch.nn.Module):
    def __init__(self, dims, heads, bias, depth) -> None:
        super(SAttention, self).__init__()
        self.heads = heads
        self.temperature = torch.nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = torch.nn.Conv2d(dims, dims*3, kernel_size=1, bias=False)
        self.qkvConv = torch.nn.Conv2d(dims*3, dims*3, kernel_size=3, stride=1, padding=1, groups=dims*3, bias=bias)
        self.projectOut = torch.nn.Conv2d(dims, dims, kernel_size=1, bias=bias)
        self.test = torch.nn.Identity()
        self.windowSize = [8, 8]
        self.depth = depth
        self.shiftSize = self.windowSize[0]//2

    def forward(self, x):

        if(self.depth)%2:
            x = torch.roll(x, shifts=(-self.shiftSize, -self.shiftSize), dims=(2, 3))
        
        Wsize = self.windowSize
        b, c, h, w = x.shape
        x = rearrange(x, 'b c (h b0) (w b1) -> (b h w) c b0 b1', b0=Wsize[0], b1=Wsize[1])

        qkv = self.qkvConv(self.qkv(x))

        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')

        q = rearrange(q, 'b c (head s) -> b head s c', head = self.heads)
        k = rearrange(k, 'b c (head s) -> b head s c', head = self.heads)
        v = rearrange(v, 'b c (head s) -> b head s c', head = self.heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)
        attn = self.test(attn)
        out = (attn @ v)

        out = rearrange(out, 'b head s c -> b c (head s)', head = self.heads, s=Wsize[0]*Wsize[1]//self.heads)
        out = rearrange(out, 'b c (h w) -> b c h w', h = Wsize[0], w=Wsize[1])
        out = rearrange(out, '(b h w) c b0 b1 -> b c (h b0) (w b1)', h=h//Wsize[0], w=w//Wsize[1], b0=Wsize[0])


        out = self.projectOut(out)
        return out




class FeedForward(torch.nn.Module):
    def __init__(self, dim, ffnExpansionFactor, bias) -> None:
        super(FeedForward, self).__init__()
        hiddenFeatures = int(dim*ffnExpansionFactor)
        self.projectIn = torch.nn.Conv2d(dim, hiddenFeatures*2, kernel_size=1, bias=bias)
        self.dwConv = torch.nn.Conv2d(hiddenFeatures* 2, hiddenFeatures* 2, kernel_size=3, stride=1, padding=1, groups=hiddenFeatures*2, bias=bias)
        self.projectOut = torch.nn.Conv2d(hiddenFeatures, dim, kernel_size=1, bias=bias)
        self.gelu = torch.nn.GELU()
    def forward(self, x):
        x = self.projectIn(x)
        x1, x2 = self.dwConv(x).chunk(2, dim=1)
        x = self.gelu(x1)*x2
        x = self.projectOut(x)
        return x

class BiasFreeLayerNorm(torch.nn.Module):
    def __init__(self, normalizedShape) -> None:
        super(BiasFreeLayerNorm, self).__init__()
        if isinstance(normalizedShape, numbers.Integral):
            normalizedShape = (normalizedShape,)
        normalizedShape = torch.Size(normalizedShape)

        assert len(normalizedShape) == 1
        self.weight = torch.nn.Parameter(torch.ones(normalizedShape))
        self.normalizedShape = normalizedShape

    def forward(self, x):
        sigma = x.var(-1, keepdim = True, unbiased = False)
        return x/ torch.sqrt(sigma+1e-5) * self.weight

class WithBiasLayerNorm(torch.nn.Module):
    def __init__(self, normalizedShape) -> None:
        super(WithBiasLayerNorm, self).__init__()
        if isinstance(normalizedShape, numbers.Integral):
            normalizedShape = (normalizedShape,)
        normalizedShape = torch.Size(normalizedShape)
        assert len(normalizedShape) == 1
        self.weight = torch.nn.Parameter(torch.ones(normalizedShape))
        self.bias = torch.nn.Parameter(torch.zeros(normalizedShape))
        self.normalizedShape = normalizedShape
    def forward(self, x):
        mu = x.mean(-1, keepdim = True)
        sigma = x.var(-1, unbiased=False, keepdim=True)
        return (x-mu)/torch.sqrt(sigma+1e-5)*self.weight+self.bias
    
def Convert3D(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def Convert4D(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(torch.nn.Module):
    def __init__(self, dim, layerNormType) -> None:
        super(LayerNorm, self).__init__()
        if layerNormType == 'BiasFree':
            self.body = BiasFreeLayerNorm(dim) 
        else:
            self.body = WithBiasLayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return Convert4D(self.body(Convert3D(x)), h, w)



class STransformerBlock(torch.nn.Module):
    def __init__(self, dim, numHeads, ffnExpansionFactor, bias, layerNormType, depth) -> None:
        super(STransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, layerNormType=layerNormType)
        self.attn = SAttention(dim, numHeads, bias, depth=depth)
        self.norm2 = LayerNorm(dim, layerNormType)
        self.ffn = FeedForward(dim, ffnExpansionFactor, bias)
    def forward(self, x):
        x = x+ self.attn(self.norm1(x))
        x = x+ self.ffn(self.norm2(x))
        return x
    


    

class STransformerList(torch.nn.Module):
    def __init__(self, dim, numBlocks, numHeads, ffnExpansionFactor, bias, layerNormType, depth) -> None:
        super(STransformerList, self).__init__()
        self.blocks = torch.nn.ModuleList([])
        for i in range(numBlocks):
            self.blocks.append(STransformerBlock(dim=dim, numHeads=numHeads, ffnExpansionFactor=ffnExpansionFactor,
                                                bias=bias, layerNormType=layerNormType, depth=depth))
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
class SPFST(torch.nn.Module):
    def __init__(self, inDim, outDim, dim, numBlocks = [2, 2, 2]) -> None:
        super(SPFST, self).__init__()
        self.dim = dim
        self.scales = len(numBlocks)
        self.embedding = torch.nn.Conv2d(inDim, self.dim, 3, 1, 1, bias=False)
        self.encoderLayer = torch.nn.ModuleList([])
        dim_scale = dim
        for i in range(self.scales-1):
            self.encoderLayer.append(torch.nn.ModuleList([
                STransformerList(dim=dim_scale, numBlocks= numBlocks[i], numHeads=4, ffnExpansionFactor=2.66, bias=False, layerNormType='WithBias', depth=i),
                torch.nn.Conv2d(dim_scale, dim_scale* 2, 4, 2, 1, bias=False)
            ]))
            dim_scale*=2


        self.bottelNeck = STransformerList(
            dim = dim_scale, numBlocks=numBlocks[-1], numHeads=4, ffnExpansionFactor=2.66, bias=False, layerNormType='WithBias', depth=self.scales)
        self.decoderLayers = torch.nn.ModuleList([])
        for i in range(self.scales-1):
            self.decoderLayers.append(torch.nn.ModuleList([
                torch.nn.ConvTranspose2d(dim_scale, dim_scale // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                torch.nn.Conv2d(dim_scale, dim_scale//2, 1, 1 , bias=False),
                STransformerList(dim=dim_scale//2, numBlocks=numBlocks[self.scales-2-i],numHeads=4, ffnExpansionFactor=2.66, bias=False, layerNormType='WithBias', depth=self.scales-i-1)
            ]))
            dim_scale //= 2

        self.mapping = torch.nn.Conv2d(self.dim, outDim, 3, 1, 1, bias=False)
        #self.decode = torch.nn.Conv2d(inDim, outDim, 1, 1, 0, bias=False)

    def forward(self, x):
        b, c, hInput, wInput = x.shape
        hb, wb =  64, 64
        hPadding = (hb- hInput% hb)%hb
        wPadding = (wb - wInput% wb)% wb
        x = F.pad(x, [0, hPadding, 0, wPadding], mode='reflect')

        fea = self.embedding(x)

        feaEncoder = []
        for (HSAB, feaDownSample) in self.encoderLayer:
            fea = HSAB(fea)
            feaEncoder.append(fea)
            fea = feaDownSample(fea)

        fea = self.bottelNeck(fea)

        for i, (feaUpSample, fution, HSAB) in enumerate(self.decoderLayers):
            fea = feaUpSample(fea)
            fea = fution(torch.cat([fea, feaEncoder[self.scales-2-i]], dim=1))
            fea = HSAB(fea)

        out = self.mapping(fea) + x
        return out[:, :, :hInput, :wInput]