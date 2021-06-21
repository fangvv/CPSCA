import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from collections import OrderedDict
import math
import numpy as np

i=0
s1  = np.zeros(shape=(64, 1,16))
s2  = np.zeros(shape=(64, 1,16))
s3  = np.zeros(shape=(64, 1,16))
s4  = np.zeros(shape=(64, 1,16))
s5  = np.zeros(shape=(64, 1,16))
s6  = np.zeros(shape=(64, 1,16))
s7  = np.zeros(shape=(64, 1,16))
s8  = np.zeros(shape=(64, 1,16))
s9  = np.zeros(shape=(64, 1,16))
s10 = np.zeros(shape=(64, 1,16))
s11 = np.zeros(shape=(64, 1,16))
s12 = np.zeros(shape=(64, 1,16))
s13 = np.zeros(shape=(64, 1,16))
s14 = np.zeros(shape=(64, 1,16))
s15 = np.zeros(shape=(64, 1,16))
s16 = np.zeros(shape=(64, 1,16))
s17 = np.zeros(shape=(64, 1,16))
s18 = np.zeros(shape=(64, 1,16))
s19 = np.zeros(shape=(64, 1,32))
s20 = np.zeros(shape=(64, 1,32))
s21 = np.zeros(shape=(64, 1,32))
s22 = np.zeros(shape=(64, 1,32))
s23 = np.zeros(shape=(64, 1,32))
s24 = np.zeros(shape=(64, 1,32))
s25 = np.zeros(shape=(64, 1,32))
s26 = np.zeros(shape=(64, 1,32))
s27 = np.zeros(shape=(64, 1,32))
s28 = np.zeros(shape=(64, 1,32))
s29 = np.zeros(shape=(64, 1,32))
s30 = np.zeros(shape=(64, 1,32))
s31 = np.zeros(shape=(64, 1,32))
s32 = np.zeros(shape=(64, 1,32))
s33 = np.zeros(shape=(64, 1,32))
s34 = np.zeros(shape=(64, 1,32))
s35 = np.zeros(shape=(64, 1,32))
s36 = np.zeros(shape=(64, 1,32))
s37 = np.zeros(shape=(64, 1,64))
s38 = np.zeros(shape=(64, 1,64))
s39 = np.zeros(shape=(64, 1,64))
s40 = np.zeros(shape=(64, 1,64))
s41 = np.zeros(shape=(64, 1,64))
s42 = np.zeros(shape=(64, 1,64))
s43 = np.zeros(shape=(64, 1,64))
s44 = np.zeros(shape=(64, 1,64))
s45 = np.zeros(shape=(64, 1,64))
s46 = np.zeros(shape=(64, 1,64))
s47 = np.zeros(shape=(64, 1,64))
s48 = np.zeros(shape=(64, 1,64))
s49 = np.zeros(shape=(64, 1,64))
s50 = np.zeros(shape=(64, 1,64))
s51 = np.zeros(shape=(64, 1,64))
s52 = np.zeros(shape=(64, 1,64))
s53 = np.zeros(shape=(64, 1,64))
s54 = np.zeros(shape=(64, 1,64))


sum=[s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,s25,s26,s27,s28,s29,s30,s31,s32,s33,s34,s35,s36,s37,s38,s39,s40,s41,s42,s43,s44,s45,s46,s47,s48,s49,s50,s51,s52,s53,s54]

class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups = 64,eps=1e-5):
        super(SpatialGroupEnhance, self).__init__()
        self.groups   = groups
        self.compress = ChannelPool()
        self.weight   = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1, 1))
        self.sig      = nn.Sigmoid()
        self.eps = eps

    def forward(self, x): # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)
        xn= self.compress(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x


class ChannelGate(nn.Module):
    def __init__(self, gate_channels,  pool_types=['avg','max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.gn = GroupNorm(gate_channels)
        self.pool_types = pool_types
    def forward(self, x):
        global i
        global s1
        global s2
        global s3
        global s4
        global s5
        global s6
        global s7
        global s8
        global s9
        global s10
        global s11
        global s12
        global s13
        global s14
        global s15
        global s16
        global s17
        global s18
        global s19
        global s20
        global s21
        global s22
        global s23
        global s24
        global s25
        global s26
        global s27
        global s28
        global s29
        global s30
        global s31
        global s32
        global s33
        global s34
        global s35
        global s36
        global s37
        global s38
        global s39
        global s40
        global s41
        global s42
        global s43
        global s44
        global s45
        global s46
        global s47
        global s48
        global s49
        global s50
        global s51
        global s52
        global s53
        global s54

        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.gn( avg_pool )

            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.gn( max_pool )

            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.gn(lp_pool)
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.gn(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = F.sigmoid( channel_att_sum ).expand_as(x)
        scale1 = F.sigmoid(channel_att_sum).unsqueeze(1)
        scale2 = scale1.squeeze(3)
        scale3 = scale2.squeeze(3)
        scale4 = scale3.detach().cpu()
        c = scale4.numpy().tolist()

        if i % 54==0:
            s1+=np.array(c[0])
            print('-------------------------conv1--------------------------------'.format(i=i%54+1))
            print(s1[0][0].tolist())
        if i % 54==1:
            s2+=np.array(c[0])
            print('-------------------------conv2--------------------------------'.format(i=i%54+1))
            print(s2[0][0].tolist())
        if i % 54==2:
            s3+=np.array(c[0])
            print('-------------------------conv3--------------------------------'.format(i=i%54+1))
            print(s3[0][0].tolist())
        if i % 54==3:
            s4+=np.array(c[0])
            print('-------------------------conv4--------------------------------'.format(i=i%54+1))
            print(s4[0][0].tolist())
        if i % 54==4:
            s5+=np.array(c[0])
            print('-------------------------conv5--------------------------------'.format(i=i%13+1))
            print(s5[0][0].tolist())
        if i % 54==5:
            s6+=np.array(c[0])
            print('-------------------------conv6--------------------------------'.format(i=i%54+1))
            print(s6[0][0].tolist())
        if i % 54==6:
            s7+=np.array(c[0])
            print('-------------------------conv7--------------------------------'.format(i=i%54+1))
            print(s7[0][0].tolist())
        if i % 54==7:
            s8+=np.array(c[0])
            print('-------------------------conv8--------------------------------'.format(i=i%54+1))
            print(s8[0][0].tolist())
        if i % 54==8:
            s9+=np.array(c[0])
            print('-------------------------conv9--------------------------------'.format(i=i%54+1))
            print(s9[0][0].tolist())
        if i % 54==9:
            s10+=np.array(c[0])
            print('-------------------------conv10--------------------------------'.format(i=i%54+1))
            print(s10[0][0].tolist())
        if i % 54==10:
            s11+=np.array(c[0])
            print('-------------------------conv11--------------------------------'.format(i=i%54+1))
            print(s11[0][0].tolist())
        if i % 54==11:
            s12+=np.array(c[0])
            print('-------------------------conv12--------------------------------'.format(i=i%54+1))
            print(s12[0][0].tolist())
        if i % 54==12:
            s13+=np.array(c[0])
            print('-------------------------conv13--------------------------------'.format(i=i%54+1))
            print(s13[0][0].tolist())
        if i % 54 == 13:
            s14 += np.array(c[0])
            print('-------------------------conv14--------------------------------'.format(i=i % 54 + 1))
            print(s14[0][0].tolist())
        if i % 54 == 14:
            s15+= np.array(c[0])
            print('-------------------------conv15--------------------------------'.format(i=i % 54 + 1))
            print(s15[0][0].tolist())
        if i % 54 == 15:
            s16+= np.array(c[0])
            print('-------------------------conv16--------------------------------'.format(i=i % 54 + 1))
            print(s16[0][0].tolist())
        if i % 54 == 16:
            s17+= np.array(c[0])
            print('-------------------------conv17--------------------------------'.format(i=i % 54 + 1))
            print(s17[0][0].tolist())
        if i % 54 == 17:
            s18+= np.array(c[0])
            print('-------------------------conv18--------------------------------'.format(i=i % 54 + 1))
            print(s18[0][0].tolist())
        if i % 54 == 18:
            s19+= np.array(c[0])
            print('-------------------------conv19--------------------------------'.format(i=i % 54 + 1))
            print(s19[0][0].tolist())
        if i % 54 == 19:
            s20+= np.array(c[0])
            print('-------------------------conv20--------------------------------'.format(i=i % 54 + 1))
            print(s20[0][0].tolist())
        if i % 54 == 20:
            s21+= np.array(c[0])
            print('-------------------------conv21--------------------------------'.format(i=i % 54 + 1))
            print(s21[0][0].tolist())
        if i % 54 == 21:
            s22+= np.array(c[0])
            print('-------------------------conv22--------------------------------'.format(i=i % 54 + 1))
            print(s22[0][0].tolist())
        if i % 54 == 22:
            s23 += np.array(c[0])
            print('-------------------------conv23--------------------------------'.format(i=i % 54 + 1))
            print(s23[0][0].tolist())
        if i % 54 == 23:
            s24 += np.array(c[0])
            print('-------------------------conv24--------------------------------'.format(i=i % 54 + 1))
            print(s24[0][0].tolist())
        if i % 54 == 24:
            s25+= np.array(c[0])
            print('-------------------------conv25--------------------------------'.format(i=i % 54 + 1))
            print(s25[0][0].tolist())
        if i % 54 == 25:
            s26 += np.array(c[0])
            print('-------------------------conv26--------------------------------'.format(i=i % 54 + 1))
            print(s26[0][0].tolist())
        if i % 54 == 26:
            s27 += np.array(c[0])
            print('-------------------------conv27--------------------------------'.format(i=i % 54 + 1))
            print(s27[0][0].tolist())
        if i % 54 == 27:
            s28 += np.array(c[0])
            print('-------------------------conv28--------------------------------'.format(i=i % 54 + 1))
            print(s28[0][0].tolist())
        if i % 54 == 28:
            s29 += np.array(c[0])
            print('-------------------------conv29--------------------------------'.format(i=i % 54 + 1))
            print(s29[0][0].tolist())
        if i % 54 == 29:
            s30 += np.array(c[0])
            print('-------------------------conv30--------------------------------'.format(i=i % 54 + 1))
            print(s30[0][0].tolist())
        if i % 54 == 30:
            s31 += np.array(c[0])
            print('-------------------------conv31--------------------------------'.format(i=i % 54 + 1))
            print(s31[0][0].tolist())
        if i % 54 == 31:
            s32 += np.array(c[0])
            print('-------------------------conv32--------------------------------'.format(i=i % 54 + 1))
            print(s32[0][0].tolist())
        if i % 54 == 32:
            s33 += np.array(c[0])
            print('-------------------------conv33--------------------------------'.format(i=i % 54 + 1))
            print(s33[0][0].tolist())
        if i % 54 == 33:
            s34 += np.array(c[0])
            print('-------------------------conv34--------------------------------'.format(i=i % 54 + 1))
            print(s34[0][0].tolist())
        if i % 54 == 34:
            s35 += np.array(c[0])
            print('-------------------------conv35--------------------------------'.format(i=i % 54 + 1))
            print(s35[0][0].tolist())
        if i % 54 == 35:
            s36 += np.array(c[0])
            print('-------------------------conv36--------------------------------'.format(i=i % 54 + 1))
            print(s36[0][0].tolist())
        if i % 54 == 36:
            s37 += np.array(c[0])
            print('-------------------------conv37--------------------------------'.format(i=i % 54 + 1))
            print(s37[0][0].tolist())
        if i % 54 == 37:
            s38 += np.array(c[0])
            print('-------------------------conv38--------------------------------'.format(i=i % 54 + 1))
            print(s38[0][0].tolist())
        if i % 54 == 38:
            s39 += np.array(c[0])
            print('-------------------------conv39--------------------------------'.format(i=i % 54 + 1))
            print(s39[0][0].tolist())
        if i % 54 == 39:
            s40 += np.array(c[0])
            print('-------------------------conv40--------------------------------'.format(i=i % 54 + 1))
            print(s40[0][0].tolist())
        if i % 54 == 40:
            s41 += np.array(c[0])
            print('-------------------------conv41--------------------------------'.format(i=i % 54 + 1))
            print(s41[0][0].tolist())
        if i % 54 == 41:
            s42 += np.array(c[0])
            print('-------------------------conv42--------------------------------'.format(i=i % 54 + 1))
            print(s42[0][0].tolist())
        if i % 54 == 42:
            s43 += np.array(c[0])
            print('-------------------------conv43--------------------------------'.format(i=i % 54 + 1))
            print(s43[0][0].tolist())
        if i % 54 == 43:
            s44 += np.array(c[0])
            print('-------------------------conv44--------------------------------'.format(i=i % 13 + 1))
            print(s44[0][0].tolist())
        if i % 54 == 44:
            s45 += np.array(c[0])
            print('-------------------------conv45--------------------------------'.format(i=i % 54 + 1))
            print(s45[0][0].tolist())
        if i % 54 == 45:
            s46 += np.array(c[0])
            print('-------------------------conv46--------------------------------'.format(i=i % 54 + 1))
            print(s46[0][0].tolist())
        if i % 54 == 46:
            s47 += np.array(c[0])
            print('-------------------------conv47--------------------------------'.format(i=i % 54 + 1))
            print(s47[0][0].tolist())
        if i % 54 == 47:
            s48 += np.array(c[0])
            print('-------------------------conv48--------------------------------'.format(i=i % 54 + 1))
            print(s48[0][0].tolist())
        if i % 54 == 48:
            s49 += np.array(c[0])
            print('-------------------------conv49--------------------------------'.format(i=i % 54 + 1))
            print(s49[0][0].tolist())
        if i % 54 == 49:
            s50 += np.array(c[0])
            print('-------------------------conv50--------------------------------'.format(i=i % 54 + 1))
            print(s50[0][0].tolist())
        if i % 54 == 50:
            s51 += np.array(c[0])
            print('-------------------------conv51--------------------------------'.format(i=i % 54 + 1))
            print(s51[0][0].tolist())
        if i % 54 == 51:
            s52 += np.array(c[0])
            print('-------------------------conv52--------------------------------'.format(i=i % 54 + 1))
            print(s52[0][0].tolist())
        if i % 54 == 52:
            s53 += np.array(c[0])
            print('-------------------------conv53--------------------------------'.format(i=i % 54 + 1))
            print(s53[0][0].tolist())
        if i % 54 == 53:
            s54 += np.array(c[0])
            print('-------------------------conv54--------------------------------'.format(i=i % 54 + 1))
            print(s54[0][0].tolist())
        i=i+1
        return x * scale

class SCA(nn.Module):
    def __init__(self, gate_channels,groups,pool_types=['avg', 'max']):
        super(SCA, self).__init__()
        self.spatial_att = SpatialGroupEnhance(groups)
        self.channel_att = ChannelGate(gate_channels)

    def forward(self, x):
        x_out = self.spatial_att(x)
        x_out = self.channel_att(x_out)
        return x_out


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=4, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias