import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules import conv
from torch.nn.modules.utils import _pair
import math
import time

from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn import init

import diffdist

class FastDeconv(conv._ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,groups=1,bias=True, eps=1e-5, n_iter=5, momentum=0.1, block=64, sampling_stride=3,freeze=False,freeze_iter=100):
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.counter=0
        self.track_running_stats=True
        super(FastDeconv, self).__init__(
            in_channels, out_channels,  _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation),
            False, _pair(0), groups, bias, padding_mode='zeros')

        if block > in_channels:
            block = in_channels
        else:
            if in_channels%block!=0:
                block=math.gcd(block,in_channels)

        if groups>1:
            #grouped conv
            block=in_channels//groups

        self.block=block

        self.num_features = kernel_size**2 *block
        if groups==1:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_cov_isqrt', torch.eye(self.num_features))
        else:
            self.register_buffer('running_mean', torch.zeros(kernel_size ** 2 * in_channels))
            self.register_buffer('running_cov_isqrt', torch.eye(self.num_features).repeat(in_channels // block, 1, 1))

        self.sampling_stride=sampling_stride*stride
        self.counter=0
        self.freeze_iter=freeze_iter
        self.freeze=freeze

    def forward(self, x):
        N, C, H, W = x.shape
        B = self.block
        frozen=self.freeze and (self.counter>self.freeze_iter)
        if self.training and self.track_running_stats:
            self.counter+=1
            self.counter %= (self.freeze_iter * 10)

        if self.training and (not frozen):

            # 1. im2col: N x cols x pixels -> N*pixles x cols
            if self.kernel_size[0]>1:
                X = torch.nn.functional.unfold(x, self.kernel_size,self.dilation,self.padding,self.sampling_stride).transpose(1, 2).contiguous()
            else:
                #channel wise
                X = x.permute(0, 2, 3, 1).contiguous().view(-1, C)[::self.sampling_stride**2,:]

            if self.groups==1:
                # (C//B*N*pixels,k*k*B)
                X = X.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1, self.num_features)
            else:
                X=X.view(-1,X.shape[-1])

            # 2. subtract mean
            X_mean = X.mean(0)
            X = X - X_mean.unsqueeze(0)

            # 3. calculate COV, COV^(-0.5), then deconv
            if self.groups==1:
                #cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                Id=torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                cov = torch.addmm(beta=self.eps, input=Id, alpha=1. / X.shape[0], mat1=X.t(), mat2=X)
                deconv = isqrt_newton_schulz_autograd(cov, self.n_iter)
            else:
                X = X.view(-1, self.groups, self.num_features).transpose(0, 1)
                Id = torch.eye(self.num_features, dtype=X.dtype, device=X.device).expand(self.groups, self.num_features, self.num_features)
                cov = torch.baddbmm(beta=self.eps, input=Id, alpha=1. / X.shape[1],  mat1=X.transpose(1, 2),  mat2=X)

                deconv = isqrt_newton_schulz_autograd_batch(cov, self.n_iter)

            if self.track_running_stats:
                self.running_mean.mul_(1 - self.momentum)
                self.running_mean.add_(X_mean.detach() * self.momentum)
                # track stats for evaluation
                self.running_cov_isqrt.mul_(1 - self.momentum)
                self.running_cov_isqrt.add_(deconv.detach() * self.momentum)

        else:
            X_mean = self.running_mean
            deconv = self.running_cov_isqrt

        #4. X * deconv * conv = X * (deconv * conv)
        if self.groups==1:
            w = self.weight.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1,self.num_features) @ deconv
            b = self.bias - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)
            w = w.view(-1, C // B, self.num_features).transpose(1, 2).contiguous()
        else:
            w = self.weight.view(C//B, -1,self.num_features)@deconv
            b = self.bias - (w @ (X_mean.view( -1,self.num_features,1))).view(self.bias.shape)

        w = w.view(self.weight.shape)
        x= F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

        return x



class FastDeconvTransposed(conv._ConvTransposeNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,bias=True, dilation=1, eps=1e-5, n_iter=5, momentum=0.1, block=64, sampling_stride=3):


        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.counter=0
        self.track_running_stats=True
        super(FastDeconvTransposed, self).__init__(
            in_channels, out_channels, _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation),
            True, _pair(output_padding), groups, bias, padding_mode='zeros')

        if block > in_channels:
            block = in_channels
        else:
            if in_channels%block!=0:
                block=math.gcd(block,in_channels)

        if groups>1:
            #grouped conv
            block=in_channels//groups

        self.block=block

        self.num_features = kernel_size**2 *block
        if groups==1:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_cov_isqrt', torch.eye(self.num_features))
        else:
            self.register_buffer('running_mean', torch.zeros(kernel_size ** 2 * in_channels))
            self.register_buffer('running_cov_isqrt', torch.eye(self.num_features).repeat(in_channels // block, 1, 1))

        self.sampling_stride=sampling_stride*stride
        self.counter=0

    def forward(self, x,output_size=None):
        N, C, H, W = x.shape
        B = self.block
        
        if self.training and self.track_running_stats:
            self.counter+=1
        
        if self.training:

            # 1. im2col: N x cols x pixels -> N*pixles x cols
            if self.kernel_size[0]>1:
                #the adjoint of a conv operation is a full correlation operation. So pad first.
                padding=(self.padding[0]+self.kernel_size[0]-1,self.padding[1]+self.kernel_size[1]-1)
                X = torch.nn.functional.unfold(x, self.kernel_size,self.dilation,self.padding,self.sampling_stride).transpose(1, 2).contiguous()
            else:
                #channel wise
                X = x.permute(0, 2, 3, 1).contiguous().view(-1, C)[::self.sampling_stride**2,:]

            if self.groups==1:
                # (C//B*N*pixels,k*k*B)
                X = X.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1, self.num_features)
            else:
                X=X.view(-1,X.shape[-1])

            # 2. subtract mean
            X_mean = X.mean(0)
            X = X - X_mean.unsqueeze(0)

            # 3. calculate COV, COV^(-0.5), then deconv
            if self.groups==1:
                #cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                Id=torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                cov = torch.addmm(beta=self.eps, input=Id, alpha=1. / X.shape[0], mat1=X.t(), mat2=X)
                deconv = isqrt_newton_schulz_autograd(cov, self.n_iter)
            else:
                X = X.view(-1, self.groups, self.num_features).transpose(0, 1)
                Id = torch.eye(self.num_features, dtype=X.dtype, device=X.device).expand(self.groups, self.num_features, self.num_features)
                cov = torch.baddbmm(beta=self.eps, input=Id, alpha=1. / X.shape[1], mat1=X.transpose(1, 2), mat2=X)

                deconv = isqrt_newton_schulz_autograd_batch(cov, self.n_iter)

            if self.track_running_stats:
                self.running_mean.mul_(1 - self.momentum)
                self.running_mean.add_(X_mean.detach() * self.momentum)
                # track stats for evaluation
                self.running_cov_isqrt.mul_(1 - self.momentum)
                self.running_cov_isqrt.add_(deconv.detach() * self.momentum)

        else:
            X_mean = self.running_mean
            deconv = self.running_cov_isqrt

        #4. X * deconv * conv = X * (deconv * conv)

        # this is to use conv2d to calculate conv_tansposed2d
        weight=torch.flip(self.weight,[2,3])

        if self.groups==1:
            w = weight.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1,self.num_features) @ deconv
            b = self.bias - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)
            w = w.view(-1, C // B, self.num_features).transpose(1, 2).contiguous()
        else:
            w = self.weight.view(C//B, -1,self.num_features)@deconv
            b = self.bias - (w @ (X_mean.view( -1,self.num_features,1))).view(self.bias.shape)

        w = w.view(self.weight.shape)

        w=torch.flip(w.view(weight.shape),[2,3])

        output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size)
        x = F.conv_transpose2d(
            x, w, b, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

        return x




class Delinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, eps=1e-5, n_iter=5, momentum=0.1, block=512):
        super(Delinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()



        if block > in_features:
            block = in_features
        else:
            if in_features%block!=0:
                block=math.gcd(block,in_features)
                print('block size set to:', block)
        self.block = block
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(self.block))
        self.register_buffer('running_cov_isqrt', torch.eye(self.block))


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        if self.training:

            # 1. reshape
            X=input.view(-1, self.block)

            # 2. subtract mean
            X_mean = X.mean(0)
            X = X - X_mean.unsqueeze(0)
            self.running_mean.mul_(1 - self.momentum)
            self.running_mean.add_(X_mean.detach() * self.momentum)

            # 3. calculate COV, COV^(-0.5), then deconv
            # cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
            Id = torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
            cov = torch.addmm(beta=self.eps, input=Id, alpha=1. / X.shape[0], mat1=X.t(), mat2=X)
            deconv = isqrt_newton_schulz_autograd(cov, self.n_iter)
            # track stats for evaluation
            self.running_cov_isqrt.mul_(1 - self.momentum)
            self.running_cov_isqrt.add_(deconv.detach() * self.momentum)

        else:
            X_mean = self.running_mean
            deconv = self.running_cov_isqrt

        w = self.weight.view(-1, self.block) @ deconv
        if self.bias is None:
            b = - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)
        else:
            b = self.bias - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)

        w = w.view(self.weight.shape)
        return F.linear(input, w, b)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )





class NormalizedDelinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, eps=1e-5, n_iter=5, momentum=0.1, block=512,norm_type='layernorm',sync=False):
        super(NormalizedDelinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        if block > in_features:
            block = in_features
        else:
            if in_features%block!=0:
                block=math.gcd(block,in_features)
                print('block size set to:', block)
        self.block = block
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(self.block))
        self.register_buffer('running_cov_isqrt', torch.eye(self.block))
        self.norm_type=norm_type
        self.sync=sync
        self.process_group=None

    def _specify_process_group(self, process_group):
        self.process_group = process_group

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if input.numel()==0:
            return input
        if self.norm_type=='layernorm':
            mean = input.mean(-1,keepdim=True)
            std = input.std(-1,keepdim=True)
            input = (input - mean) / (std + self.eps)
        elif self.norm_type=='groupnorm':
            N,C=input.shape
            G=min(16,self.block)
            x=input.reshape(N,G,-1)
            #x=input.view(N,-1,G).transpose(1,2)
            mean = x.mean(-1,keepdim=True)
            std = x.std(-1,keepdim=True)
            x = (x - mean) / (std + self.eps)
            #x=x.transpose(1,2)
            input=x.view(N,C)
            
            
        if self.training:
            # 1. reshape
            X=input.view(-1, self.block)

            # 2. calculate mean,cov,cov_isqrt

            if self.sync:
                process_group = self.process_group
                world_size = 1
                if not self.process_group:
                    process_group = torch.distributed.group.WORLD


                N = X.shape[0]
                X_mean = X.mean(0)
                XX_mean = X.t()@X/N 

                if torch.distributed.is_initialized():
                    world_size = torch.distributed.get_world_size(process_group)
                    X_mean_list=[torch.empty_like(X_mean) for k in range(world_size)]
                    #torch.distributed.all_gather(X_mean_list, X_mean, process_group, async_op=False)
                    X_mean_list = diffdist.functional.all_gather(X_mean_list, X_mean)

                    XX_mean_list=[torch.empty_like(XX_mean) for k in range(world_size)]
                    #torch.distributed.all_gather(XX_mean_list, XX_mean, process_group, async_op=False)
                    XX_mean_list = diffdist.functional.all_gather(XX_mean_list, XX_mean)

                    X_mean=torch.stack(X_mean_list).mean(0)
                    XX_mean=torch.stack(XX_mean_list).mean(0)

                cov= XX_mean- X_mean.unsqueeze(1) @X_mean.unsqueeze(0)
                Id = torch.eye(cov.shape[1], dtype=cov.dtype, device=cov.device)
                cov_isqrt = isqrt_newton_schulz_autograd(cov+self.eps*Id, self.n_iter)
                
            else:
                
                X_mean = X.mean(0)
                X = X - X_mean.unsqueeze(0)
                # cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                Id = torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                cov = torch.addmm(beta=self.eps, input=Id, alpha=1. / X.shape[0], mat1=X.t(), mat2=X)
                cov_isqrt = isqrt_newton_schulz_autograd(cov, self.n_iter)
        
            # track stats for evaluation
            self.running_mean.mul_(1 - self.momentum)
            self.running_mean.add_(X_mean.detach() * self.momentum)
            self.running_cov_isqrt.mul_(1 - self.momentum)
            self.running_cov_isqrt.add_(cov_isqrt.detach() * self.momentum)
                
        else:

            X_mean = self.running_mean
            cov_isqrt = self.running_cov_isqrt

        #3. decorrelate via weight transform

        w = self.weight.view(-1, self.block) @ cov_isqrt
        if self.bias is None:
            b = - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)
        else:
            b = self.bias - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)

        w = w.view(self.weight.shape)
        return F.linear(input, w, b)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class NormalizedDeconv(conv._ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,groups=1,bias=True, eps=1e-5, n_iter=5, momentum=0.1, block=64, sampling_stride=3,freeze=False,freeze_iter=100,norm_type='layernorm',sync=False):
  
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.counter=0
        self.track_running_stats=True
        super(NormalizedDeconv, self).__init__(
            in_channels, out_channels,  _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation),
            False, _pair(0), groups, bias, padding_mode='zeros')

        if block > in_channels:
            block = in_channels
        else:
            if in_channels%block!=0:
                block=math.gcd(block,in_channels)

        if groups>1:
            #grouped conv
            block=in_channels//groups

        self.block=block

        self.num_features = kernel_size**2 *block
        if groups==1:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_cov_isqrt', torch.eye(self.num_features))
        else:
            self.register_buffer('running_mean', torch.zeros(kernel_size ** 2 * in_channels))
            self.register_buffer('running_cov_isqrt', torch.eye(self.num_features).repeat(in_channels // block, 1, 1))

        self.sampling_stride=sampling_stride*stride
        self.counter=0
        self.freeze_iter=freeze_iter
        self.freeze=freeze
        self.norm_type=norm_type        
        self.sync=sync
        self.process_group=None
        
    def _specify_process_group(self, process_group):
        self.process_group = process_group
    
    def forward(self, x):

        if x.numel()==0:
            return x

        N, C, H, W = x.shape
        B = self.block


        if self.norm_type=='layernorm':
            x=x.reshape(N,-1)
            mean = x.mean(-1,keepdim=True)
            std = x.std(-1,keepdim=True)
            x = (x - mean) / (std + self.eps)
            x=x.reshape(N,C,H,W)

        elif self.norm_type=='groupnorm':
            G=min(16,B)
            x=x.reshape(N,G,-1)
            #x=x.view(N,C//G,G,-1).transpose(1,2).reshape(N,G,-1)
            mean = x.mean(-1,keepdim=True)
            std = x.std(-1,keepdim=True)
            x = (x - mean) / (std + self.eps)
            #x=x.reshape(N,G,C//G,H,W).transpose(1,2)
            x=x.reshape(N,C,H,W)


        if self.training:
            self.counter+=1
            frozen=self.freeze and (self.counter% (self.freeze_iter * 10) >self.freeze_iter)

        if self.training and (not frozen):

            # 1. im2col: N x cols x pixels -> N*pixles x cols
            if self.kernel_size[0]>1:
                X = torch.nn.functional.unfold(x, self.kernel_size,self.dilation,self.padding,self.sampling_stride).transpose(1, 2).contiguous()
            else:
                #channel wise
                X = x.permute(0, 2, 3, 1).contiguous().view(-1, C)[::self.sampling_stride**2,:]

            if self.groups==1:
                # (C//B*N*pixels,k*k*B)
                X = X.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1, self.num_features)
            else:
                X=X.view(-1,X.shape[-1])

            # 2. calculate mean,cov,cov_isqrt

            if self.sync:
                process_group = self.process_group
                world_size = 1
                if not self.process_group:
                    process_group = torch.distributed.group.WORLD

                X_mean = X.mean(0)

                if self.groups==1:
                    M = X.shape[0]
                    XX_mean = X.t()@X/M 
                else:
                    M = X.shape[1]
                    XX_mean = X.transpose(1,2)@X/M 
                    
                if torch.distributed.is_initialized():
                    world_size = torch.distributed.get_world_size(process_group)
                    X_mean_list=[torch.empty_like(X_mean) for k in range(world_size)]
                    #torch.distributed.all_gather(X_mean_list, X_mean, process_group, async_op=False)
                    X_mean_list = diffdist.functional.all_gather(X_mean_list, X_mean)

                    XX_mean_list=[torch.empty_like(XX_mean) for k in range(world_size)]
                    #torch.distributed.all_gather(XX_mean_list, XX_mean, process_group, async_op=False)
                    XX_mean_list = diffdist.functional.all_gather(XX_mean_list, XX_mean)

                    X_mean=torch.stack(X_mean_list).mean(0)
                    XX_mean=torch.stack(XX_mean_list).mean(0)

                if self.groups==1:
                    cov= XX_mean- X_mean.unsqueeze(1) @X_mean.unsqueeze(0)
                    Id = torch.eye(cov.shape[1], dtype=cov.dtype, device=cov.device)
                    cov_isqrt = isqrt_newton_schulz_autograd(cov+self.eps*Id, self.n_iter)
                else:
                    cov= (XX_mean- (X_mean.unsqueeze(2)) @X_mean.unsqueeze(1))
                    Id = torch.eye(self.num_features, dtype=cov.dtype, device=cov.device).expand(self.groups, self.num_features, self.num_features)
                    cov_isqrt = isqrt_newton_schulz_autograd_batch(cov+self.eps*Id, self.n_iter)

            else:
            
                X_mean = X.mean(0)
                X = X - X_mean.unsqueeze(0)

                if self.groups==1:
                    #cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                    Id=torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                    cov = torch.addmm(beta=self.eps, input=Id, alpha=1. / X.shape[0], mat1=X.t(), mat2=X)
                    cov_isqrt = isqrt_newton_schulz_autograd(cov, self.n_iter)
                else:
                    X = X.view(-1, self.groups, self.num_features).transpose(0, 1)
                    Id = torch.eye(self.num_features, dtype=X.dtype, device=X.device).expand(self.groups, self.num_features, self.num_features)
                    cov = torch.baddbmm(beta=self.eps, input=Id, alpha=1. / X.shape[1],  mat1=X.transpose(1, 2),  mat2=X)
                    cov_isqrt = isqrt_newton_schulz_autograd_batch(cov, self.n_iter)

            # track stats for evaluation.
            self.running_mean.mul_(1 - self.momentum)                
            self.running_mean.add_(X_mean.detach() * self.momentum)
            self.running_cov_isqrt.mul_(1 - self.momentum)
            self.running_cov_isqrt.add_(cov_isqrt.detach() * self.momentum)

        else:
            X_mean = self.running_mean
            cov_isqrt = self.running_cov_isqrt

        #3. X * deconv * conv = X * (deconv * conv)
        if self.groups==1:
            w = self.weight.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1,self.num_features) @ cov_isqrt
            b = self.bias - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)
            w = w.view(-1, C // B, self.num_features).transpose(1, 2).contiguous()
        else:
            w = self.weight.view(C//B, -1,self.num_features)@cov_isqrt
            b = self.bias - (w @ (X_mean.view( -1,self.num_features,1))).view(self.bias.shape)

        w = w.view(self.weight.shape)
        x= F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

        return x



class NormalizedDeconvTransposed(conv._ConvTransposeNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,bias=True, dilation=1, eps=1e-5, n_iter=5, momentum=0.1, block=64, sampling_stride=3,norm_type='layernorm',sync=False):


        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.counter=0
        self.track_running_stats=True
        super(NormalizedDeconvTransposed, self).__init__(
            in_channels, out_channels, _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation),
            True, _pair(output_padding), groups, bias, padding_mode='zeros')

        if block > in_channels:
            block = in_channels
        else:
            if in_channels%block!=0:
                block=math.gcd(block,in_channels)

        if groups>1:
            #grouped conv
            block=in_channels//groups

        self.block=block

        self.num_features = kernel_size**2 *block
        if groups==1:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_cov_isqrt', torch.eye(self.num_features))
        else:
            self.register_buffer('running_mean', torch.zeros(kernel_size ** 2 * in_channels))
            self.register_buffer('running_cov_isqrt', torch.eye(self.num_features).repeat(in_channels // block, 1, 1))

        self.sampling_stride=sampling_stride*stride
        self.counter=0
        self.norm_type = norm_type
        self.sync=sync
        self.process_group=None

    def _specify_process_group(self, process_group):
        self.process_group = process_group

    def forward(self, x,output_size=None):
        if x.numel()==0:
            return x
        N, C, H, W = x.shape
        B = self.block
        
        if self.norm_type=='layernorm':
            x=x.reshape(N,-1)
            mean = x.mean(-1,keepdim=True)
            std = x.std(-1,keepdim=True)
            x = (x - mean) / (std + self.eps)
            x=x.reshape(N,C,H,W)

        elif self.norm_type=='groupnorm':
            G=min(16,B)
            x=x.reshape(N,G,-1)
            #x=x.view(N,C//G,G,-1).transpose(1,2).reshape(N,G,-1)
            mean = x.mean(-1,keepdim=True)
            std = x.std(-1,keepdim=True)
            x = (x - mean) / (std + self.eps)
            #x=x.reshape(N,G,C//G,H,W).transpose(1,2)
            x=x.reshape(N,C,H,W)

        if self.training and self.track_running_stats:
            self.counter+=1

            #1. im2col: N x cols x pixels -> N*pixles x cols
            if self.kernel_size[0]>1:
                #the adjoint of a conv operation is a full correlation operation. So pad first.
                padding=(self.padding[0]+self.kernel_size[0]-1,self.padding[1]+self.kernel_size[1]-1)
                X = torch.nn.functional.unfold(x, self.kernel_size,self.dilation,self.padding,self.sampling_stride).transpose(1, 2).contiguous()
            else:
                #channel wise
                X = x.permute(0, 2, 3, 1).contiguous().view(-1, C)[::self.sampling_stride**2,:]

            if self.groups==1:
                # (C//B*N*pixels,k*k*B)
                X = X.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1, self.num_features)
            else:
                X=X.view(-1,X.shape[-1])


            if self.sync:
                process_group = self.process_group
                world_size = 1
                if not self.process_group:
                    process_group = torch.distributed.group.WORLD
            

                # 2. calculate mean,cov,cov_isqrt

                X_mean = X.mean(0)

                if self.groups==1:
                    M = X.shape[0]
                    XX_mean = X.t()@X/M 
                else:
                    M = X.shape[1]
                    XX_mean = X.transpose(1,2)@X/M 

                if torch.distributed.is_initialized():
                    world_size = torch.distributed.get_world_size(process_group)
                    X_mean_list=[torch.empty_like(X_mean) for k in range(world_size)]
                    #torch.distributed.all_gather(X_mean_list, X_mean, process_group, async_op=False)
                    X_mean_list = diffdist.functional.all_gather(X_mean_list, X_mean)

                    XX_mean_list=[torch.empty_like(XX_mean) for k in range(world_size)]
                    #torch.distributed.all_gather(XX_mean_list, XX_mean, process_group, async_op=False)
                    XX_mean_list = diffdist.functional.all_gather(XX_mean_list, XX_mean)

                    X_mean=torch.stack(X_mean_list).mean(0)
                    XX_mean=torch.stack(XX_mean_list).mean(0)


                if self.groups==1:
                    cov= XX_mean- X_mean.unsqueeze(1) @X_mean.unsqueeze(0)
                    Id = torch.eye(cov.shape[1], dtype=cov.dtype, device=cov.device)
                    cov_isqrt = isqrt_newton_schulz_autograd(cov+self.eps*Id, self.n_iter)
                else:
                    cov= (XX_mean- (X_mean.unsqueeze(2)) @X_mean.unsqueeze(1))
                    Id = torch.eye(self.num_features, dtype=cov.dtype, device=cov.device).expand(self.groups, self.num_features, self.num_features)
                    cov_isqrt = isqrt_newton_schulz_autograd_batch(cov+self.eps*Id, self.n_iter)


            else:

                X_mean = X.mean(0)
                X = X - X_mean.unsqueeze(0)

                if self.groups==1:
                    #cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                    Id=torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                    cov = torch.addmm(beta=self.eps, input=Id, alpha=1. / X.shape[0], mat1=X.t(), mat2=X)
                    cov_isqrt = isqrt_newton_schulz_autograd(cov, self.n_iter)
                else:
                    X = X.view(-1, self.groups, self.num_features).transpose(0, 1)
                    Id = torch.eye(self.num_features, dtype=X.dtype, device=X.device).expand(self.groups, self.num_features, self.num_features)
                    cov = torch.baddbmm(beta=self.eps, input=Id, alpha=1. / X.shape[1], mat1=X.transpose(1, 2), mat2=X)
                    cov_isqrt = isqrt_newton_schulz_autograd_batch(cov, self.n_iter)


            self.running_mean.mul_(1 - self.momentum)
            self.running_mean.add_(X_mean.detach() * self.momentum)
            
            self.running_cov_isqrt.mul_(1 - self.momentum)
            self.running_cov_isqrt.add_(cov_isqrt.detach() * self.momentum)


        else:
            X_mean = self.running_mean
            cov_isqrt = self.running_cov_isqrt

        #3. X * deconv * conv = X * (deconv * conv)

        # this is to use conv2d to calculate conv_tansposed2d
        weight=torch.flip(self.weight,[2,3])

        if self.groups==1:
            w = weight.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1,self.num_features) @ cov_isqrt
            b = self.bias - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)
            w = w.view(-1, C // B, self.num_features).transpose(1, 2).contiguous()
        else:
            w = self.weight.view(C//B, -1,self.num_features)@cov_isqrt
            b = self.bias - (w @ (X_mean.view( -1,self.num_features,1))).view(self.bias.shape)

        w = w.view(self.weight.shape)

        w=torch.flip(w.view(weight.shape),[2,3])

        output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size)
        x = F.conv_transpose2d(
            x, w, b, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

        return x



def isqrt_newton_schulz_autograd(A, numIters,norm='norm',method='denman_beavers'):
    dim = A.shape[0]

    if norm=='norm':
        normA=A.norm()
    else:
        normA=A.trace()

    I = torch.eye(dim, dtype=A.dtype, device=A.device)
    Y = A.div(normA)

    Z = torch.eye(dim, dtype=A.dtype, device=A.device)

    if method=='denman_beavers':
        for i in range(numIters):
            #T = 0.5*(3.0*I - Z@Y)
            T=torch.addmm(beta=1.5, input=I, alpha=-0.5, mat1=Z, mat2=Y)

            Y = Y.mm(T)
            Z = T.mm(Z)
    else:
        for i in range(numIters):
            #Z =  1.5 * Z - 0.5* Z@ Z @ Z @ Y
            Z = torch.addmm(beta=1.5, input=Z, alpha=-0.5, mat1=torch.matrix_power(Z, 3), mat2=Y)
    #A_sqrt = Y* torch.sqrt(normA)
    A_isqrt =Z/ torch.sqrt(normA)
    return A_isqrt


def isqrt_newton_schulz_autograd_batch(A, numIters):
    batchSize,dim,_ = A.shape
    normA=A.view(batchSize, -1).norm(2, 1).view(batchSize, 1, 1)
    Y = A.div(normA)
    I = torch.eye(dim,dtype=A.dtype,device=A.device).unsqueeze(0).expand_as(A)
    Z = torch.eye(dim,dtype=A.dtype,device=A.device).unsqueeze(0).expand_as(A)

    for i in range(numIters):
        T = 0.5*(3.0*I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    #A_sqrt = Y*torch.sqrt(normA)
    A_isqrt = Z / torch.sqrt(normA)

    return A_isqrt





"""
#this does not work

def transform_bn_to_nd(state_dict):
    #this function is assuming each bn is paired with a conv in the following straightforward way
    print('Transforming batch normalization to network deconvolution in a pretrained model.')
    original_keys = sorted(state_dict.keys())
    bn_keys=[k for k in original_keys if ('bn' in k  or 'downsample.1' in k) ]

    for bn_bias in bn_keys:#this is assuming bias is used in bn

        if 'bias' in bn_bias:
            conv_bias=bn_bias.replace("bn", "conv").replace("downsample.1", "downsample.0")#bn in downsampling
            conv_weight=bn_bias.replace("bias", "weight").replace("bn", "conv").replace("downsample.1", "downsample.0")#bn in downsampling

            bn_running_mean=bn_bias.replace("bias", "running_mean")
            bn_running_var=bn_bias.replace("bias", "running_var")
            bn_weight=bn_bias.replace("bias", "weight")

            if bn_running_mean in state_dict:
                mu=state_dict[bn_running_mean]
            else:
                mu=0.
            if bn_running_var in state_dict:
                rstd=torch.rsqrt(state_dict[bn_running_var])
            else:
                rstd=1.

            print(conv_weight, state_dict[conv_weight].shape,' scaled with ', bn_weight, state_dict[bn_weight].shape)

            state_dict[conv_weight]=state_dict[conv_weight]*(rstd*state_dict[bn_weight]).view(-1,1,1,1)
            if conv_bias in state_dict:
                print(conv_bias, state_dict[conv_bias].shape,' replaced with ', bn_bias, state_dict[bn_bias].shape)
                mu=mu-state_dict[conv_bias]
            else:
                print(conv_bias, 'created with ', bn_bias, state_dict[bn_bias].shape)

                state_dict[conv_bias]=state_dict[bn_bias]-mu*rstd*state_dict[bn_weight]

            
    for bn_key in bn_keys:
        del state_dict[bn_key]
    return state_dict

"""