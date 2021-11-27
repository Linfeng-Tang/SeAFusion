# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)

class DenseBlock(nn.Module):
    def __init__(self,channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)
    def forward(self,x):
        x=torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x

class RGBD(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(RGBD, self).__init__()
        self.dense =DenseBlock(in_channels)
        self.convdown=Conv1(3*in_channels,out_channels)
        self.sobelconv=Sobelxy(in_channels)
        self.convup =Conv1(in_channels,out_channels)
    def forward(self,x):
        x1=self.dense(x)
        x1=self.convdown(x1)
        x2=self.sobelconv(x)
        x2=self.convup(x2)
        return F.leaky_relu(x1+x2,negative_slope=0.1)

class FusionNet(nn.Module):
    def __init__(self, output):
        super(FusionNet, self).__init__()
        vis_ch = [16,32,48]
        inf_ch = [16,32,48]
        output=1
        self.vis_conv=ConvLeakyRelu2d(1,vis_ch[0])
        self.vis_rgbd1=RGBD(vis_ch[0], vis_ch[1])
        self.vis_rgbd2 = RGBD(vis_ch[1], vis_ch[2])
        # self.vis_rgbd3 = RGBD(vis_ch[2], vis_ch[3])
        self.inf_conv=ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = RGBD(inf_ch[0], inf_ch[1])
        self.inf_rgbd2 = RGBD(inf_ch[1], inf_ch[2])
        # self.inf_rgbd3 = RGBD(inf_ch[2], inf_ch[3])
        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2]+inf_ch[2], vis_ch[1]+vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1]+inf_ch[1], vis_ch[0]+inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnTanh2d(vis_ch[0], output)
    def forward(self, image_vis,image_ir):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        x_inf_origin = image_ir
        # encode
        x_vis_p=self.vis_conv(x_vis_origin)
        x_vis_p1=self.vis_rgbd1(x_vis_p)
        x_vis_p2=self.vis_rgbd2(x_vis_p1)
        # x_vis_p3=self.vis_rgbd3(x_vis_p2)

        x_inf_p=self.inf_conv(x_inf_origin)
        x_inf_p1=self.inf_rgbd1(x_inf_p)
        x_inf_p2=self.inf_rgbd2(x_inf_p1)
        # x_inf_p3=self.inf_rgbd3(x_inf_p2)
        # decode
        x=self.decode4(torch.cat((x_vis_p2,x_inf_p2),dim=1))
        # x=self.decode4(x)
        x=self.decode3(x)
        x=self.decode2(x)
        x=self.decode1(x)
        return x

def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(2,4,480,640).astype(np.float32))
    model = FusionNet(output=1)
    y = model(x)
    print('output shape:', y.shape)
    assert y.shape == (2,1,480,640), 'output shape (2,1,480,640) is expected!'
    print('test ok!')

if __name__ == '__main__':
    unit_test()
