import torch
import torch.nn as nn
import torch.nn.functional as F

'''
NeRF神经网络部分仅仅是一个8层的MLP，网络部分很简单，就是输入
'''

class NeRF_Model(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        '''
        :param D: MLP层数
        :param W: 隐藏层的宽度
        :param input_ch: 输入位置向量，xyz
        :param input_ch_views: 输入视角向量，这里与原论文中的二维向量不同，这里也是采用的三维向量
        :param output_ch:
        :param skips:
        :param use_viewdirs: 输入是否包含方向信息，也就是以5D向量作为输入，否则仅由位置信息作为3D输入
        '''
        super(NeRF_Model, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.output_ch = output_ch
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if self.use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        '''
        :param x:采样点或者采样点+视角方向，shape为[N_rays*N_samples, 63]或者[N_rays*N_samples, 90]
        :return:
        '''
        # input_pts: [N_rays*N_samples, 63]
        # input_views: [N_rays*N_samples, 27]
        if self.use_viewdirs:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        else:
            input_pts = x
        h = input_pts
        # 先经过8层全连接层
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        # 最后的输出层，如果加入了视角输入的话，需要额外经过一些网络层
        if self.use_viewdirs:
            # [N_rays*N_samples, 1]
            alpha = self.alpha_linear(h)
            # [N_rays*N_samples, 256]
            feature = self.feature_linear(h)
            # [N_rays*N_samples, 283]
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                # [N_rays*N_samples, 283] --> [N_rays*N_samples, 128]
                h = self.views_linears[i](h)
                h = F.relu(h)
            # [N_rays*N_samples, 128] --> [N_rays*N_samples, 3]
            rgb = self.rgb_linear(h)
            # [N_rays*N_samples, 4]
            outputs = torch.cat([rgb, alpha], -1)

        else:
            # [N_rays*N_samples, 4]
            outputs = self.output_linear(h)

        return outputs

