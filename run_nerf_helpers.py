import glob
import time
import os

import cv2
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import scipy.signal

'''
位置编码
'''
class Embedder:
    def __init__(self, include_input=True, input_dim=3, multires=10, log_sampling=True):
        '''
        :param include_input: 位置编码是否加入自身
        :param input_dim: 输入维度
        :param multires: 位置编码数量，如果是输入三维xyz，那么一般是取10，如果输入是视角向量，一般取4
        :param log_sampling: 是否用指数位置编码
        '''
        self.include_input = include_input
        self.input_dim = input_dim
        self.multires = multires
        self.log_sampling = log_sampling
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dim
        out_dim = 0
        # 把input本身加入到位置编码中，xyz占3位
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.multires - 1
        N_freqs = self.multires
        # 采样间隔freq_bands: [N_freqs]
        if self.log_sampling:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)

        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)
        #
        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, input):
        '''
        :param input: [N_rays*N_samples,3],  N_rays条光线，每条光线N_samples个点，3代表xyz
        :return: 位置编码嵌入之后得到[N_rays*N_samples, 63]
        '''
        # 每经过一个fn就是把一个xyz编码了，shape为3，一共有20*3个再加一个原始的未经编码的input，所以cat起来之后一共是63维
        return torch.cat([fn(input) for fn in self.embed_fns], -1)


'''
根据相机内外参，从图片的二维像素获取对应三维空间的光线原点以及方向
此为tensor版本
'''
def get_rays(H, W, K, c2w, device):
    '''
    :param H: 图片的高
    :param W: 图片的宽
    :param K: 相机内参矩阵 [3,3]
    :param c2w: 相机坐标系到世界坐标系的转换矩阵 [3,4]，也叫做相机外参的逆矩阵
    :return:
    '''

    # i: [[0,...,0],...,[W-1,...,W-1]]
    # j: [[0,...,H-1],...,[0,...,H-1]]
    # 可能有个疑问是光线应该是穿过像素中心的(也就是i和j都要加0.5)，但是这里相当于穿过像素的左上角，作者解释在实验中发现这个不影响
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    # .t()表示转换为torch.Tensor
    i = i.t()
    j = j.t()
    # 这里相当于是像素坐标系转换为图像坐标系，然后构建[x, y, 3]，假设HW都为400，这里得到的shape是[400, 400, 3]
    # 这里得到的是从相机原点指向该像素的方向
    # 这里y轴和z轴要取-1是因为相机的投影变换这个过程用的是opencv坐标系，和nerf坐标系中的y、z轴方向相反，所以要取负值
    # 同样的在NeRF中相机是看向z轴负方向的，所以需要将z轴翻转
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    dirs = dirs.to(device)
    # 将光线的方向向量从相机坐标系转换到世界坐标系
    # dirs[..., np.newaxis, :]-->[400,400,1,3]，与[3,3]相乘之前会先broadcast到[400,400,3,3](先扩充一维也是为了broadcast)
    # 点乘结果[400,400,3]
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    # 将相机原点从相机坐标系转换到世界坐标系，这是所有光线的原点
    # expand: [3] --> [400, 400, 3]
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


'''
根据相机内外参，从图片的二维像素获取对应三维空间的光线原点以及方向
此为numpy版本，代码逻辑与get_rays()一致
'''
def get_rays_np(H, W, K, c2w):
    '''
    :param H: 图片的高
    :param W: 图片的宽
    :param K: 相机内参矩阵 [3,3]
    :param c2w: 相机坐标系到世界坐标系的转换矩阵 [3,4]，也叫做相机外参的逆矩阵
    :return:
    '''
    i, j = np.meshgrid(np.linspace(0, W-1, W, dtype=np.float32), np.linspace(0, H-1, H, dtype=np.float32))
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d

'''
用于face forward场景，将射线从相机空间转换到NDC空间，这对于处理无界场景比较有用，
因为它允许NeRF模型在一个标准化的空间内进行渲染和学习，从而提高效率和稳定性。通过这种转换，射线可以在一个有限的、标准化的空间内进行处理，
即使它们原本代表远离相机的点
'''
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # 首先计算每条射线与近平面的交点
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d
    # 基于相机透视投影原理将射线的原点和方向投影到NDC空间
    # (o0,o1,o2)为计算更新后的射线原点在NDC空间中的坐标
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]
    # (d0,d1,d2)为计算更新后的射线方向在NDC空间中的坐标
    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]
    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)
    return rays_o, rays_d


'''
在训练代码中调用的方法，包含了数据准备、输入网络、得到结果
'''
def render(H, W, K, args, model, embed_fn, embeddirs_fn, model_fine=None, return_raw=False, rays=None,c2w=None,
           near=0., far=1., c2w_staticcam=None, device=None, is_train=True):
    '''
    :param H: 图片H
    :param W: 图片W
    :param K: 相机内参矩阵
    :param args: 传入的超参
    :param model: 粗网络
    :param embed_fn: xyz的位置编码方法
    :param embeddirs_fn: 角度的位置编码方法
    :param model_fine: 精细网络
    :param return_raw: 传入render_rays方法中，如果为True，则返回值包含网络的原始输出，否则仅返回处理过后的输出
    :param rays: [2, N_rays, 3] 采样的光线,包含了光线的原点和方向，如果没提供的话就通过get_rays方法获取
    :param c2w: [3, 4] 相机坐标系到世界坐标系的转换矩阵
    :param near: [N_rays] 一条光线最近的点
    :param far: [N_rays] 一条光线最远的点
    :param c2w_staticcam: 此参数仅在use_viewdirs为True时生效，意思是用这个矩阵来代替c2w
    :param device: cuda
    :param is_train: 训练的话就加上扰动，测试的话就不加扰动
    :return:
      rgb_map: [N_rays, 3] 对每条光线预测的rgb
      disp_map: [N_rays] 视差图
      acc_map: [N_rays] 沿着一条光线累计叠加得到的透明度
      extras: 返回的其他输入，看render_rays()的其他输出，包括网络原始输出还有粗网络的输出
    '''
    if is_train:
        perturb = True
        raw_noise_std = 1e0
    else:
        perturb = False
        raw_noise_std = 0.
    if c2w is not None:
        # rays_o: [H, W, 3] rays_d: [H, W, 3]
        # 这种是特殊情况，需要渲染整张图片，一般是验证的时候
        rays_o, rays_d = get_rays(H, W, K, c2w=c2w, device=device)
    else:
        # 训练的时候都会直接传入rays，这种情况下是带有batch维度的
        # rays_o: [N_rays, 3]  rays_d: [N_rays, 3]
        rays_o, rays_d = rays
    # 如果网络要加入相机视角方向输入
    if args.use_viewdirs:
        viewdirs = rays_d
        if c2w_staticcam is not None:
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam, device=device)
        # 标准化，torch.norm返回平方和再开根号
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        # [N_rays, 3]
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    # [N_rays, 3]
    sh = rays_d.shape
    # 如果不是环绕场景，而是前向场景
    if not args.no_ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    # 如果是训练的时候（也就是直接传入rays的话），rays_o和rays_d的shape本来就是[N_rays, 3]
    # 但是如果是验证的时候(也就是传入的是整张图每个像素对应的光线时)，rays_o和rays_d本来的shape是[H, W, 3]，将其reshape成[N_rays, 3]
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    # [N_rays, 1]
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    # [N_rays, 8]
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    # 将viewdirs加入输入中，值得注意的是viewdirs与rays_d不同，其是rays_d的标准化形式
    if args.use_viewdirs:
        # [N_rays, 11]
        rays = torch.cat([rays, viewdirs], -1)

    # 执行渲染操作
    ret = render_rays(rays, args, model, embed_fn, embeddirs_fn, model_fine=model_fine, return_raw=return_raw,
                      perturb=perturb, raw_noise_std=raw_noise_std, device=device, is_train=is_train)
    for k in ret:
        k_sh = list(sh[:-1]) + list(ret[k].shape[1:])
        ret[k] = torch.reshape(ret[k], k_sh)
    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    # 将其中三项提取出来组成list[rgb_map, disp_map, acc_map]
    ret_list = [ret[k] for k in k_extract]
    # 其他的依然还是以字典形式存储{'raw':raw, 'rgb0':rgb0, ...}
    ret_dict = {k: ret[k] for k in ret if k not in k_extract}
    # 最后返回的是[rgb_map, disp_map, acc_map, {'raw':raw, 'rgb0':rgb0, ...}]
    return ret_list + [ret_dict]

'''
将args.chunk条光线输入网络得到结果，在render方法中调用
'''
def render_rays(ray_batch, args, model, embed_fn, embeddirs_fn, model_fine=None, return_raw=False, perturb=True,
                raw_noise_std=1e0, device=None, is_train=True):
    '''
    :param ray_batch: [chunk, 8](origin, dir, near, far) or [chunk, 11](origin, dir, near, far, dir)
    :param args: 传入的超参
    :param model: 粗网络，NeRF
    :param embed_fn: 用于位置xyz位置编码的方法
    :param embeddirs_fn: 用于方向位置编码的方法
    :param model_fine: 精细网络，也是NeRF，如果不传入的话则用粗网络中的模型
    :param return_raw: 如果为True，则返回值包含网络的原始输出，否则仅返回处理过后的输出
    :param perturb: 如果为True，则给采样点的位置加上随机噪声
    :param raw_noise_std: 用在raw2outputs方法中，用于给预测出来的体素密度加噪声
    :return:
        rgb_map: [num_rays, 3] 每条光线预测的rgb，来自精细模型
        disp_map: [num_rays] 视差图, 1 / depth，来自精细模型
        acc_map: [num_rays] 每条光线上透明度的累加，来自精细模型
        raw: [num_rays, num_samples, 4] 精细模型的原始输出
        rgb0: [num_rays, 3] 每条光线预测的rgb，来自粗模型
        disp0: [num_rays] 视差图, 1 / depth，来自粗模型
        acc0: [num_rays] 每条光线上透明度的累加，来自粗模型
        z_std: [num_rays] 精细网络采样点位置的标准差
    '''
    # 有多少条光线
    N_rays = ray_batch.shape[0]
    # 光线的起始位置，光线的方向 [chunk ,3]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    # 视角的单位向量 [chunk, 3]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    # 光线上的最近和最远点 [chunk, 1, 2]
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    # near far [chunk, 1] [chunk, 1]
    near, far = bounds[..., 0], bounds[..., 1]
    # 采样点，0到1之间均匀采样N_samples个点
    t_vals = torch.linspace(0., 1., steps=args.N_samples, device=device)
    # 将采样点还原到真实尺度上，基于深度采样点
    if not args.lindisp:
        z_vals = near + (far - near) * t_vals
    # 基于视差采样点
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    # [chunk] --> [chunk, N_samples]
    z_vals = z_vals.expand([N_rays, args.N_samples])
    # 如果加上扰动
    if perturb:
        # 获取采样点中的中点, [chunk, 1]
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # [chunk, 2(中点和最大的点)]
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        # [chunk, 2(最小点和中点)]
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # 随机生成0-1的值 [chunk, N_samples]
        t_rand = torch.rand(z_vals.shape, device=device)
        # 加上随机噪声
        z_vals = lower + (upper - lower) * t_rand
    # 构建网络的输入
    # [chunk, N_sample_points, 3]
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    # 输入网络得到输出: [chunk, N_samples, 4]
    raw = run_network(pts, viewdirs, model, embed_fn, embeddirs_fn, args.N_rand, device=device, is_train=is_train)
    # 将网络输出变成所需要的输出
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, device=device)
    # 以上为粗网络的输出，接下来是精细网络
    if args.N_importance > 0:
        # 先将粗网络的结果保存一下
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        # 取粗网络采样点各区间的中点
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # 通过粗网络的结果进一步采样得到精细网络的采样点
        # [chunk, N_importance]
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], args.N_importance, device=device)
        z_samples = z_samples.detach()
        # 将粗网络的采样点和精细网络的采样点拼接起来再按顺序排好
        # [chunk, N_samples + N_importance]
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        # 采样点准确的xyz，[chunk, N_samples + N_importance, 3]
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        # 输入网络得到结果
        model_fine = model if model_fine is None else model_fine
        raw = run_network(pts, viewdirs, model_fine, embed_fn, embeddirs_fn, args.N_rand, device=device, is_train=is_train)
        # 将网络输出变成需要的输出
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, device=device)
    # 将返回结果组织成字典的形式
    return_items = {'rgb_map':rgb_map, 'disp_map':disp_map, 'acc_map':acc_map}
    # 如果需要将网络的输出结果一起返回
    if return_raw:
        return_items['raw'] = raw
    # 粗网络的结果
    if args.N_importance > 0:
        return_items['rgb0'] = rgb_map_0
        return_items['disp0'] = disp_map_0
        return_items['acc0'] = acc_map_0
        return_items['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

    return return_items

'''
将采样点和方向输入网络，得到原始输出，在render_rays方法中调用
'''
def run_network(pts, viewdirs, model, embed_fn, embeddirs_fn, batch, device, is_train):
    '''
    pts: 采样点 [N_rays, N_samples, 3]
    viewdirs: 视角的单位向量 [N_rays, 3]
    model: NeRF模型
    embed_fn: 位置编码方法（针对xyz的位置编码）
    embeddirs_fn: 位置编码方法（针对视角方向的位置编码）
    batch: 分batch输入网络，主要是为了防止test输入整张图片显存爆炸
    Returns:
    '''
    # [N_rays, N_samples, 3] --> [N_rays*N_samples, 3]
    inputs_flat = torch.reshape(pts, [-1, pts.shape[-1]])
    if not is_train:
        inputs_flat = inputs_flat.cpu()
    # [N_rays*N_samples, 3] --> [N_rays*N_samples, 63]
    embedded = embed_fn.embed(inputs_flat)
    # 如果输入里面包含了视角方向输入
    if viewdirs is not None:
        # [N_rays, 3] --> [N_rays, 1, 3] --> [N_rays, N_samples, 3]
        input_dirs = viewdirs[:,None].expand(pts.shape)
        if not is_train:
            input_dirs = input_dirs.cpu()
        # [N_rays, N_samples, 3] --> [N_rays*N_samples, 3]
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        # 位置编码之后的shape为[N_rays*N_samples, 27]
        embedded_dirs = embeddirs_fn.embed(input_dirs_flat)
        # 将编码后的采样点位置和视角方向cat在一起，[N_rays*N_samples, 90]
        embedded = torch.cat([embedded, embedded_dirs], -1)

    # 将采样点分成minibatch输入网络得到输出，每一个minibatch输出的shape是[chunk, 4]，最后cat起来变成[N_rays*N_samples, 4]
    outputs_flat = torch.cat([model((embedded[i:i+batch]).to(device)) for i in range(0, embedded.shape[0], batch)], 0)
    # outputs_flat = model(embedded)
    # [N_rays*N_samples, 4] --> [N_rays, N_samples, 4]
    outputs = torch.reshape(outputs_flat, list(pts.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

'''
将网络输出变成所需要的输出，也就是变成rgb图、深度图等
这里面用到了体素渲染的方法
NeRF中的体素渲染原理：Volume Rendering Digest(for NeRF)   https://arxiv.org/pdf/2209.02417.pdf
'''
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=1e0, white_bkgd=False, device=None):
    """
        raw: 网络的输出，[N_rays, N_samples, 4]
        z_vals: 每个采样点在光线上的位置，[N_rays, N_samples]
        rays_d: 每条光线的防线，[N_rays, 3]
        raw_noise_std: 给密度加一个噪声
        white_bkgd: 是否假设为白色背景，该参数仅在deepvoxels数据集设置为True，真实数据集不用这个
    Returns:
        rgb_map: 每条光线的rgb，[N_rays, 3]
        disp_map: 视差图，也就是每一条光线的视差，[N_rays]
        acc_map: [N_rays]. 单条光线的权重之和.
        weights: [N_rays, N_samples]. 每一个采样点rgb的权重.
        depth_map: [N_rays]. 深度图.
    """
    # 体渲染公式的中间项，1-exp(-σδ)，σ就是预测出来该采样点处的体素密度，δ就是相邻点之间的距离
    # 这整个一项可以理解为不透明度，体素密度越大、相邻点距离越大则不透明度越大
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    # 每个采样点之间的距离，[N_rays, N_samples-1]
    dists = z_vals[...,1:] - z_vals[...,:-1]
    # 补上最后一段距离（100），[N_rays, N_samples]
    last = torch.Tensor([1e10]).expand(dists[...,:1].shape).to(device)
    dists = torch.cat([dists, last], -1)
    # 将距离乘上方向 [N_rays, N_samples] * [N_rays, 1, 3]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    # 把rgb取出来并缩放至0-1区间，[N_rays, N_samples, 3]
    rgb = torch.sigmoid(raw[...,:3])
    # 给预测出来的体素密度加上噪声
    noise = 0.
    # noise:[N_rays, N_samples]
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape, device=device) * raw_noise_std
    # 计算得到中间项的值，[N_rays, N_samples]
    alpha = raw2alpha(raw[...,3] + noise, dists)
    # 体渲染公式中的第一项与第二项相乘得到最终的系数
    # 第一项可以理解为光线在区间内传播而不碰到任何粒子的概率，可以理解为透明度，也就是对颜色的衰减系数（因为采样点的颜色由反射光的粒子决定，与透过的光无关）
    # 原先这一项的表达式是Tn = exp(Σ-σδ) = Πexp(-σδ) = Π(1-alpha), 这里加上1e-10是防止乘积为0
    # 每一条光线的一开始cat上一个1，因为第一个点是没有受影响的，第二个点受到的是第一个点的影响，第三个点受到的是第一个点和第二个点的影响...
    # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    # 将结果与体渲染公式中的第三项相乘，并累加求和得到最终渲染的rgb
    # [N_rays, 3]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    # [N_rays, N_samples] * [N_rays, N_samples] --> [N_rays]
    depth_map = torch.sum(weights * z_vals, -1)
    # [N_rays]
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    # [N_rays]
    acc_map = torch.sum(weights, -1)
    # 真实数据集不用这个
    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map

'''
分层采样，用于粗网络得到结果后，总的来说第二次采样就是会在第一次计算结果中weights大的地方采样多一些点，weights小的地方采样少一些点
'''
def sample_pdf(z_vals, weights, N_importance, det=False, device=None):
    '''
    Args:
        z_vals: 传进来的点在光线上的位置，此处传进来的是粗网络中各区间的中点，所以点会比粗网络少一个，[N_rays, N_samples-1]
        weights: 体渲染计算出来的权重除去头尾，[N_rays, N_samples-2]
        N_importance: 精细网络每条光线上采样点的数量
        det: 是否均匀采样

    Returns:
        samples: 精细网络的采样点，[N_rays, N_importance]
    '''
    # 防止为0
    weights = weights + 1e-5
    # 求每一条光线上采样点的概率密度,[N_rays, N_samples-2]
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    # 求每一条光线上采样点的累积分布函数,[N_rays, N_samples-2]
    cdf = torch.cumsum(pdf, -1)
    # 将累积分布的0项添加到每一条光线中, [N_rays, N_samples-1]
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    if det:
        # 均匀采样, [N_rays, N_importance]
        u = torch.linspace(0., 1., steps=N_importance, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])
    else:
        # 随机采样, [N_rays, N_importance]
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)
    # 下面是进行逆变换采样
    u = u.contiguous()
    # 得到u中的值在cdf中的区间索引，[N_rays, N_importance]
    inds = torch.searchsorted(cdf, u, right=True)
    # 新的采样点所在区间的底边,[N_rays, N_importance]，限制在0以上
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    # 新的采样点所在区间的顶边,[N_rays, N_importance]，限制在N_importance-2以下
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    # 将新采样点所在区间的上下限组合起来，[N_rays, N_importance, 2]
    inds_g = torch.stack([below, above], -1)
    # [N_rays, N_importance, N_samples-1]
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    # cdf.unsqueeze(1): [N_rays, N_samples-1] --> [N_rays, 1, N_samples-1]
    # .expand(matched_shape): [N_rays, 1, N_samples-1] --> [N_rays, N_importance, N_samples-1]
    # [N_rays, N_importance, 2]， 代表每个采样点对应累积分布中的区间上下限
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    # z_vals.unsqueeze(1): [N_rays, N_samples-1] --> [N_rays, 1, N_samples-1]
    # .expand(matched_shape): [N_rays, 1, N_samples-1] --> [N_rays, N_importance, N_samples-1]
    # [N_rays, N_importance, 2]，代表每个采样点对应原始区间的上下限
    bins_g = torch.gather(z_vals.unsqueeze(1).expand(matched_shape), 2, inds_g)
    # 上限-下限，每一段的区间长度，[N_rays, N_importance]
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    # 区间长度如果很小，则取1
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    # (采样点位置-累积分布中区间的下限)/区间的长度
    t = (u - cdf_g[..., 0]) / denom
    # 这里相当于是将采样点从累积分布中的位置映射到原始区间的位置
    # [N_rays, N_importance]，精细网络的采样点
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples


'''
测试和推理
'''
def render_path(render_poses, hwf, K, args, model, embed_fn, embeddirs_fn, model_fine=None, near=0.,far=1.,
                c2w_staticcam=None, savedir=None, device=None):
    '''
    Args:
        render_poses:
        hwf: [3] H, W, focal
        K: 相机内参矩阵, [3, 3], 用于从图片获取光线
        args: 超参
        model: 粗网络模型
        embed_fn: xyz的位置编码方法
        embeddirs_fn: 角度的位置编码方法
        model_fine: 精细网络
        near: 场景内近的点
        far: 场景内远的点
        c2w_staticcam: 此参数仅在use_viewdirs为True时生效，意思是用这个矩阵来代替c2w
        savedir: 保存图片的路径
        device: cpu or cuda?
    '''
    H, W, focal = hwf
    rgbs = []
    disps = []
    # 遍历每一个需要渲染的视角
    for i, c2w in enumerate(tqdm(render_poses)):
        # 渲染
        rgb, disp, _, _ = render(H, W, K, args, model, embed_fn, embeddirs_fn, model_fine=model_fine, c2w=c2w[:3,:4],
                                near=near, far=far, c2w_staticcam=c2w_staticcam, device=device, is_train=False)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        # 保存生成的视角图片，为节省时间仅保存3张
        if savedir is not None and len(rgbs) < 3:
            rgb8 = to8b(rgbs[-1])
            file_name = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(file_name, rgb8)
    # list --> array
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps

# 将图片转换为uint8
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

'''
记录loss并可视化
'''
class LossHistory():
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []
        if not os.path.exists(log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        # try:
        #     if len(self.losses) < 25:
        #         num = 5
        #     else:
        #         num = 15
        #     plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
        #              label='smooth train loss')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
        #              label='smooth val loss')
        # except:
        #     pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")

# 将图片转换成视频，基于opencv的实现
def imgs2video(save_path, images, fps):
    # 先定义好视频的编码格式
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 如果images是list，每一项是image的路径
    if type(images) is list:
        size = (int(cv2.imread(images[0]).shape[1]), int(cv2.imread(images[0]).shape[0]))
        video = cv2.VideoWriter(save_path, fourcc, fps, size)
        for image in images:
            img = cv2.imread(image)
            video.write(img)
        video.release()
    # [B, H, W] or [B, H, W, C]
    elif type(images) is np.ndarray:
        imgs = [images[i] for i in range(images.shape[0])]
        size = (imgs[0].shape[1], imgs[0].shape[0])
        video = cv2.VideoWriter(save_path, fourcc, fps, size)
        for image in images:
            video.write(image)
        video.release()

if __name__ == '__main__':
    # z_vals = torch.rand([5, 3])
    # weights = torch.rand([5, 3])
    # c = torch.rand([4, 3])
    # lis = [z_vals,weights, c]
    # out = torch.cat(lis, 0)
    # print(out.shape)
    # N_importance = 10
    # samples = sample_pdf(z_vals, weights, N_importance, det=True)
    # print(samples.shape)
    # 读取所有 PNG 图片
    # folder = r'C:\Users\ENFI\Desktop\images'
    # images = []
    # for file_name in sorted(os.listdir(folder)):
    #     if file_name.endswith('.jpg'):
    #         path = os.path.join(folder, file_name)
    #         images.append(cv2.imread(path))
    # fps = 10  # 每秒钟30帧
    # with imageio.get_writer('test.mp4') as video:
    #     for image in images:
    #         video.append_data(image)
    path = r'C:\Users\ENFI\Desktop\images'
    filelist = glob.glob(os.path.join(path, '*.jpg'))
    filelist = sorted(filelist)  # 按照文件数字进行顺序排序
    print(len(filelist))
    imgs2video(filelist, 'test.mp4', 10.0)

