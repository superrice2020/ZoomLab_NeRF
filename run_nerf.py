import argparse
import warnings
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio
from load_real_data import load_real_data
from run_nerf_helpers import *
import datetime
from nerf_model import NeRF_Model
'''
NeRF论文：https://arxiv.org/abs/2003.08934
NeRF官方代码（基于TensorFlow实现）：https://github.com/bmild/nerf
NeRF Pytorch实现代码（本代码参考的版本）：https://github.com/yenchenlin/nerf-pytorch
NeRF的数据集分为真实数据集以及合成数据集，官方写了四种数据集的加载方法，包括llff，nerf_synthetic(Nerf原论文使用的数据集)，deepvoxels，以及LINEMOD
llff：真实数据集，相机视角面向对象小范围运动，一共八个场景，每个场景几十张图像，每个场景的图像有三种分辨率：[4032, 3024],[1008, 756],[504, 378]，除此之外还包含了点云数据
nerf_synthetic：合成数据集，一共有八个场景，都是blender模型拍摄的图片，每个场景100张图片作为训练集，100张图片作为验证集，200张作为测试集，
                测试集除了rbg图片还包含了深度图。
deepvoxels：这个数据集包含了很多真实和合成的数据
这里只写了的针对真实数据集的加载以及训练

注意！！！
1. imageio要装2.9.0版本，高版本会报错需要改东西
2. 如果是forward facing的场景'--no_ndc''--lindisp''--spherify'这三个参数要设为False，如果是360环绕场景，则都设为True
3. 首次训练可以将'--i_valid'调小一些（比如100或者200），如果数据集没问题，200个iters一般能输出个模糊的场景了
'''

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

'''
训练超参设置
'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=r'E:\datasets\nerf\nerf_llff_data\horns', help='这个路径下需要有poses_bounds.npy以及images文件夹')
    # 网络模型参数
    parser.add_argument('--netdepth', type=int, default=8, help='网络层数')
    parser.add_argument('--netwidth', type=int, default=256, help='网络层的通道数')
    parser.add_argument('--netdepth_fine', type=int, default=8, help='精调网络的网络层数')
    parser.add_argument('--netwidth_fine', type=int, default=256, help='精调网络的通道数')
    # 训练参数
    parser.add_argument('--seed', type=int, default=666, help='固定随机种子，-1就是随机')
    parser.add_argument('--iters', type=int, default=200000, help='训练一共迭代多少次')
    parser.add_argument('--N_rand', type=int, default=1024*2, help='batch_size，训练一次迭代中用多少像素点')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--lr_decay', type=int, default=250, help='学习率衰减')
    parser.add_argument('--batching', type=bool, default=True, help='是从所有图片中取一个batch的光线还是从一张图片中取一个batch的光线，合成数据集一般为Fasle，真实数据集一般为True')
    parser.add_argument('--model_pretrain', type=str, default=None, help='粗网络的权重路径')
    parser.add_argument('--model_fine_pretrain', type=str, default=None, help='精网络的权重路径')
    parser.add_argument("--precrop_iters", type=int, default=0, help='使用中心裁剪的先训练多少个iter，然后再全图训练')
    parser.add_argument("--precrop_frac", type=float, default=.5, help='中心裁剪的范围')
    # 渲染参数
    parser.add_argument('--N_samples', type=int, default=64, help='粗网络每条光线上采样点的数量')
    parser.add_argument('--N_importance', type=int, default=128, help='精细网络每条光线上采样点的数量')
    parser.add_argument('--use_viewdirs', type=bool, default=False, help='训练时是否使用视角数据(也就是6D输入)，不使用的话就是3D输入，实测不加效果也差不多')
    parser.add_argument('--multires', type=int, default=10, help='位置编码的组合（3D位置），一般设为10')
    parser.add_argument('--multires_views', type=int, default=4, help='位置编码的组合（3D视角），一般设为4')
    parser.add_argument("--render_only", type=bool, default=False, help='仅渲染，不训练')
    parser.add_argument("--render_test", type=bool, default=False, help='不使用生成的新视角，而是渲染测试集')
    # 数据集参数
    parser.add_argument("--factor", type=int, default=16, help='数据处理时的下采样倍率')
    parser.add_argument("--no_ndc", type=bool, default=False, help='不使用归一化设备坐标系，环绕场景设为True，前向场景一般用ndc，但也可以不用，效果会差一些')
    parser.add_argument("--lindisp", type=bool, default=False, help='是否基于视差图采样而不是深度')
    parser.add_argument("--spherify", type=bool, default=False, help='用于360度场景')
    parser.add_argument("--llffhold", type=int, default=8, help='每隔N张图片选一张做测试集, 论文中用的8')
    # 训练记录和保存参数
    parser.add_argument("--i_log", type=int, default=200, help='每多少个iter打印一次loss信息')
    parser.add_argument("--i_valid", type=int, default=1000, help='验证的频率，会保存权重，计算验证集loss和psnr')
    parser.add_argument("--i_test", type=int, default=1000, help='测试的频率，会生成新视角图像，比较耗时，建议隔久一些')
    return parser.parse_args()

def main(args):
    # 固定随机种子
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

    K = None
    # 加载数据集，bds只用于计算near和far，且前向场景没用到
    images, poses, bds, render_poses = load_real_data(args.datadir, args.factor, recenter=True,
                                                      spherify=args.spherify, path_zflat=True)
    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]
    print('已加载数据集', '\n  image_shape: ', images.shape, '\n  render_poses_shape: ', render_poses.shape,
          '\n  hwf: ', hwf, '\n  bds_min:', np.ndarray.min(bds), '    bds_max：', np.ndarray.max(bds), '     device: ', device,
          '\n  数据集路径: ', args.datadir)
    # 挑选出测试集图片
    i_test = []
    if args.llffhold > 0:
        i_test = np.arange(images.shape[0])[::args.llffhold]

    i_val = i_test
    # 使用不在测试集中的图片进行训练
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)])
    # 获取验证集rgb
    valid_images = images[i_val]
    # 如果不使用ndc的话则场景范围不归一化到0-1之间，注意前向场景也可以不使用ndc，只是用了ndc效果会好一些（实测），不用ndc的话一定不能将场景归一化
    if args.no_ndc:
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.
    else:
        near = 0.
        far = 1.
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # K就是相机的内参矩阵，这里默认u和v是图像宽高的一半
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    # 使用测试集的pose，而不是用那个生成的环绕路径的render_pose
    if args.render_test:
        render_poses = np.array(poses[i_test])

    # 获取当前时间，创建本次训练的目录
    train_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    train_start_time = train_start_time.replace(' ', '_').replace('-', '_').replace(':', '_')
    save_folder = os.path.join('train_result', train_start_time)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 创建log文件夹并保存训练信息
    log_folder = os.path.join(save_folder, 'Log')
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    # 保存超参设置
    f = os.path.join(save_folder, 'Log', 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    # 创建iter结果保存文件夹
    iters_folder = os.path.join(save_folder, 'Iters')
    if not os.path.exists(iters_folder):
        os.makedirs(iters_folder)

    # 实例化记录loss的类
    loss_history = LossHistory(log_folder)

    # 实例化Embedder
    # embed_fn是对位置xyz的编码方法，embed_fn_dirs是对方向xyz的编码方法，区别在于编码数量不同（即multires参数）
    embed_fn = Embedder(include_input=True, input_dim=3, multires=args.multires, log_sampling=True)
    embed_fn_dirs = Embedder(include_input=True, input_dim=3, multires=args.multires_views, log_sampling=True)
    input_ch = embed_fn.out_dim
    input_ch_views = embed_fn_dirs.out_dim

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    # 实例化模型
    model = NeRF_Model(D=args.netdepth, W=args.netwidth, input_ch=input_ch, input_ch_views=input_ch_views,
                 output_ch=output_ch, skips=skips, use_viewdirs=args.use_viewdirs).to(device)
    # 加载预训练模型
    if args.model_pretrain is not None:
        ckpt = torch.load(args.model_pretrain)
        model.load_state_dict(ckpt)
    grad_vars = list(model.parameters())
    # 如果需要精调网络
    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF_Model(D=args.netdepth_fine, W=args.netwidth_fine, input_ch=input_ch, output_ch=output_ch,
                          skips=skips, input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        if args.model_fine_pretrain is not None:
            ckpt_fine = torch.load(args.model_fine_pretrain)
            model_fine.load_state_dict(ckpt_fine)
        grad_vars += list(model_fine.parameters())
    # 优化器
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lr, betas=(0.9, 0.999))

    # 仅推理
    render_poses = torch.Tensor(render_poses).to(device)
    if args.render_only:
        print('仅渲染')
        with torch.no_grad():
            testsavedir = os.path.join(save_folder, 'render_only_result')
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args, model, embed_fn, embed_fn_dirs, model_fine, near, far,
                                  savedir=testsavedir, device=device)
            print('渲染已完成', testsavedir)
            # 存为视频
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            return
    # 如果训练数据是从所有图片中随机抽取的光线
    if args.batching:
        # 获取所有的光线, [N, ro+rd(2), H, W, 3]
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)
        # images[:, None]: [N_images, 1, H, W, 3]
        # 将图片拼接上来，[N, ro+rd+rgb(3), H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None]], 1)
        # [N, H, W, ro+rd+rgb(3), 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        # 仅取训练的数据
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)
        # [i_train*H*W, ro+rd+rgb(3), 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        # 打乱光线的顺序
        np.random.shuffle(rays_rgb)

        i_batch = 0
        images = torch.Tensor(images).to(device)
        rays_rgb = torch.Tensor(rays_rgb).to(device)
    poses = torch.Tensor(poses).to(device)

    # render方法：准备好训练数据
    # render_rays方法：在rays上采样点并输入网络得到结果，分为粗网络和精细网络，最后进行体渲染得到最终结果
    # run_network方法：输入模型得到输出
    # 每一次迭代是将args.N_rand条光线输入render方法，然后render方法中准备好数据就传入render_rays方法，render_rays方法中的粗网络和
    # 精细网络调用run_network方法得到网络输出，最后通过raw2outputs方法得到结果
    for i in trange(1, args.iters + 1):
        time0 = time.time()
        # 如果训练数据是从所有图片中随机抽取光线
        if args.batching:
            # 一次训练输入的光线数量为N_rand,[N_rand, 3, 3]
            batch = rays_rgb[i_batch:i_batch + args.N_rand]
            # [3, N_rand, 3]
            batch = torch.transpose(batch, 0, 1)
            # 前两个是rayo和rayd，最后一个是image
            batch_rays, target_s = batch[:2], batch[2]
            # 索引变为下一个batch的索引
            i_batch += args.N_rand
            # 如果当前batch索引超出了总光线数量，则打乱所有光线从头开始采样
            if i_batch >= rays_rgb.shape[0]:
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        # 不然的话就是从一张图片中随机采样光线
        else:
            # 随机选取一张图片的序号
            img_i = np.random.choice(i_train)
            # 获取到该图片
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            # [3, 4]
            pose = poses[img_i, :3, :4]
            # 对于一张图片，选取其中的一部分像素点进行训练
            if args.N_rand is not None:
                # 获取光线，shape都是[H, W, 3]
                rays_o, rays_d = get_rays(H, W, K, pose)
                # 前面的iters用中心裁剪
                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    # 中间区域的像素坐标，[dH, hW, 2]
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                        ), -1)
                else:
                    # 全图区域的像素坐标，[H, W, 2]
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                         -1)
                # [H, W, 2] --> [H * W, 2]
                coords = torch.reshape(coords, [-1, 2])
                # 随机选像素点，[N_rand]
                select_inds = np.random.choice(coords.shape[0], size=[args.N_rand], replace=False)
                # 得到选择的像素点的坐标，[N_rand, 2]
                select_coords = coords[select_inds].long()
                # 分别取出对应的光线原点和方向，shape都是[N_rand, 3]
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
                # [2, N_rand, 3]
                batch_rays = torch.stack([rays_o, rays_d], 0)
                # 取出对应像素点处的rgb,[N_rand, 3]
                target_s = target[select_coords[:, 0], select_coords[:, 1]]

        # render方法包括数据准备、输入网络、得到结果
        rgb, disp, acc, extras = render(H, W, K, args, model=model, embed_fn=embed_fn, embeddirs_fn=embed_fn_dirs,
                                        model_fine=model_fine, return_raw=True, rays=batch_rays, c2w=None,
                                        near=near, far=far, device=device, is_train=True)
        optimizer.zero_grad()
        # 计算均方误差
        img_loss = torch.mean((rgb - target_s) ** 2)
        loss = img_loss
        # 计算psnr
        psnr = -10. * torch.log(img_loss.cpu()) / torch.log(torch.Tensor([10.]))
        # 是否也计算粗网络的loss
        if 'rgb0' in extras:
            img_loss0 = torch.mean((extras['rgb0'] - target_s) ** 2)
            loss = loss + img_loss0
            psnr0 = -10. * torch.log(img_loss0.cpu()) / torch.log(torch.Tensor([10.]))
        # 梯度回传并更新优化器
        loss.backward()
        optimizer.step()
        # 更新学习率
        decay_rate = 0.1
        decay_steps = args.lr_decay * 1000
        new_lr = args.lr * (decay_rate ** (i / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        # 当前iter所用的时间
        dt = time.time() - time0
        # 反馈当前进程中Torch.Tensor所占用的GPU显存
        mem = f'{torch.cuda.max_memory_allocated(device) / 1E9 if torch.cuda.is_available() else 0:.3g}G'
        # 每隔一段时间打印一下训练信息
        if i % args.i_log == 0:
            print(f"Step: {i}, LR:{new_lr}, Loss: {loss}, Time: {dt}, Memory_allocated: {mem}, PSNR: {psnr.item()}")
            # 记录loss
        # 隔一段时间保存权重并验证效果
        if i % args.i_valid == 0:
            print('验证中...')
            iter_save_folder = os.path.join(iters_folder, str(i))
            # 创建权重保存文件夹
            model_save_folder = os.path.join(iter_save_folder, 'Model')
            if not os.path.exists(model_save_folder):
                os.makedirs(model_save_folder)
            # 创建验证文件夹
            valid_save_path = os.path.join(iter_save_folder, 'Valid_Result')
            if not os.path.exists(valid_save_path):
                os.makedirs(valid_save_path)
            # 保存权重
            torch.save(model.state_dict(),
                       os.path.join(model_save_folder, "model-iter{}-loss{}.pth".format(i, loss)))
            if args.N_importance > 0:
                torch.save(model_fine.state_dict(),
                       os.path.join(model_save_folder, "modelfine-iter{}.pth".format(i)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_save_folder, "optimizer-iter{}.pth".format(i)))
            # 验证集，仅保存图片，不生成视频
            with torch.no_grad():
                rgbs, disps = render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args, model, embed_fn, embed_fn_dirs, model_fine, near, far,
                                  savedir=valid_save_path, device=device)
                # 计算valid loss
                img_valid_loss = np.mean((rgbs - valid_images) ** 2)
                # 计算psnr
                valid_psnr = -10. * np.log(img_valid_loss) / np.log(np.array([10.]))
                print(f"Step: {i}, Valid_Loss: {img_valid_loss}, Valid_PSNR: {valid_psnr}")
                loss_history.append_loss(i, loss.item(), img_valid_loss)
                loss_history.loss_plot()
                del rgbs, disps, img_valid_loss, valid_psnr
                print('验证完成')
        if i % args.i_test == 0:
            iter_save_folder = os.path.join(iters_folder, str(i))
            # 创建保存视频的文件夹
            render_save_path = os.path.join(iter_save_folder, 'Render_Result')
            if not os.path.exists(render_save_path):
                os.makedirs(render_save_path)
            # 生成新视角并保存为视频
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args, model, embed_fn, embed_fn_dirs, model_fine, near, far,
                                  savedir=render_save_path, device=device)
            moviebase = os.path.join(render_save_path, 'spiral_{:06d}_'.format(i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
    loss_history.writer.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)


