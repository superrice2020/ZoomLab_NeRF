import imageio
import numpy as np
import os
import cv2

'''
真实数据集加载方法，通过手机拍摄图片之后，通过colmap估计出相机的参数，得到poses_bounds.npy文件，再通过load_llff_data方法加载
需要先执行poses文件夹中的pose_utils.py脚本生成poses_bounds.npy
'''


'''
缩放图片并保存
'''
def scale_image_and_save(basedir, factor):
    '''
    Args:
        basedir: 图片文件夹路径
        factor: 下采样倍数
    '''
    # 获取images文件夹下所有图片
    imagedir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imagedir, f) for f in sorted(os.listdir(imagedir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]

    for r in factor:
        name = 'images_{}'.format(r)
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        os.makedirs(imgdir)
        for img in imgs:
            image = cv2.imread(img)
            image_resize = cv2.resize(image, (int(image.shape[1] / r), int(image.shape[0] / r)))
            new_path = os.path.join(imgdir, os.path.basename(img))
            cv2.imwrite(new_path, image_resize)
    print('已完成图片缩放')



'''
用于将向量变为单位向量的方法
np.linalg.norm默认是l2范数
'''
def normalize_matrix(x):
    return x / np.linalg.norm(x)

'''
用于构造相机矩阵的方法，这里仅传入两个方向的向量和相机位置，再通过两个向量计算出第三个向量最后再和位置拼接在一起得到3×4的相机矩阵
来自：https://github.com/Fyusion/LLFF/blob/master/llff/math/pose_math.py
'''
def view_matrix(z, up, pos):
    '''
    相机矩阵：
    (X) (Y-up) (Z)  (pos)
    r11  r12   r12   t1
    r21  r22   r23   t2
    r31  r32   r33   t3
    :param z: 相机的z轴旋转向量
    :param up: 相机的y轴旋转向量
    :param pos: 相机中心
    :return: 返回相机矩阵
    '''
    # 归一化z轴向量
    vec2 = normalize_matrix(z)
    vec1_avg = up
    # 通过叉乘z轴单位向量和y轴单位向量得到x轴单位向量
    vec0 = normalize_matrix(np.cross(vec1_avg, vec2))
    # 再通过叉乘z轴单位向量和x轴单位向量得到y轴单位向量
    # 这样做的原因是原来的y轴是通过一些计算得到的，不一定和z轴垂直，这样做就保证了三个轴都是互相垂直的
    vec1 = normalize_matrix(np.cross(vec2, vec0))
    # 按列的形式进行拼接
    # [X, Y, Z, pos]
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

'''
输入多个相机的位姿，返回平均相机位姿
来自：https://github.com/Fyusion/LLFF/blob/master/llff/math/pose_math.py
'''
def poses_avg(poses):
    '''
    :param poses: 相机矩阵[N, 3, 5]
    :return: 新的相机矩阵[3, 5]
    '''
    # 将H,W,f值先取出来，方便后续拼接为新的相机矩阵
    hwf = poses[0, :3, -1:]
    # 求所有相机中心点的平均
    center = poses[:, :3, 3].mean(0)
    # 求所有相机z轴向量的平均
    vec2 = normalize_matrix(poses[:, :3, 2].sum(0))
    # 求y轴旋转向量的平均
    up = poses[:, :3, 1].sum(0)
    # 按列进行拼接
    c2w = np.concatenate([view_matrix(vec2, up, center), hwf], 1)
    return c2w

'''
此方法适用于faceforward场景，这个函数与训练无关，仅用于验证，不改变原有视角，仅仅是用于生成新的相机视角，此方法生成的是一段螺旋式的相机轨迹，
相机绕着一个轴旋转，其中相机始终注视着一个焦点。
来自：https://github.com/Fyusion/LLFF/blob/master/llff/math/pose_math.py
'''
def render_path_spiral(c2w, up, rads, focal, zrate, N_rots, N_views):
    '''
    :param c2w: 平均位姿[3, 5]
    :param up: 所有相机位姿的up向量的平均值
    :param rads: [3]生成的圆形视角轨迹的半径
    :param focal: 相机位姿注视着的焦点
    :param zrate: 取0.5，这样的z的变化值就会小一些
    :param N_rots: 绕几圈
    :param N_views: 需要生成多少个视角
    :return: 是一个list，每一项是新生成的相机位姿，[[3, 5], [3, 5], ...]
    '''
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    for theta in np.linspace(0., 2. * np.pi * N_rots, int(N_views+1))[:-1]:
        # c是当前相机在世界坐标系的位置
        # [np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads就是以平均位姿为坐标系生成的坐标
        # 生成的新相机视角在xy平面是一个圆,xz,yz平面是一个椭圆，z轴为0的话则仅在xy平面生成圆排布的新视角
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)
        # np.dot(c2w[:3, :4], np.array([0, 0,-focal, 1.]))得到的是焦点在世界坐标系中的位置
        # 用位置向量c减焦点的位置向量得到相机z轴在世界坐标系的朝向
        z = normalize_matrix(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        # 将所有变化过后的相机位姿添加进列表中
        render_poses.append(np.concatenate([view_matrix(z, up, c), hwf], 1))
    return render_poses


'''
输入N个相机位姿，返回N个相机位姿，变换后N个相机的平均位置处在世界坐标系原点，平均相机位姿的原点与世界坐标系原点一致，XYZ轴与世界坐标系保持一致
这样也算是归一化的一种方式。。。吧
作者自己的解释：https://github.com/bmild/nerf/issues/34
'''
def recenter_poses(poses):
    '''
    :param poses: 相机矩阵[N, 3, 5]
    :return: 返回新的相机矩阵[N, 3, 5]
    '''
    poses_ = poses
    bottom = np.reshape([0,0,0,1.], [1,4])
    # 先获得所有输入相机的平均位姿
    c2w = poses_avg(poses)
    # 先丢掉hwf，仅保留旋转平移向量
    # concat之后的形式为标准的旋转平移矩阵
    # r11  r12   r12   t1
    # r21  r22   r23   t2
    # r31  r32   r33   t3
    #  0    0     0    1
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    # reshape: [[[0,0,0,1.]]], 按[N,1,1]进行tile运算得到[N, 1, 4]
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    # [N, 4, 4]
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    # c2w的逆(shape为[4, 4])左乘poses(shape为[N,4,4])，这相当于对每一个相机位姿都做一个旋转平移变换
    # 使得变换后的平均相机位姿位置处在世界坐标系原点，XYZ轴与世界坐标系保持一致
    poses = np.linalg.inv(c2w) @ poses
    # 丢掉bottom，shape变回[N, 3, 4]
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


'''
该方法用于找到与所有相机发出的射线距离之和最短的点，用于找到场景中心
需要注意这里的rays_d都是单位向量，具体原理可自行搜索最小二乘法
'''
def min_line_dist(rays_o, rays_d):
    # [N_images, 3, 3]
    A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
    # b_i = -A_i @ rays_o
    # [N_images, 3, 1]
    b_i = -(np.transpose(A_i, [0, 2, 1]) @ A_i) @ rays_o
    # pt_mindist:[3]
    pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
    return pt_mindist


'''
该方法用于“球面化”相机分布并返回一个环绕的相机轨迹用于新视角合成，如果是360度环绕场景则用这个方法
这个方法会改变原来的相机位姿，具体步骤就是将相机位姿从世界坐标系转换到以视角中心为原点的坐标系，然后缩放到到单位圆内，场景边界也进行对应缩放
并且该方法会生成120个环绕场景中心的相机位姿用于测试
'''
def spherify_poses(poses, bds):
    '''
    :param poses: 相机矩阵[N, 3, 5]
           bds: 每个相机的范围[N, 2]
    :return: poses_reset: 原相机位姿经过一些全局变换得到的用于训练的位姿
             new_poses: 新生成的用于测试的相机位姿
             bds: 新的边界
    '''
    # 定义一个方法，将[0,0,0,1]添加到位姿矩阵的下面，构造[N,4,4]旋转平移矩阵
    p34_to_p44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0],1,1])],1)
    # rays_d:[N, 3, 1], z轴旋转向量，也就是相机在世界坐标系中的中心射线方向
    # 注意rays_d是单位向量
    rays_d = poses[:, :3, 2:3]
    # rays_o:[N, 3, 1], 平移向量，也就是相机在世界坐标的位置
    # 注意rays_o没有归一化，是真实的在世界坐标中的位置
    rays_o = poses[:, :3, 3:4]
    # 得到与所有相机中心发出的射线距离之和最短的点，也就是找出所有环绕视角的中心
    center = min_line_dist(rays_o, rays_d)
    # 将所有相机位置减去新的中心点坐标(变成以此中心点为坐标原点)，再取所有相机位置的平均
    # mean: [N, 3] --> [3]，取平均向量作为Z轴
    vec_z = (poses[:,:3,3] - center).mean(0)
    # 变为单位向量，[3]
    vec_z = normalize_matrix(vec_z)
    # [3] 叉乘 [3] --> [3], 先随便找一个与vec_z垂直的单位向量
    vec_x = normalize_matrix(np.cross([.1, .2, .3], vec_z))
    # 根据已有的两个向量叉乘得到第三个向量
    vec_y = normalize_matrix(np.cross(vec_z, vec_x))
    pos = center
    # 以上面得到的中心点为中心，三个两两垂直的向量为xyz轴构建坐标系
    # c2w: [3, 4]，这个矩阵是从新的坐标系转换为原来的世界坐标系的旋转平移矩阵
    c2w = np.stack([vec_x, vec_y, vec_z, pos], 1)
    # poses_reset:[N, 4, 4]，将原先相机位姿左乘旋转平移矩阵的逆得到在新的坐标系下的相机位姿
    poses_reset = np.linalg.inv(p34_to_p44(c2w[None])) @ p34_to_p44(poses[:, :3, :4])
    # 求所有相机位置到新坐标系原点距离的平均，也就是找到圆的半径
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]),-1)))
    # 缩放系数
    sc = 1./rad
    # 将相机位置缩放到单位圆内
    poses_reset[:,:3,3] *= sc
    # 缩放边界
    bds *= sc
    # 1
    rad *= sc
    # 缩放之后再计算一次平均位置
    centroid = np.mean(poses_reset[:, :3, 3], 0)
    # 平均位置的z值
    zh = centroid[2]
    # 得到圆的半径
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []
    # 均匀按圆形排布，生成120个新视角
    for th in np.linspace(0., 2.*np.pi, 120):
        # 视角原点，z轴坐标固定为上面得到的平均z值
        cam_origin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        # 世界坐标z轴
        up = np.array([0, 0, -1.])
        # 视角原点与坐标原点连成的向量作为新视角相机坐标的z轴
        vec2 = normalize_matrix(cam_origin)
        # vec0既垂直于z轴也垂直于vec2，也就是新视角相机坐标的x轴
        vec0 = normalize_matrix(np.cross(vec2, up))
        # 最后得到y轴
        vec1 = normalize_matrix(np.cross(vec2, vec0))
        pos = cam_origin
        # 组合成相机位姿矩阵
        p = np.stack([vec0, vec1, vec2, pos], 1)
        new_poses.append(p)
    # 将列表变成array
    new_poses = np.stack(new_poses, 0)
    # poses[0, :3, -1:]: [3, 1] 取出hwf
    # new_poses[:, :3, -1:]: [120, 3, 1] np.broadcast_to就是将hwf复制120份
    # 把hwf拼接上去，得到[120, 3, 5]的新视角相机位姿矩阵
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    # poses[0, :3, -1:]: [3, 1] 取出hwf
    # poses_reset[:, :3, -1:]: [N, 3, 1] np.broadcast_to就是将hwf复制N份
    # 把hwf拼接上去，得到[N, 3, 5]的原视角相机矩阵
    poses_reset = np.concatenate([poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, new_poses, bds


'''
路径下需要有poses_bounds.npy文件以及images文件夹存放图片
'''
def _load_data(basedir, factor=1):
    '''
    Args:
        basedir: 文件夹路径
        factor: 图片的下采样倍数
    Returns:
        poses: [3, 5, N_images]
        bds: [2, N_images]
        imgs: [H, W, 3, N_images]

    '''
    # poses_bounds.npy中是一个[N, 17]矩阵，N是图片数量，每一张图片有17个参数，前面15个参数包含旋转平移矩阵、图像的H，W和焦距f
    # 后两个参数用于表示场景的范围，是该相机视角下场景点离相机中心最近和最远的距离
    # 重排之后前15个参数为以下形式：
    # r11 r12 r12 t1 H
    # r21 r22 r23 t2 W
    # r31 r32 r33 t3 f
    # !!注意这里的r1,r2,r3都是单位向量
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    # [N, 15] --> [N, 3, 5] --> [3, 5, N]
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    # [N, 2] --> [2, N]
    # bds的两个值是相机坐标z轴上的小值和大值，也就是深度值
    bds = poses_arr[:, -2:].transpose([1, 0])

    # 文件夹后缀
    sfx = ''

    # 如果需要缩放图片
    if factor != 1:
        sfx = '_{}'.format(factor)
        scale_image_and_save(basedir, factor=[factor])

    # 图片路径
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, '不存在')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print('相机数据与图片数据不匹配！')
        return

    sh = imageio.imread(imgfiles[0]).shape
    # 把缩放后的HW赋值给poses矩阵
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    # 图片缩放后相机参数仅需要将f等比缩放即可
    poses[2, 4, :] = poses[2, 4, :] * 1./factor

    # 读取图片[[H, W, 3], [H, W, 3]...]
    imgs = [imageio.imread(f)[..., :3] / 255. for f in imgfiles]
    # [H, W, 3, N]
    imgs = np.stack(imgs, -1)
    return poses, bds, imgs



'''
读取真实数据格式的数据
'''
def load_real_data(basedir, factor=8, recenter=True, spherify=False, path_zflat=False, bd_factor=0.75):
    '''
    :param basedir: 文件夹路径
    :param factor: 图片的下采样倍率
    :param recenter: 是否将所有相机的位姿做中心化，具体操作看recenter_poses方法
    :param spherify: 用于处理环绕视角数据
    :param path_zflat: 新生成的相机视角的z坐标值是否都为0
    :param bd_factor: 用于确保场景到相机的最小距离大于1，与后续的args.ndc配合使用
    :return:
    '''
    # 读取相机参数和图片，每张图片对应一个相机参数
    # poses: [3, 5, N_images]; bds: [2, N_images]; imgs: [H, W, 3, N_images]
    poses, bds, imgs = _load_data(basedir, factor=factor)

    # print('已加载图片', imgs.shape, poses[:, -1, 0])
    # 将旋转矩阵第一列（X轴）和第二列（Y轴）互换，并且对第二列（Y轴）取反方向，目的是将LLFF的相机坐标系变成OpenGL/NeRF的相机坐标系
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    # [3, 5, N_images] --> [N_images, 3, 5]
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    # [H, W, 3, N_images] --> [N_images, H, W, 3]
    # 这就是最终返回的图片数据
    images = np.moveaxis(imgs, -1, 0).astype(np.float32)
    # [2, N_images] --> [N_images, 2]
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # 这个缩放是针对face forward场景，为了配合NDC使用，因为NDC要求整个场景必须位于z=-near平面之后，
    # 这里是将场景到相机的最小距离被缩放为1（因为在NeRF中设定近平面为z=-1），再通过bd_factor确保距离大于1
    sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
    # 平移向量乘以缩放系数
    poses[:, :3, 3] *= sc
    # 边界范围乘以缩放系数
    # 这就是最终返回的边界数据
    bds *= sc
    # 是否做归一化
    if recenter:
        poses = recenter_poses(poses)
    # 是否将相机视角做球形归一化并生成360度环绕新视角
    if spherify:
        # poses: [N, 3, 5]
        # render_poses: [120, 3, 5]
        # bds: [N, 2]
        poses, render_poses, bds = spherify_poses(poses, bds)
    # 适用于前向场景
    else:
        # 获取所有相机的平均位姿作为新的坐标系
        c2w = poses_avg(poses)
        # 获取所有相机位姿up向量的平均
        up = normalize_matrix(poses[:, :3, 1].sum(0))
        # 得到一个适合的焦点
        close_depth, inf_depth = bds.min() * 0.9, bds.max() * 2.
        dt = 0.5
        # 源代码是通过视差取焦点，经实验，直接通过深度也可以
        # focal = 1./((1.-dt)/close_depth + dt/inf_depth)
        focal = close_depth + (inf_depth - close_depth) * dt

        # 各相机的位置
        pts = poses[:, :3, 3]
        # 得到xyz都分别在90%位置的一个点
        rads = np.percentile(np.abs(pts), 90, 0)
        c2w_path = c2w
        # 生成120个视角
        N_views = 120
        # 两圈
        N_rots = 2
        # 是否保持生成的相机原点坐标z值为0，仅在xy平面上生成
        if path_zflat:
            # zloc = -close_depth * 0.1
            # 将所有相机都沿着z轴往靠近场景的方向移动一小点距离
            # c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            # z轴设为0
            rads[2] = 0.
            # 只生成1圈
            N_rots = 1
            # 只生成60个视角
            N_views /= 2
        # 生成螺旋路径的新视角的位姿
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zrate=0.5, N_rots=N_rots, N_views=N_views)
    # 都转成np.float32格式
    render_poses = np.array(render_poses).astype(np.float32)
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses


if __name__ == '__main__':
    print('123')