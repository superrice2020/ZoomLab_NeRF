import numpy as np
import os
from poses.colmap_wrapper import run_colmap
import poses.colmap_read_model as read_model


'''
源代码路径：https://github.com/Fyusion/LLFF/tree/master/llff/poses
2023.11.20
输入图片，生成NeRF所需要的相机参数
如果没有下载colmap，需要先在https://github.com/colmap/colmap/releases下载COLMAP-3.8-windows-cuda.zip
还需要将colmap.bat所在的目录加入环境变量，在cmd执行'colmap.bat'成功则证明添加成功
项目文件夹下需要有‘images’文件夹，图片放在这里面

'''

'''
读取colmap的运行结果
'''
def load_colmap_data(realdir):
    # 读取colmap输出的相机参数文件
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    # 只有一个相机，把这个相机的参数取出来
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    # print( 'Cameras', len(cam))
    # 组成hwf矩阵
    h, w, f = cam.height, cam.width, cam.params[0]
    hwf = np.array([h,w,f]).reshape([3,1])

    # 读取每张图片对应的相机视角位姿数据（旋转平移矩阵）
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    # 获取所有的图片名字
    names = [imdata[k].name for k in imdata]
    for i in np.argsort(names):
        print(names[i], end='  ')
    print( 'Images #', len(names))
    # 存储排序后的图片在原列表中的位置，这是为了让最后的结果按图片顺序输出
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        # 获得旋转矩阵 [3, 3]
        R = im.qvec2rotmat()
        # 获得平移矩阵 [3, 1]
        t = im.tvec.reshape([3,1])
        # 合并成旋转平移矩阵 [4, 4]
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    # 从list变为数组 [匹配上的图片的数量, 4, 4]
    w2c_mats = np.stack(w2c_mats, 0)
    # 对旋转平移矩阵取逆，得到相机坐标系到世界坐标系的转换矩阵
    c2w_mats = np.linalg.inv(w2c_mats)
    # 去掉底下的[0,0,0,1]，并将N放到最后一个维度 [3, 4, 匹配上的图片的数量]
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    # [3, 4, N]与[3, 1, N]拼接在一起，得到[3, 5, N]
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    # 读取点文件
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    # 包含了所有的点的数据，包括了点的xyz以及点所关联的图片id
    pts3d = read_model.read_points3d_binary(points3dfile)
    
    # colmap坐标系转换到llff坐标系： [x, y, z] --> [y, x, -z]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    return poses, pts3d, perm


'''
读取colmap的输出之后转换成nerf需要的格式并保存成poses_bounds.npy文件
'''
def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    inds = []
    # 获取图片文件夹下的所有图片名字，然后判断是否有图片没有匹配上
    images = os.listdir(os.path.join(basedir, 'images'))
    for k in pts3d:
        for ind in pts3d[k].image_ids:
            inds.append(ind)
    inds = list(set(inds))
    inds = [i - 1 for i in inds]
    if len(images) != poses.shape[-1]:
        ids = list(range(0, len(images), 1))
        ids_unused = [i for i in ids if i not in inds]
        images_unused = [images[i] for i in ids_unused]
        print(f'以下{len(images_unused)}张图片未匹配上，请删除“sparse”文件夹、colmap_output.txt、database.db以及这些图片后从头执行')
        print('images_unused:', images_unused)
        return
    # 遍历所有的点，记录每一个点与哪些相机位姿关联
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        # 如果有图片没有匹配上，那么生成的相机位姿数量就会少于图片数量导致这一步报错，所以上面要先删除未匹配上的图片
        for ind in pts3d[k].image_ids:
            # if len(cams) < ind - 1:
            #     print('ERROR: the correct camera poses for current points cannot be accessed')
            #     return
            cams[ind-1] = 1
        vis_arr.append(cams)
    # [点的数量, 3]
    pts_arr = np.array(pts_arr)
    # [点的数量, 匹配上的图片的数量] 用于记录每个点与哪些图片关联
    vis_arr = np.array(vis_arr)
    print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape)
    # 获取每个点在对应相机视角z轴上的位置，也就是深度值 [点的数量, 匹配上的图片的数量]
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    
    save_arr = []
    # 按顺序遍历每一张图片
    for i in perm:
        # [点的数量]，每一项是该点是否与当前图片有关联
        vis = vis_arr[:, i]
        # [点的数量]，每一项是点所处的深度
        zs = zvals[:, i]
        # 筛选出与当前图片有关联的点
        zs = zs[vis==1]
        # 获得最近的点和最远的点, [2]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        # ravel: [3, 5] --> [15]
        # concat: [17] 每张图片对应的[R(9),T(3),hwf(3),bd(2)]
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)
    
    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)
    print('已生成poses_bounds.npy文件')

'''
总的方法，包括运行colmap、读取colmap数据以及将数据转化为NeRF的形式进行保存
run_colmap这一步也可以用colmap的可视化界面生成，参考https://zhuanlan.zhihu.com/p/576416530
'''
def gen_poses(basedir, match_type):
    # 判断项目文件夹下的sparse/0文件夹中是否有数据，有的话就不要运行colmap，直接执行后处理，否则需要执行colmap
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []
    if not all([f in files_had for f in files_needed]):
        print('未在sparse/0文件夹内发现cameras.bin,images.bin和points3D.bin文件，正在运行colmap')
        run_colmap(basedir, match_type)
    else:
        print('无需运行colmap')
        
    print('colmap后处理')
    # 读取colmap数据
    poses, pts3d, perm = load_colmap_data(basedir)
    # 将数据转换成NeRF格式并保存
    save_poses(basedir, poses, pts3d, perm)
    return True
    
if __name__ == '__main__':
    # 传入的路径下需要有images文件夹，图片放在此文件夹内
    gen_poses(r'D:\code\Nerf\nerf_test', match_type='exhaustive_matcher')