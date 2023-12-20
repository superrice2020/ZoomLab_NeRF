import os
import subprocess



# $ DATASET_PATH=/path/to/dataset

# 图像特征提取
# $ colmap feature_extractor \
#    --database_path $DATASET_PATH/database.db \
#    --image_path $DATASET_PATH/images

# 图像特征匹配
# $ colmap exhaustive_matcher \
#    --database_path $DATASET_PATH/database.db


# $ mkdir $DATASET_PATH/sparse

# 稀疏三维重建
# $ colmap mapper \
#     --database_path $DATASET_PATH/database.db \
#     --image_path $DATASET_PATH/images \
#     --output_path $DATASET_PATH/sparse

# $ mkdir $DATASET_PATH/dense
def run_colmap(basedir, match_type):
    
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    
    feature_extractor_args = [
        'colmap.bat', 'feature_extractor',
            '--database_path', os.path.join(basedir, 'database.db'), 
            '--image_path', os.path.join(basedir, 'images'),
            '--ImageReader.camera_model', 'OPENCV',
            '--ImageReader.single_camera', '1',
            '--SiftExtraction.use_gpu', '0',
    ]
    feat_output = (subprocess.check_output(feature_extractor_args, universal_newlines=True))
    logfile.write(feat_output)
    print('Features extracted')

    exhaustive_matcher_args = [
        'colmap.bat', match_type,
        '--database_path', os.path.join(basedir, 'database.db'),
        '--SiftMatching.guided_matching', 'true',
    ]

    match_output = (subprocess.check_output(exhaustive_matcher_args, universal_newlines=True) )
    logfile.write(match_output)
    print('Features matched')
    
    p = os.path.join(basedir, 'sparse')
    if not os.path.exists(p):
        os.makedirs(p)

    # mapper_args = [
    #     'colmap', 'mapper', 
    #         '--database_path', os.path.join(basedir, 'database.db'), 
    #         '--image_path', os.path.join(basedir, 'images'),
    #         '--output_path', os.path.join(basedir, 'sparse'),
    #         '--Mapper.num_threads', '16',
    #         '--Mapper.init_min_tri_angle', '4',
    # ]
    mapper_args = [
        'colmap.bat', 'mapper',
            '--database_path', os.path.join(basedir, 'database.db'),
            '--image_path', os.path.join(basedir, 'images'),
            '--output_path', os.path.join(basedir, 'sparse'), # --export_path changed to --output_path in colmap 3.6
            '--Mapper.num_threads', '16',
            '--Mapper.init_min_tri_angle', '4',
            '--Mapper.multiple_models', '0',
            '--Mapper.extract_colors', '0',
    ]

    map_output = (subprocess.check_output(mapper_args, universal_newlines=True))
    logfile.write(map_output)

    # ba_args = [
    #     'colmap.bat', 'bundle_adjuster',
    #         '--input_path', os.path.join(basedir, 'sparse', '0'),
    #         '--output_path', os.path.join(basedir, 'sparse', '0'),
    #         '--BundleAdjustment.refine_principal_point', '1',
    # ]
    # ba_output = (subprocess.check_output(ba_args, universal_newlines=True))
    # logfile.write(ba_output)
    logfile.close()
    print('Sparse map created')
    
    print( 'Finished running COLMAP, see {} for logs'.format(logfile_name) )


