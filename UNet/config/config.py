import os
import os.path as osp
import yaml


def _check_dir(dir, make_dir=True):
    if not osp.exists(dir):
        if make_dir:
            print('Create directory {}'.format(dir))
            os.mkdir(dir)
        else:
            raise Exception('Directory not exist: {}'.format(dir))


def get_train_config(root_path, config_file='config/train_config.yaml'):
    with open(os.path.join(root_path, config_file), 'r') as f:
        cfg = yaml.load(f)

    _check_dir(os.path.join(root_path,cfg['dataset']['data_root']), make_dir=False)
    _check_dir(os.path.join(root_path,cfg['ckpt_root']))

    return cfg


def get_test_config(root_path, config_file='config/test_config.yaml'):
    with open(os.path.join(root_path, config_file), 'r') as f:
        cfg = yaml.load(f)

    _check_dir(os.path.join(root_path,cfg['dataset']['data_root']), make_dir=False)

    return cfg
