

import sys
import os
import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

TRAIN_DIR_KEY = 'train_dir'
TEST_DIR_KEY = 'test_dir'
VERSION_DICT = {
    'MSMT17_V1': {
        TRAIN_DIR_KEY: 'train',
        TEST_DIR_KEY: 'test',
    },
    'MSMT17_V2': {
        TRAIN_DIR_KEY: 'mask_train_v2',
        TEST_DIR_KEY: 'mask_test_v2',
    }
}


@DATASET_REGISTRY.register()
class MSMT17(ImageDataset):

    dataset_url = None
    dataset_name = 'msmt17'

    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = root

        has_main_dir = False
        for main_dir in VERSION_DICT:
            if osp.exists(osp.join(self.dataset_dir, main_dir)):
                train_dir = VERSION_DICT[main_dir][TRAIN_DIR_KEY]
                test_dir = VERSION_DICT[main_dir][TEST_DIR_KEY]
                has_main_dir = True
                break
        assert has_main_dir, 'Dataset folder not found'

        self.train_dir = osp.join(self.dataset_dir, main_dir, train_dir)
        self.test_dir = osp.join(self.dataset_dir, main_dir, test_dir)
        self.list_train_path = osp.join(self.dataset_dir, main_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, main_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, main_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, main_dir, 'list_gallery.txt')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.test_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, self.list_train_path)
        val = self.process_dir(self.train_dir, self.list_val_path)
        query = self.process_dir(self.test_dir, self.list_query_path, is_train=False)
        gallery = self.process_dir(self.test_dir, self.list_gallery_path, is_train=False)

        num_train_pids = self.get_num_pids(train)
        query_tmp = []
        for img_path, pid, camid in query:
            query_tmp.append((img_path, pid+num_train_pids, camid))
        del query
        query = query_tmp

        gallery_temp = []
        for img_path, pid, camid in gallery:
            gallery_temp.append((img_path, pid+num_train_pids, camid))
        del gallery
        gallery = gallery_temp

        if 'combineall' in kwargs and kwargs['combineall']:
            train += val
        super(MSMT17, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, list_path, is_train=True):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        data = []

        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  
            camid = int(img_path.split('_')[2]) - 1  
            img_path = osp.join(dir_path, img_path)
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
