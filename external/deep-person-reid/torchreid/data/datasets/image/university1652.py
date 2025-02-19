from __future__ import division, print_function, absolute_import
import os
import glob
import os.path as osp
import gdown

from ..dataset import ImageDataset


class University1652(ImageDataset):
    dataset_dir = "university1652"
    dataset_url = "https://drive.google.com/uc?id=1iVnP4gjw-iHXa0KerZQ1IfIO0i1jADsR"

    def __init__(self, root="", **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        print(self.dataset_dir)
        if not os.path.isdir(self.dataset_dir):
            os.mkdir(self.dataset_dir)
            gdown.download(self.dataset_url, self.dataset_dir + "data.zip", quiet=False)
            os.system("unzip %s" % (self.dataset_dir + "data.zip"))
        self.train_dir = osp.join(self.dataset_dir, "University-Release/train/")
        self.query_dir = osp.join(
            self.dataset_dir, "University-Release/test/query_drone"
        )
        self.gallery_dir = osp.join(
            self.dataset_dir, "University-Release/test/gallery_satellite"
        )

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        self.fake_camid = 0
        train = self.process_dir(self.train_dir, relabel=True, train=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(University1652, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False, train=False):
        IMG_EXTENSIONS = (
            ".jpg",
            ".jpeg",
            ".png",
            ".ppm",
            ".bmp",
            ".pgm",
            ".tif",
            ".tiff",
            ".webp",
        )
        if train:
            img_paths = glob.glob(osp.join(dir_path, "*/*/*"))
        else:
            img_paths = glob.glob(osp.join(dir_path, "*/*"))
        pid_container = set()
        for img_path in img_paths:
            if not img_path.lower().endswith(IMG_EXTENSIONS):
                continue
            pid = int(os.path.basename(os.path.dirname(img_path)))
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        data = []
        for img_path in img_paths:
            if not img_path.lower().endswith(IMG_EXTENSIONS):
                continue
            pid = int(os.path.basename(os.path.dirname(img_path)))
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, self.fake_camid))
            self.fake_camid += 1
        return data
