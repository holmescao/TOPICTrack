import os
import pdb

import torch
import cv2
import numpy as np
from pycocotools.coco import COCO
from torchvision import transforms


def get_mot_loader(dataset, test, data_dir="data", workers=4, size=(800, 1440)):
    if dataset == "mot17":
        direc = "mot"
        if test:
            name = "test"
            annotation = "test.json"
        else:
            name = "train"
            annotation = "val_half.json"
    elif dataset == "mot20":
        direc = "MOT20"
        if test:
            name = "test"
            annotation = "test.json"
        else:
            name = "train"
            annotation = "val_half.json"
    elif dataset == "dance":
        direc = "dancetrack"
        if test:
            name = "test"
            annotation = "test.json"
        else:
            annotation = "val.json"
            name = "val"
    elif dataset == "BEE24":
        direc = "BEE24"
        if test:
            name = "test"
            annotation = "test.json"
        else:
            annotation = "val.json"
            name = "val"
    elif dataset == "gmot":
        direc = "gmot"
        if test:
            name = "test"
            annotation = "test.json"
        else:
            annotation = "val.json"
            name = "val"

    else:
        raise RuntimeError("Specify path here.")

    valdataset = MOTDataset(
        data_dir=os.path.join(data_dir, direc),
        json_file=annotation,
        img_size=size,
        name=name,
        preproc=ValTransform(
            # rgb_means=(0.485, 0.456, 0.406),
            # std=(0.229, 0.224, 0.225),
            rgb_means=(0.491, 0.461, 0.406),
            std=(0.240, 0.220, 0.208),
        )
    )

    sampler = torch.utils.data.SequentialSampler(valdataset)
    dataloader_kwargs = {
        "num_workers": workers,
        "pin_memory": True,
        "sampler": sampler,
    }
    dataloader_kwargs["batch_size"] = 1
    val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    return val_loader


class MOTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        json_file="train_half.json",
        name="train",
        img_size=(608, 1088),
        preproc=None,
    ):
        self.input_dim = img_size
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        self.name = name
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = obj["track_id"]

        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"
        img_info = (height, width, frame_id, video_id, file_name)

        del im_ann, annotations

        return (res, img_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, file_name = self.annotations[index]
       
        img_file = os.path.join(self.data_dir, self.name, file_name)
        img = cv2.imread(img_file)

        assert img is not None

        return img, res.copy(), img_info, np.array([id_])

    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)
        tensor, target = self.preproc(img, target, self.input_dim)
        return (tensor, img), target, img_info, img_id


class ValTransform:
    def __init__(self, rgb_means=None, std=None, swap=(2, 0, 1)):
        self.means = rgb_means
        self.swap = swap
        self.std = std
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.means, self.std, self.swap)
        return img, np.zeros((1, 5))


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r
