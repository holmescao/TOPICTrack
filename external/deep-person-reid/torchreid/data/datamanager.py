from __future__ import division, print_function, absolute_import
import torch

from torchreid.data.sampler import build_train_sampler
from torchreid.data.datasets import init_image_dataset, init_video_dataset
from torchreid.data.transforms import build_transforms


class DataManager(object):
    r"""Base data manager.

    Args:
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(
        self,
        sources=None,
        targets=None,
        height=256,
        width=128,
        transforms="random_flip",
        norm_mean=None,
        norm_std=None,
        use_gpu=False,
    ):
        self.sources = sources
        self.targets = targets
        self.height = height
        self.width = width

        if self.sources is None:
            raise ValueError("sources must not be None")

        if isinstance(self.sources, str):
            self.sources = [self.sources]

        if self.targets is None:
            self.targets = self.sources

        if isinstance(self.targets, str):
            self.targets = [self.targets]

        self.transform_tr, self.transform_te = build_transforms(
            self.height,
            self.width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.use_gpu = torch.cuda.is_available() and use_gpu

    @property
    def num_train_pids(self):
        """Returns the number of training person identities."""
        return self._num_train_pids

    @property
    def num_train_cams(self):
        """Returns the number of training cameras."""
        return self._num_train_cams

    def fetch_test_loaders(self, name):
        """Returns query and gallery of a test dataset, each containing
        tuples of (img_path(s), pid, camid).

        Args:
            name (str): dataset name.
        """
        query_loader = self.test_dataset[name]["query"]
        gallery_loader = self.test_dataset[name]["gallery"]
        return query_loader, gallery_loader

    def preprocess_pil_img(self, img):
 
        return self.transform_te(img)


class ImageDataManager(DataManager):
    data_type = "image"

    def __init__(
        self,
        root="",
        sources=None,
        targets=None,
        height=256,
        width=128,
        transforms="random_flip",
        k_tfm=1,
        norm_mean=None,
        norm_std=None,
        use_gpu=True,
        split_id=0,
        combineall=False,
        load_train_targets=False,
        batch_size_train=32,
        batch_size_test=32,
        workers=4,
        num_instances=4,
        num_cams=1,
        num_datasets=1,
        train_sampler="RandomSampler",
        train_sampler_t="RandomSampler",
        cuhk03_labeled=False,
        cuhk03_classic_split=False,
        market1501_500k=False,
    ):

        super(ImageDataManager, self).__init__(
            sources=sources,
            targets=targets,
            height=height,
            width=width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std,
            use_gpu=use_gpu,
        )

        print("=> Loading train (source) dataset")
        trainset = []
        for name in self.sources:
            trainset_ = init_image_dataset(
                name,
                transform=self.transform_tr,
                k_tfm=k_tfm,
                mode="train",
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k,
            )
            trainset.append(trainset_)
        trainset = sum(trainset)

        self._num_train_pids = trainset.num_train_pids
        self._num_train_cams = trainset.num_train_cams

        self.train_loader = torch.utils.data.DataLoader(
            trainset,
            sampler=build_train_sampler(
                trainset.train,
                train_sampler,
                batch_size=batch_size_train,
                num_instances=num_instances,
                num_cams=num_cams,
                num_datasets=num_datasets,
            ),
            batch_size=batch_size_train,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True,
        )

        self.train_loader_t = None
        if load_train_targets:
            assert (
                len(set(self.sources) & set(self.targets)) == 0
            ), "sources={} and targets={} must not have overlap".format(
                self.sources, self.targets
            )

            print("=> Loading train (target) dataset")
            trainset_t = []
            for name in self.targets:
                trainset_t_ = init_image_dataset(
                    name,
                    transform=self.transform_tr,
                    k_tfm=k_tfm,
                    mode="train",
                    combineall=False,
                    root=root,
                    split_id=split_id,
                    cuhk03_labeled=cuhk03_labeled,
                    cuhk03_classic_split=cuhk03_classic_split,
                    market1501_500k=market1501_500k,
                )
                trainset_t.append(trainset_t_)
            trainset_t = sum(trainset_t)

            self.train_loader_t = torch.utils.data.DataLoader(
                trainset_t,
                sampler=build_train_sampler(
                    trainset_t.train,
                    train_sampler_t,
                    batch_size=batch_size_train,
                    num_instances=num_instances,
                    num_cams=num_cams,
                    num_datasets=num_datasets,
                ),
                batch_size=batch_size_train,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=True,
            )

        print("=> Loading test (target) dataset")
        self.test_loader = {
            name: {"query": None, "gallery": None} for name in self.targets
        }
        self.test_dataset = {
            name: {"query": None, "gallery": None} for name in self.targets
        }

        for name in self.targets:
            queryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode="query",
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k,
            )
            self.test_loader[name]["query"] = torch.utils.data.DataLoader(
                queryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False,
            )

            galleryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode="gallery",
                combineall=combineall,
                verbose=False,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k,
            )
            self.test_loader[name]["gallery"] = torch.utils.data.DataLoader(
                galleryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False,
            )

            self.test_dataset[name]["query"] = queryset.query
            self.test_dataset[name]["gallery"] = galleryset.gallery

        print("\n")
        print("  **************** Summary ****************")
        print("  source            : {}".format(self.sources))
        print("  # source datasets : {}".format(len(self.sources)))
        print("  # source ids      : {}".format(self.num_train_pids))
        print("  # source images   : {}".format(len(trainset)))
        print("  # source cameras  : {}".format(self.num_train_cams))
        if load_train_targets:
            print("  # target images   : {} (unlabeled)".format(len(trainset_t)))
        print("  target            : {}".format(self.targets))
        print("  *****************************************")
        print("\n")


class VideoDataManager(DataManager):
    data_type = "video"

    def __init__(
        self,
        root="",
        sources=None,
        targets=None,
        height=256,
        width=128,
        transforms="random_flip",
        norm_mean=None,
        norm_std=None,
        use_gpu=True,
        split_id=0,
        combineall=False,
        batch_size_train=3,
        batch_size_test=3,
        workers=4,
        num_instances=4,
        num_cams=1,
        num_datasets=1,
        train_sampler="RandomSampler",
        seq_len=15,
        sample_method="evenly",
    ):

        super(VideoDataManager, self).__init__(
            sources=sources,
            targets=targets,
            height=height,
            width=width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std,
            use_gpu=use_gpu,
        )

        print("=> Loading train (source) dataset")
        trainset = []
        for name in self.sources:
            trainset_ = init_video_dataset(
                name,
                transform=self.transform_tr,
                mode="train",
                combineall=combineall,
                root=root,
                split_id=split_id,
                seq_len=seq_len,
                sample_method=sample_method,
            )
            trainset.append(trainset_)
        trainset = sum(trainset)

        self._num_train_pids = trainset.num_train_pids
        self._num_train_cams = trainset.num_train_cams

        train_sampler = build_train_sampler(
            trainset.train,
            train_sampler,
            batch_size=batch_size_train,
            num_instances=num_instances,
            num_cams=num_cams,
            num_datasets=num_datasets,
        )

        self.train_loader = torch.utils.data.DataLoader(
            trainset,
            sampler=train_sampler,
            batch_size=batch_size_train,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True,
        )

        print("=> Loading test (target) dataset")
        self.test_loader = {
            name: {"query": None, "gallery": None} for name in self.targets
        }
        self.test_dataset = {
            name: {"query": None, "gallery": None} for name in self.targets
        }

        for name in self.targets:

            queryset = init_video_dataset(
                name,
                transform=self.transform_te,
                mode="query",
                combineall=combineall,
                root=root,
                split_id=split_id,
                seq_len=seq_len,
                sample_method=sample_method,
            )
            self.test_loader[name]["query"] = torch.utils.data.DataLoader(
                queryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False,
            )

            galleryset = init_video_dataset(
                name,
                transform=self.transform_te,
                mode="gallery",
                combineall=combineall,
                verbose=False,
                root=root,
                split_id=split_id,
                seq_len=seq_len,
                sample_method=sample_method,
            )
            self.test_loader[name]["gallery"] = torch.utils.data.DataLoader(
                galleryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False,
            )

            self.test_dataset[name]["query"] = queryset.query
            self.test_dataset[name]["gallery"] = galleryset.gallery

        print("\n")
        print("  **************** Summary ****************")
        print("  source             : {}".format(self.sources))
        print("  # source datasets  : {}".format(len(self.sources)))
        print("  # source ids       : {}".format(self.num_train_pids))
        print("  # source tracklets : {}".format(len(trainset)))
        print("  # source cameras   : {}".format(self.num_train_cams))
        print("  target             : {}".format(self.targets))
        print("  *****************************************")
        print("\n")
