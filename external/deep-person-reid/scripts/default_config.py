from yacs.config import CfgNode as CN


def get_default_config():
    cfg = CN()

    cfg.model = CN()
    cfg.model.name = "resnet50"
    cfg.model.pretrained = (
        True 
    )
    cfg.model.load_weights = ""  
    cfg.model.resume = ""

    cfg.data = CN()
    cfg.data.type = "image"
    cfg.data.root = "reid-data"
    cfg.data.sources = ["market1501"]
    cfg.data.targets = ["market1501"]
    cfg.data.workers = 4 
    cfg.data.split_id = 0 
    cfg.data.height = 256 
    cfg.data.width = 128 
    cfg.data.combineall = False 
    cfg.data.transforms = ["random_flip"] 
    cfg.data.k_tfm = (
        1 
    )
    cfg.data.norm_mean = [0.485, 0.456, 0.406] 
    cfg.data.norm_std = [0.229, 0.224, 0.225] 
    cfg.data.save_dir = "log" 
    cfg.data.load_train_targets = False 

    cfg.market1501 = CN()
    cfg.market1501.use_500k_distractors = (
        False 
    )
    cfg.cuhk03 = CN()
    cfg.cuhk03.labeled_images = (
        False 
    )
    cfg.cuhk03.classic_split = False 
    cfg.cuhk03.use_metric_cuhk03 = False 

    cfg.sampler = CN()
    cfg.sampler.train_sampler = "RandomSampler" 
    cfg.sampler.train_sampler_t = "RandomSampler"
    cfg.sampler.num_instances = (
        4 
    )
    cfg.sampler.num_cams = (
        1 
    )
    cfg.sampler.num_datasets = (
        1 
    )

    cfg.video = CN()
    cfg.video.seq_len = 15  
    cfg.video.sample_method = "evenly"  
    cfg.video.pooling_method = "avg"  

    cfg.train = CN()
    cfg.train.optim = "adam"
    cfg.train.lr = 0.0003
    cfg.train.weight_decay = 5e-4
    cfg.train.max_epoch = 60
    cfg.train.start_epoch = 0
    cfg.train.batch_size = 32
    cfg.train.fixbase_epoch = 0 
    cfg.train.open_layers = [
        "classifier"
    ] 
    cfg.train.staged_lr = False 
    cfg.train.new_layers = ["classifier"] 
    cfg.train.base_lr_mult = 0.1 
    cfg.train.lr_scheduler = "single_step"
    cfg.train.stepsize = [20] 
    cfg.train.gamma = 0.1 
    cfg.train.print_freq = 20 
    cfg.train.seed = 1 

    cfg.sgd = CN()
    cfg.sgd.momentum = 0.9 
    cfg.sgd.dampening = 0.0
    cfg.sgd.nesterov = False
    cfg.rmsprop = CN()
    cfg.rmsprop.alpha = 0.99
    cfg.adam = CN()
    cfg.adam.beta1 = 0.9
    cfg.adam.beta2 = 0.999

    cfg.loss = CN()
    cfg.loss.name = "softmax"
    cfg.loss.softmax = CN()
    cfg.loss.softmax.label_smooth = True 
    cfg.loss.triplet = CN()
    cfg.loss.triplet.margin = 0.3
    cfg.loss.triplet.weight_t = 1.0
    cfg.loss.triplet.weight_x = 0.0 

    cfg.test = CN()
    cfg.test.batch_size = 100
    cfg.test.dist_metric = "euclidean"
    cfg.test.normalize_feature = (
        False 
    )
    cfg.test.ranks = [1, 5, 10, 20] 
    cfg.test.evaluate = False 
    cfg.test.eval_freq = (
        -1
    ) 
    cfg.test.start_eval = 0 
    cfg.test.rerank = False
    cfg.test.visrank = (
        False
    )
    cfg.test.visrank_topk = 10

    return cfg


def imagedata_kwargs(cfg):
    return {
        "root": cfg.data.root,
        "sources": cfg.data.sources,
        "targets": cfg.data.targets,
        "height": cfg.data.height,
        "width": cfg.data.width,
        "transforms": cfg.data.transforms,
        "k_tfm": cfg.data.k_tfm,
        "norm_mean": cfg.data.norm_mean,
        "norm_std": cfg.data.norm_std,
        "use_gpu": cfg.use_gpu,
        "split_id": cfg.data.split_id,
        "combineall": cfg.data.combineall,
        "load_train_targets": cfg.data.load_train_targets,
        "batch_size_train": cfg.train.batch_size,
        "batch_size_test": cfg.test.batch_size,
        "workers": cfg.data.workers,
        "num_instances": cfg.sampler.num_instances,
        "num_cams": cfg.sampler.num_cams,
        "num_datasets": cfg.sampler.num_datasets,
        "train_sampler": cfg.sampler.train_sampler,
        "train_sampler_t": cfg.sampler.train_sampler_t,
        "cuhk03_labeled": cfg.cuhk03.labeled_images,
        "cuhk03_classic_split": cfg.cuhk03.classic_split,
        "market1501_500k": cfg.market1501.use_500k_distractors,
    }


def videodata_kwargs(cfg):
    return {
        "root": cfg.data.root,
        "sources": cfg.data.sources,
        "targets": cfg.data.targets,
        "height": cfg.data.height,
        "width": cfg.data.width,
        "transforms": cfg.data.transforms,
        "norm_mean": cfg.data.norm_mean,
        "norm_std": cfg.data.norm_std,
        "use_gpu": cfg.use_gpu,
        "split_id": cfg.data.split_id,
        "combineall": cfg.data.combineall,
        "batch_size_train": cfg.train.batch_size,
        "batch_size_test": cfg.test.batch_size,
        "workers": cfg.data.workers,
        "num_instances": cfg.sampler.num_instances,
        "num_cams": cfg.sampler.num_cams,
        "num_datasets": cfg.sampler.num_datasets,
        "train_sampler": cfg.sampler.train_sampler,
        "seq_len": cfg.video.seq_len,
        "sample_method": cfg.video.sample_method,
    }


def optimizer_kwargs(cfg):
    return {
        "optim": cfg.train.optim,
        "lr": cfg.train.lr,
        "weight_decay": cfg.train.weight_decay,
        "momentum": cfg.sgd.momentum,
        "sgd_dampening": cfg.sgd.dampening,
        "sgd_nesterov": cfg.sgd.nesterov,
        "rmsprop_alpha": cfg.rmsprop.alpha,
        "adam_beta1": cfg.adam.beta1,
        "adam_beta2": cfg.adam.beta2,
        "staged_lr": cfg.train.staged_lr,
        "new_layers": cfg.train.new_layers,
        "base_lr_mult": cfg.train.base_lr_mult,
    }


def lr_scheduler_kwargs(cfg):
    return {
        "lr_scheduler": cfg.train.lr_scheduler,
        "stepsize": cfg.train.stepsize,
        "gamma": cfg.train.gamma,
        "max_epoch": cfg.train.max_epoch,
    }


def engine_run_kwargs(cfg):
    return {
        "save_dir": cfg.data.save_dir,
        "max_epoch": cfg.train.max_epoch,
        "start_epoch": cfg.train.start_epoch,
        "fixbase_epoch": cfg.train.fixbase_epoch,
        "open_layers": cfg.train.open_layers,
        "start_eval": cfg.test.start_eval,
        "eval_freq": cfg.test.eval_freq,
        "test_only": cfg.test.evaluate,
        "print_freq": cfg.train.print_freq,
        "dist_metric": cfg.test.dist_metric,
        "normalize_feature": cfg.test.normalize_feature,
        "visrank": cfg.test.visrank,
        "visrank_topk": cfg.test.visrank_topk,
        "use_metric_cuhk03": cfg.cuhk03.use_metric_cuhk03,
        "ranks": cfg.test.ranks,
        "rerank": cfg.test.rerank,
    }
