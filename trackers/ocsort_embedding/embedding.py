from collections import OrderedDict
from pathlib import Path
import os
import pickle

import torch
import cv2
import torchvision
import torchreid
import numpy as np

from external.adaptors.fastreid_adaptor import FastReID


class EmbeddingComputer:
    def __init__(self, dataset, test_dataset, max_batch=1024):
        self.model = None
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.crop_size = (128, 384) 
        
        os.makedirs("./cache/embeddings/", exist_ok=True)
        self.cache_path = "./cache/embeddings/{}_embedding.pkl"
        self.cache = {}
        self.cache_name = ""
        self.max_batch = max_batch

        self.normalize = False

    def load_cache(self, path):
        
        self.cache_name = path
        cache_path = self.cache_path.format(path)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fp:
                self.cache = pickle.load(fp)

    def get_horizontal_split_patches(self, image, bbox, tag, idx, viz=False):
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            h, w = image.shape[2:]

        bbox = np.array(bbox)
        bbox = bbox.astype(int)
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > w or bbox[3] > h:
         
            bbox[0] = np.clip(bbox[0], 0, None)
            bbox[1] = np.clip(bbox[1], 0, None)
            bbox[2] = np.clip(bbox[2], 0, image.shape[1])
            bbox[3] = np.clip(bbox[3], 0, image.shape[0])

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
    
        split_boxes = [
            [x1, y1, x1 + w, y1 + h / 3],
            [x1, y1 + h / 3, x1 + w, y1 + (2 / 3) * h],
            [x1, y1 + (2 / 3) * h, x1 + w, y1 + h],
        ]

        split_boxes = np.array(split_boxes, dtype="int")
        patches = []
       
        for ix, patch_coords in enumerate(split_boxes):
            if isinstance(image, np.ndarray):
                
                im1 = image[patch_coords[1]: patch_coords[3],
                            patch_coords[0]: patch_coords[2], :]
               
                if viz:  
                    dirs = "./viz/{}/{}".format(tag.split(":")
                                                [0], tag.split(":")[1])
                    Path(dirs).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(dirs, "{}_{}.png".format(idx, ix)),
                        im1.squeeze(0).permute(
                            1, 2, 0).detach().cpu().numpy() * 255,
                    )
                patch = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
                patch = cv2.resize(patch, self.crop_size,
                                   interpolation=cv2.INTER_LINEAR)
                patch = torch.as_tensor(
                    patch.astype("float32").transpose(2, 0, 1))
                patch = patch.unsqueeze(0)
              
                patches.append(patch)
            else:
                im1 = image[:, :, patch_coords[1]: patch_coords[3],
                            patch_coords[0]: patch_coords[2]]
                patch = torchvision.transforms.functional.resize(
                    im1, (256, 128))
                patches.append(patch)

        patches = torch.cat(patches, dim=0)

        

        return patches

    def compute_embedding(self, img, bbox, tag):
       
        if self.cache_name != tag.split(":")[0]:
       
            self.load_cache(tag.split(":")[0])
       
        if tag in self.cache:
          
            embs = self.cache[tag]
            if embs.shape[0] != bbox.shape[0]:
                raise RuntimeError(
                    "ERROR: The number of cached embeddings don't match the "
                    "number of detections.\nWas the detector model changed? Delete cache if so."
                )
            return embs

        if self.model is None:
            # 新旧数据切换
            print('seq1: ', tag,tag[5:7])
            self.initialize_model(tag)

        crops = []
      
       
       
        h, w = img.shape[:2]
        results = np.round(bbox).astype(np.int32)
        results[:, 0] = results[:, 0].clip(0, w)
        results[:, 1] = results[:, 1].clip(0, h)
        results[:, 2] = results[:, 2].clip(0, w)
        results[:, 3] = results[:, 3].clip(0, h)

        crops = []
        for p in results:
            
            crop = img[p[1]: p[3], p[0]: p[2]]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(
                crop, self.crop_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
            if self.normalize:
                
                crop /= 255
                crop -= np.array((0.485, 0.456, 0.406))
                crop /= np.array((0.229, 0.224, 0.225))
            crop = torch.as_tensor(crop.transpose(2, 0, 1))
            crop = crop.unsqueeze(0)
            crops.append(crop)
        
        crops = torch.cat(crops, dim=0)

        embs = []

        for idx in range(0, len(crops), self.max_batch):
            batch_crops = crops[idx: idx + self.max_batch]
            batch_crops = batch_crops.cuda()

            with torch.no_grad():

                batch_embs = self.model(batch_crops)

            embs.extend(batch_embs)
        embs = torch.stack(embs)
        embs = torch.nn.functional.normalize(embs, dim=-1)

        embs = embs.cpu().numpy()

        self.cache[tag] = embs

        return embs

    def initialize_model(self,tag):

        if self.dataset == "mot17":
            if self.test_dataset:
                path = "external/weights/mot17_sbs_S50.pth"

            else:
               
                path = "external/weights/mot17_sbs_S50.pth"
                
        elif self.dataset == "mot20":
            if self.test_dataset:
                path = "external/weights/mot20_sbs_S50.pth"
            else:
                path = "external/weights/mot20_sbs_S50.pth"
                
        elif self.dataset == "dance":
            path = "external/weights/dance_sbs_S50.pth"
        elif self.dataset == "BEE24":
            path = "external/weights/bee24_AGW.pth"
                
        elif self.dataset == "gmot":
          
            path = "external/weights/gmot_AGW.pth"

        else:
            raise RuntimeError("Need the path for a new ReID model.")

        model = FastReID(path)
        model.eval()
        model.cuda()
        model.half()
        self.model = model

    def _get_general_model(self):
       
        model = torchreid.models.build_model(
            name="osnet_ain_x1_0", num_classes=2510, loss="softmax", pretrained=False)

        sd = torch.load(
            "external/weights/osnet_ain_ms_d_c.pth.tar")["state_dict"]

        new_state_dict = OrderedDict()
        for k, v in sd.items():
            name = k[7:] 
            new_state_dict[name] = v
       
        model.load_state_dict(new_state_dict)
        model.eval()
        model.cuda()
        self.model = model
        self.crop_size = (128, 256)
        
        self.normalize = True

    def dump_cache(self):
        if self.cache_name:
            with open(self.cache_path.format(self.cache_name), "wb") as fp:
                pickle.dump(self.cache, fp)
