import os
import numpy as np
import json
import cv2


DATA_PATH = "data/GenericMOT_JPEG_Sequence"
OUT_PATH = os.path.join(DATA_PATH, "annotations")

SPLITS = [
    "train_half",
    "val_half",
    "train",
]
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True


if __name__ == "__main__":

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        if split == "test":
            data_path = os.path.join(DATA_PATH, "test")
        else:
            data_path = os.path.join(DATA_PATH, "train")
        out_path = os.path.join(OUT_PATH, "{}.json".format(split))
        out = {
            "images": [],
            "annotations": [],
            "videos": [],
            "categories": [{"id": 1, "name": "pedestrian"}],
        }
        seqs = os.listdir(data_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_curr = 0
        tid_last = -1
        for seq in sorted(seqs):
            if ".DS_Store" in seq:
                continue
            
            video_cnt += 1 
            out["videos"].append({"id": video_cnt, "file_name": seq})
            seq_path = os.path.join(data_path, seq)
            img_path = os.path.join(seq_path, "img1")
            ann_path = os.path.join(seq_path, "gt/gt.txt")
            images = os.listdir(img_path)
            num_images = len([image for image in images if "jpg" in image])  # half and half

            if HALF_VIDEO and ("test" in split):
                image_range = [0, num_images // 2] if "train" in split else [num_images // 2 + 1, num_images - 1]
            else:
                image_range = [0, num_images - 1]

            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue
                
                img = cv2.imread(os.path.join(data_path, "{}/img1/{:06d}.jpg".format(seq, i)))
                height, width = img.shape[:2]
                image_info = {
                    "file_name": "{}/img1/{:06d}.jpg".format(seq, i),  
                    "id": image_cnt + i,  
                    "frame_id": i - image_range[0], 
                    "prev_image_id": image_cnt + i if i > 0 else -1, 
                    "next_image_id": image_cnt + i + 1 if i < num_images - 1 else -1,
                    "video_id": video_cnt,
                    "height": height,
                    "width": width,
                }
                out["images"].append(image_info)
            print("{}: {} images".format(seq, num_images))
            if split != "test":
               
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=",")
               
                if CREATE_SPLITTED_ANN and ("half" in split):
                    anns_out = np.array(
                        [
                            anns[i]
                            for i in range(anns.shape[0])
                            if int(anns[i][0]) - 1 >= image_range[0] and int(anns[i][0]) - 1 <= image_range[1]
                        ],
                        np.float32,
                    )
                    anns_out[:, 0] -= image_range[0]
                    gt_out = os.path.join(seq_path, "gt/gt_{}.txt".format(split))
                    fout = open(gt_out, "w")
                    for o in anns_out:
                        fout.write(
                            "{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n".format(
                                int(o[0]),
                                int(o[1]),
                                int(o[2]),
                                int(o[3]),
                                int(o[4]),
                                int(o[5]),
                                int(o[6]),
                                int(o[7]),
                                o[8],
                            )
                        )
                    fout.close()
                

                print("{} ann images".format(int(anns[:, 0].max())))
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                        continue
                    track_id = int(anns[i][1])
                    cat_id = int(anns[i][7])
                    ann_cnt += 1
                    if not ("15" in DATA_PATH):
                       
                        if not (int(anns[i][6]) == 1):  
                            continue
                        if int(anns[i][7]) in [3, 4, 5, 6, 9, 10, 11]:  
                            continue
                        if int(anns[i][7]) in [2, 7, 8, 12]:  
                            category_id = -1
                        else:
                            category_id = 1  
                            if not track_id == tid_last:
                                tid_curr += 1
                                tid_last = track_id
                    else:
                        category_id = 1
                    ann = {
                        "id": ann_cnt,
                        "category_id": category_id,
                        "image_id": image_cnt + frame_id,
                        "track_id": tid_curr,
                        "bbox": anns[i][2:6].tolist(),
                        "conf": float(anns[i][6]),
                        "iscrowd": 0,
                        "area": float(anns[i][4] * anns[i][5]),
                    }
                    out["annotations"].append(ann)
            image_cnt += num_images
            print(tid_curr, tid_last)
        print("loaded {} for {} images and {} samples".format(split, len(out["images"]), len(out["annotations"])))
        json.dump(out, open(out_path, "w"))
