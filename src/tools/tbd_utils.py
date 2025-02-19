from threading import Thread
import datetime
import cv2
import os
import numpy as np
import time


def RunTime(func):
    def wrapper(*args, **kw):
        local_time = time.time()
        res = func(*args, **kw)
        print('current Function [%s] run time is %.7f' %
              (func.__name__, time.time() - local_time))
        return res
    return wrapper


def tlwh2xyxy(bbox):
    x1, y1 = bbox[0], bbox[1]
    w, h = bbox[2], bbox[3]
    x2, y2 = x1+w, y1+h

    return x1, y1, x2, y2


def xyxy2tlwh(bbox):
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]

    return (x, y, w, h)


def IsInboundary(bbox_pos, boundary):
    x1, y1, x2, y2 = tlwh2xyxy(boundary)

    cx = bbox_pos[0] + (bbox_pos[2] - bbox_pos[0])/2
    cy = bbox_pos[1] + (bbox_pos[3] - bbox_pos[1])/2


    inboundary = 1 if (x1 <= cx <= x2) and (y1 <= cy <= y2) else 0

    return inboundary


def GetLastNLines(inputfile, n, line_width=100):
    if n == 0:
        return []

    filesize = os.path.getsize(inputfile)
    dat_file = open(inputfile, 'r')

    blocksize = n * line_width
    if filesize > blocksize:
        maxseekpoint = (filesize // blocksize)
        dat_file.seek((maxseekpoint-1)*blocksize)
    elif filesize:
        dat_file.seek(0, 0)
    lines = dat_file.readlines()

    dat_file.close()

    return lines[-n:]


def Txt2Numpy(txt):
    tmp_list = []
    for line in txt:
        line = line.strip("\n").split(",")
        tmp_list.append(list(map(float, line)))

    return np.array(tmp_list, dtype=int)


def SaveRecords(trackers, frame_id, results_path):
    records = []
    for ti in trackers:
        tlwh = xyxy2tlwh(ti[:4])
       
        records.append("%d,%d,%.2f,%.2f,%.2f,%.2f,1\n" % (
            frame_id, ti[-1], round(tlwh[0], 0), round(tlwh[1], 0), round(tlwh[2], 0), round(tlwh[3], 0)))

    if len(trackers) > 0:
        with open(results_path, 'a+') as f:
            f.writelines(records)


def ExtractInfoFromRecords(results, frame_idx):
    mask = results[:, 0].astype(np.int) <= frame_idx
    frame_track_ids = results[mask, 1].astype(np.int)
    boxes = results[mask, 2:6].astype(np.int)
    centors_x = (boxes[:, 0] + boxes[:, 2] / 2)
    centors_y = (boxes[:, 1] + boxes[:, 3] / 2)
    ids_centors = np.column_stack((results[mask, 0].astype(np.int), results[mask, 1].astype(np.int),
                                   centors_x, centors_y))
    id_min, id_max = min(frame_track_ids), max(frame_track_ids)

    id_range = (id_min, id_max)

    return id_range, ids_centors


def SaveImg(img_name, image, width, height, save_path, img_type=".jpg"):
    h, w, _ = image.shape
    if not (h == height and w == width):
        image = cv2.resize(image, dsize=(width, height))

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    save_img_path = os.path.join(save_path, img_name+img_type)
    cv2.imwrite(save_img_path, image)


def SaveInfo(info, result_path):
    record = ','.join(list(map(str, info)))

    with open(result_path, "a+") as f:
        f.write(record+"\n")


class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None
