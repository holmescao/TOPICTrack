from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from tqdm import tqdm
import os
import sys
import cv2
import json
import copy
import numpy as np
import warnings

from opts import opts
from detector import Detector
from tools import tbd_utils, visualization


warnings.filterwarnings("ignore")

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    detector = Detector(opt)

    if opt.demo == 'webcam' or \
            opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        is_video = True
       
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    else:
        is_video = False
      
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]

    out = None
    out_name = opt.demo[opt.demo.rfind('/') + 1:]
    if opt.save_video:
        if not os.path.exists('../results'):
            os.mkdir('../results')
      
        fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
        save_video_dir = '../results/{}/videos/'.format(
            opt.exp_id + '_' + out_name)
        if not os.path.exists(save_video_dir):
            os.makedirs(save_video_dir, exist_ok=True)
        save_seq_path = os.path.join(save_video_dir, out_name)

        out = cv2.VideoWriter(save_seq_path+".avi", fourcc, opt.save_framerate, (
            opt.input_w, opt.input_h))

    if opt.debug < 5:
        detector.pause = False
    cnt = 0
    results = {}

    remain_frame = 30
    last_n_frames_bboxes = np.zeros(remain_frame, np.int32)

    save_video_dir = '../results/{}/records/'.format(
        opt.exp_id + '_' + out_name)
    if not os.path.exists(save_video_dir):
        os.makedirs(save_video_dir, exist_ok=True)
    results_mot_file_path = os.path.join(save_video_dir, out_name+".txt")

    while True:
        if is_video:
            _, img = cam.read()
            if img is None:
                save_and_exit(opt, out, results, out_name)
        else:
            if cnt < len(image_names):
                img = cv2.imread(image_names[cnt])
            else:
               
                avi_to_mp4 = "ffmpeg -i %s.avi %s.mp4" % (
                    save_seq_path, save_seq_path)
                del_avi_file = "rm %s.avi" % (save_seq_path)
                os.system(avi_to_mp4)
                os.system(del_avi_file)
                out.release()
                return
              

        cnt += 1

        if opt.resize_video:
            img = cv2.resize(img, (opt.input_w, opt.input_h))

        if cnt < opt.skip_first:
            continue

        if not opt.save_video:
            cv2.imshow('input', img)

        ret = detector.run(img)

        time_str = '[{}] frame {} |'.format(opt.demo.split("/")[-1], cnt)
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)

        results[cnt] = ret['results']

        save_img_dir = "../results/%s/imgs/" % (opt.exp_id + '_' + out_name)
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir, exist_ok=True)
        if opt.save_video:
         
            trackers = []
            for tracklet in results[cnt]:
                trackers.append(list(tracklet['bbox']) +
                                list([tracklet['score']]) +
                                list([tracklet['tracking_id']]))

            frame_id = cnt-1
            tbd_utils.SaveRecords(trackers, frame_id, results_mot_file_path)
          
            last_n_frames_bboxes[frame_id % remain_frame] = len(trackers)
            n_lines = last_n_frames_bboxes.sum()
            img = visualization.DrawTrackers(img, trackers, frame_id,
                                             n_lines, results_mot_file_path)
           
            out.write(img)
         
            if not is_video:
                cv2.imwrite(save_img_dir+'{}.jpg'.format(cnt), img)
                
        if cv2.waitKey(1) == 27:
            save_and_exit(opt, out, results, out_name)
            return
    save_and_exit(opt, out, results)


def save_and_exit(opt, out=None, results=None, out_name=''):
    if opt.save_results and (results is not None):
        save_dir = '../results/{}_results.json'.format(
            opt.exp_id + '_' + out_name)
        print('saving results to', save_dir)
        json.dump(_to_list(copy.deepcopy(results)),
                  open(save_dir, 'w'))
    if opt.save_video and out is not None:
        out.release()
    sys.exit(0)


def _to_list(results):
    for img_id in results:
        for t in range(len(results[img_id])):
            for k in results[img_id][t]:
                if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
                    results[img_id][t][k] = results[img_id][t][k].tolist()
    return results


if __name__ == '__main__':
    data_root = "/home/caoxiaoyan/MOT_benchmark/BEE20_v2"
    seq_list = os.listdir(data_root)
    bar = tqdm(seq_list)
    for seq_name in bar:
        bar.set_description("Processing:%s" % seq_name)
        if "beev2" in seq_name:
            opt = opts().init()
            opt.demo = os.path.join(data_root, seq_name)
            demo(opt)
