'''
Author: your name
Date: 2021-05-31 20:14:18
LastEditTime: 2022-01-08 14:22:40
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /unbox_yolov5_deepsort_counting/utils/visualization.py
'''

import datetime
import cv2
import colorsys
from tools import tbd_utils


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


def rectangle(image, x1, y1, x2, y2, label, color):
    """Draw a rectangle.
    """
    thickness = round(0.001 * (image.shape[0] + image.shape[1]) / 2) + 1

    pt1 = int(x1), int(y1)
    pt2 = int(x2), int(y2)
    cv2.rectangle(image, pt1, pt2, color, 1)

    text_size = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_PLAIN, 1, 1)

    center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
    pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + \
        text_size[0][1]
    cv2.rectangle(image, pt1, pt2, color, -1)
    cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
                1, (255, 255, 255), 1)


def draw_bboxes(image, bboxes):

    for item_box in bboxes:
        x1, y1, x2, y2, _, pos_id = item_box
        color = create_unique_color_uchar(pos_id)

        rectangle(image, x1, y1, x2, y2, str(pos_id), color)

    return image


def draw_centors(image, id_range, ids_centors):
    for i in range(id_range[0], id_range[1]+1):
        id_mark = ids_centors[:, 1] == i
        frame_id_centors = ids_centors[id_mark, :]

        if len(frame_id_centors) == 0:
            continue

        id_centors = frame_id_centors[:, 1:]
        color = create_unique_color_uchar(i)
        for j in range(1, len(id_centors[:, 1])):
            cv2.line(image, (int(id_centors[j - 1, 1]), int(id_centors[j - 1, 2])),
                     (int(id_centors[j, 1]), int(id_centors[j, 2])), color, 1)

    return image


def SceneShow(img):
    cv2.imshow('demo', img)
    cv2.waitKey(1)


def DrawStatistic(im, text_draw):

    h, w, _ = im.shape
    y0 = int(h * 0.05)
    dy = 30
    for i, text in enumerate(text_draw.split('\n')):
        y = y0+i*dy
        im = cv2.putText(img=im,
                         text=text,
                         org=(int(w * 0.01), y),
                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                         fontScale=0.8,
                        
                         color=(0, 0, 255),
                         thickness=2)

    return im


def DrawTrackers(im, trackers,  frame_id,
                 n, results_txt_path):
   
    im = draw_bboxes(im, trackers)
   
    pre_records = tbd_utils.GetLastNLines(results_txt_path, n)
    if len(pre_records) > 0:
        results = tbd_utils.Txt2Numpy(pre_records)
        id_range, ids_centors = tbd_utils.ExtractInfoFromRecords(
            results, frame_id)
        im = draw_centors(im, id_range, ids_centors)

    return im


def DrawAdditional(im, trackers, boundary, frame_id,
                   n, results_txt_path, total_entry, total_out):
   
    im = draw_bboxes(im, trackers)
   
    pre_records = tbd_utils.GetLastNLines(results_txt_path, n)
    if len(pre_records) > 0:
        results = tbd_utils.Txt2Numpy(pre_records)
        id_range, ids_centors = tbd_utils.ExtractInfoFromRecords(
            results, frame_id)
        im = draw_centors(im, id_range, ids_centors)
 
    x1, y1, x2, y2 = tbd_utils.tlwh2xyxy(boundary)
    im = cv2.rectangle(im, (x1, y1), (x2, y2),
                       color=(0, 0, 255), thickness=5)
   

    return im
