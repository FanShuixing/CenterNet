from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory
import pandas as pd
import json

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    if opt.demo == 'webcam' or \
            opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        detector.pause = False
        while True:
            _, img = cam.read()
            cv2.imshow('input', img)
            ret = detector.run(img)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
            if cv2.waitKey(1) == 27:
                return  # esc to quit
    else:
        #         if os.path.isdir(opt.demo):
        with open('/input1/%s' % opt.demo) as fr:
            file_list = pd.read_csv(fr).values[:, 1]
            image_names = []
            for each in file_list:
                image_names.append(os.path.join('/input1', each))
        #             image_names = []
        #             ls = os.listdir(opt.demo)
        #             for file_name in sorted(ls):
        #                 ext = file_name[file_name.rfind('.') + 1:].lower()
        #                 if ext in image_ext:
        #                     image_names.append(os.path.join(opt.demo, file_name))

        total_dict = {}
        for (image_name) in image_names:
            name_no_suffix = image_name.split('/')[-1].replace('.jpg', '')
            with open('/input1/mask_labels/%s.json' % name_no_suffix) as fr:
                info = json.load(fr)
                gt_box = info['num_box']
            img = cv2.imread(image_name)
            h, w, _ = img.shape
            img = cv2.resize(img, (768, 576))

            info_dict = {}
            bboxes_json = []
            ret = detector.run(image_name)
            # 将输出结果写入到json中
            results = ret['results']
            for j in range(1, 2):
                for bbox in results[j]:
                    tmp = {}
                    if bbox[4] > opt.vis_thresh:
                        tmp['x_min'] = bbox[0] / w
                        tmp['y_min'] = bbox[1] / h
                        tmp['x_max'] = bbox[2] / w
                        tmp['y_max'] = bbox[3] / h
                        tmp['label'] = 'mucai'
                        tmp['confidence'] = 1
                        bboxes_json.append(tmp)
                        cv2.rectangle(img, (int(bbox[0] / w * 768), int(bbox[1] / h * 576)),
                                      (int(bbox[2] / w * 768), int(bbox[3] / h * 576)), (255, 0, 0), 2)
            cv2.imwrite('predict/%s_pred%s_gt%s.jpg' % (name_no_suffix, len(bboxes_json), gt_box), img)
            info_dict['image_height'] = 768
            info_dict['image_width'] = 576
            info_dict['num_box'] = len(bboxes_json)
            info_dict['bboxes'] = bboxes_json
            total_dict[name_no_suffix] = info_dict
        with open('predict.json', 'w+') as fr:
            json.dump(total_dict, fr)

            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)


if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
