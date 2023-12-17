import os
import cv2
import torch
import json
import numpy as np
import mmcv
from tqdm import tqdm

from mmdet.apis import inference_detector, init_detector  #, show_result_pyplot
from tools.detect_evaluate import evaluate_api

def show_result_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 10)):
    """Visualize the detection results on the image.
    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    return img


def bitmap_to_polygon(bitmap):
    """Convert masks from the form of bitmaps to polygons.

    Args:
        bitmap (ndarray): masks in bitmap representation.

    Return:
        list[ndarray]: the converted mask in polygon representation.
        bool: whether the mask has holes.
    """
    bitmap = np.ascontiguousarray(bitmap).astype(np.uint8)
    # cv2.RETR_CCOMP: retrieves all of the contours and organizes them
    #   into a two-level hierarchy. At the top level, there are external
    #   boundaries of the components. At the second level, there are
    #   boundaries of the holes. If there is another contour inside a hole
    #   of a connected component, it is still put at the top level.
    # cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
    outs = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = outs[-2]
    hierarchy = outs[-1]
    if hierarchy is None:
        return [], False
    # hierarchy[i]: 4 elements, for the indexes of next, previous,
    # parent, or nested contours. If there is no corresponding contour,
    # it will be -1.
    with_hole = (hierarchy.reshape(-1, 4)[:, 3] >= 0).any()
    contours = [c.reshape(-1, 2) for c in contours]
    return contours, with_hole

def segm_to_polygon(segm):
    contours, with_hole = bitmap_to_polygon(segm)
    if not len(contours):
        return []
    if with_hole:
        areas = []
        for contour in contours:
            x1 = contour[:, 0].min()
            y1 = contour[:, 1].min()
            x2 = contour[:, 0].max()
            y2 = contour[:, 1].max()
            area = (x2 - x1) * (y2 - y1)
            areas.append(area)
        # largest mask
        idx = np.argmax(areas)
        contour = contours[idx]
    else:
        contour = contours[0]
    contour = cv2.convexHull(contour)[:,0,:].astype(float).tolist()
    return contour

def contour2minAreaBbox(contour):
    contour = np.array(contour, dtype=np.float32)
    rotatedRect = cv2.minAreaRect(contour)
    bbox = cv2.boxPoints(rotatedRect).tolist()
    return bbox

def compute_polygon_area(points):
    point_num = len(points)
    if(point_num < 3):
        return 0.0
    s = points[0][1] * (points[point_num-1][0] - points[1][0])
    #for i in range(point_num): # (int i = 1 i < point_num ++i):
    for i in range(1, point_num): # 有小伙伴发现一个bug，这里做了修改，但是没有测试，需要使用的亲请测试下，以免结果不正确。
        s += points[i][1] * (points[i-1][0] - points[(i+1)%point_num][0])
    return abs(s/2.0)

def between_class_nms(boxes, labels, iou_threshold=0.7):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    areas = (x2-x1) * (y2-y1)
    keep = []

    # 按置信度从大到小排序
    index = np.argsort(scores)[::-1]

    while index.size > 0:
        # 置信度最高的框
        i = index[0]
        keep.append(i)

        if index.size == 1: # 如果只剩一个框，直接返回
            break

        # 计算当前bbox与剩余的bbox之间的IoU
        # 计算IoU需要两个bbox中最大左上角的坐标点和最小右下角的坐标点
        # 即重合区域的左上角坐标点和右下角坐标点
        inter_x1 = np.maximum(x1[i], x1[index[1:]])
        inter_y1 = np.maximum(y1[i], y1[index[1:]])
        inter_x2 = np.minimum(x2[i], x2[index[1:]])
        inter_y2 = np.minimum(y2[i], y2[index[1:]])
        # 计算交集的面积
        inter_area = np.maximum(inter_x2-inter_x1, 0) * np.maximum(inter_y2-inter_y1, 0)
        # 计算当前框与其余框的iou
        iou = inter_area / (areas[index[1:]] + areas[i] - inter_area)
        # 删除IoU大于指定阈值的bbox(重合度高), 保留小于指定阈值的bbox
        ids = np.where(iou <= iou_threshold)[0]
        index = index[ids+1]

    return boxes[keep], labels[keep]

def format_result(result, classes, score_thr=0.3, mode='bbox', nms_classes=None):
        assert mode in ('bbox', 'segm')
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        labels = []
        for i, bbox in enumerate(bbox_result):
            if not len(bbox):
                continue
            labels.extend([classes[i]] * len(bbox))
        labels = np.array(labels, dtype=str)
        #pdb.set_trace()
        #labels = [[classes[i]] * len(bbox)  for i, bbox in enumerate(bbox_result) if len(bbox)]
        #labels = np.array(labels)[0]
        bboxes = np.vstack(bbox_result).astype(np.float)
        # print(len(bboxes), bboxes)

        segms = []
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)

        if score_thr > 0:
            assert bboxes is not None and bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            if segm_result is not None and len(segms):
                segms = segms[inds, ...]

        # draw segmentation masks
        shapes = []
        if len(segms):
            for i, segm in enumerate(segms):
                contour = segm_to_polygon(segm)
                if mode == 'bbox':
                    contour = contour2minAreaBbox(contour)
                area1 = compute_polygon_area(contour)
                x1, y1, x2, y2, score = bboxes[i]
                area2 = (y2 - y1) * (x2 - x1)
                if not len(contour) or len(contour) < 3 or area1 <= 0.2 * area2:
                    contour = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    print('Found illegal contour', i, labels[i]) # contour
                    continue                                     # 过滤该contour
                shape = dict(label=labels[i], points=contour, shape_type='polygon')
                shapes.append(shape)
        elif nms_classes is not None:
            # 对bbox结果做类间NMS
            nms_inds, extra_inds = [], []
            nms_classes = set(nms_classes)
            for i, label in enumerate(labels):
                if label in nms_classes:
                    nms_inds.append(i)
                else:
                    extra_inds.append(i)
            nms_bbox, nms_labels = bboxes[nms_inds], labels[nms_inds]
            extra_bbox, extra_labels = bboxes[extra_inds], labels[extra_inds]
            nms_bbox, nms_labels = between_class_nms(boxes=nms_bbox, labels=nms_labels)
            bboxes = np.concatenate((nms_bbox, extra_bbox))
            labels = np.concatenate((nms_labels, extra_labels))

            for i, (x1, y1, x2, y2, score) in enumerate(bboxes):
                contour = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                shape = dict(label=labels[i], points=contour, shape_type='polygon')
                shapes.append(shape)
        else:
            for i, (x1, y1, x2, y2, score) in enumerate(bboxes):
                contour = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                shape = dict(label=labels[i], points=contour, shape_type='polygon')
                shapes.append(shape)
        return shapes


def get_images_names(root):
    """支持图片的格式"""
    pic_form = ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']
    filelist = [f for f in os.listdir(root) if os.path.splitext(f)[-1] in pic_form]
    return filelist

def main():
    score_thr = 0.2
    config_file = r'./configs/retinanet/Shrank_retinanet_r101_fpn_mstrain_640-800_3x_coco.py'
    checkpoint_file = r'./work_dirs/Shrank_retinanet_r101/best_bbox_mAP_50_epoch_31_map0799.pth'
    img_dir = r'/data4/wh/JML_Shrank/test'
    out_dir = r'/data4/wh/JML_Shrank/test_pd'
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    model = init_detector(config_file, checkpoint=checkpoint_file, device='cuda:7')
    #nms_classes = ('rh_ys_daiding_102', 'rh_yg_daiding_103', 'rh_others_daiding_104', 'rh_ygcm_daiding_105')
    #nms_classes = None
    nms_classes = model.CLASSES
    
    imgs_list = get_images_names(img_dir)
    for filename in tqdm(imgs_list):
        img_path = os.path.join(img_dir, filename)
        result = inference_detector(model, img_path)
        #img = show_result_pyplot(model, img_path, result, score_thr=score_thr)
        #cv2.imwrite(os.path.join(out_dir, filename), img)
        # save json:
        json_result = format_result(result, model.CLASSES, score_thr, mode='bbox', nms_classes=nms_classes)
        json_result = dict(shapes=json_result, imagePath=filename, imageData=None)
        json_path = os.path.join(out_dir, os.path.splitext(filename)[0] + '.json')
        with open(json_path, 'w', encoding='utf8') as json_file:
            json.dump(json_result, json_file, ensure_ascii=False, indent=2)
    
    # 评估性能
    save_xlsx = out_dir + '.xlsx'
    evaluate_api(img_dir, out_dir, save_xlsx)

if __name__ == '__main__':
    main()
