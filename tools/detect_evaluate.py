import os
from collections import Counter
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from terminaltables import AsciiTable
#from cvio import cvio
try:
    from cvio import cvio
except:
    from .cvio import cvio


def bbox_overlaps(bboxes1, bboxes2, mode='standard'):
    assert mode in ('standard', 'minimum')
    if isinstance(bboxes1, list):
        bboxes1 = np.array(bboxes1)
    if isinstance(bboxes2, list):
        bboxes2 = np.array(bboxes2)
    iou = np.zeros((len(bboxes2), len(bboxes1)))
    for i in range(len(bboxes2)):
        bboxes = bboxes2[i][None, :]
        bboxes = np.repeat(bboxes, len(bboxes1), axis=0)
        x1 = np.concatenate((bboxes[:, 0:1], bboxes1[:, 0:1]), axis=1).max(1)
        y1 = np.concatenate((bboxes[:, 1:2], bboxes1[:, 1:2]), axis=1).max(1)
        x2 = np.concatenate((bboxes[:, 2:3], bboxes1[:, 2:3]), axis=1).min(1)
        y2 = np.concatenate((bboxes[:, 3:4], bboxes1[:, 3:4]), axis=1).min(1)
        dw = x2 - x1
        dw[dw < 0] = 0
        dh = y2 - y1
        dh[dh < 0] = 0
        inter = dw * dh
        areas1 = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        areas2 = (bboxes1[:, 2] - bboxes1[:, 0]) * \
            (bboxes1[:, 3] - bboxes1[:, 1])
        if mode == 'standard':
            union = areas1 + areas2 - inter
        else:
            union = np.concatenate(
                (areas1[:, None], areas2[:, None]), 1).min(1)
        iui = inter / union
        iou[i] = iui
    return iou.T

def mask_overlaps(shapes1, shapes2, mode='standard'):
    assert mode in ('standard', 'minimum')
    ious = []
    for shape1 in shapes1:
        points = get_points(shape1)
        mask1 = Polygon(points).buffer(0)
        ious.append([])
        for shape2 in shapes2:
            points = get_points(shape2)
            mask2 = Polygon(points).buffer(0)
            inter = mask1.intersection(mask2).area
            if mode == 'standard':
                union = mask1.union(mask2).area
            else:
                union = min(mask1.area, mask2.area)
            iou = inter / union
            ious[-1].append(iou)
    return np.array(ious)

def get_points(shape):
    points = shape['points']
    if shape['shape_type'] == 'rectangle':
        assert len(points) == 2
        (x1, y1), (x2, y2) = points
        points = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
    return points

def mask2bbox(mask):
    points = mask['points']
    points = np.array(points)
    xmin = points[:, 0].min()
    xmax = points[:, 0].max()
    ymin = points[:, 1].min()
    ymax = points[:, 1].max()
    return [xmin, ymin, xmax, ymax]

def shapes2bboxes(shapes):
    bboxes = [mask2bbox(shape) for shape in shapes]
    return bboxes

def calculate_overlaps(gts, dets, mode='bbox', iou_mode='standard'):
    assert mode in ('bbox', 'mask')
    if mode == 'bbox':
        gts = shapes2bboxes(gts)
        dets = shapes2bboxes(dets)
        return bbox_overlaps(gts, dets, iou_mode)
    return mask_overlaps(gts, dets, iou_mode)

def count_labels(shapes):
    return dict(Counter([shape['label'] for shape in shapes]))

def load_dataset(gtsrc, detsrc):
    gtlist = cvio.load_ext_list(gtsrc, ext_type='.json')
    dataset = []
    for gt in gtlist:
        det = os.path.join(detsrc, os.path.basename(gt))
        gt = cvio.load_ann(gt)
        if not os.path.exists(det):
            print('Not found: ', det)
            det = None
        else:
            det = cvio.load_ann(det)
        dataset.append([gt, det])
    return dataset

def load_classes(classes):
    if classes in ('', None):
        return
    if isinstance(classes, str):
        assert os.path.exists(classes), classes
        with open(classes) as fp:
            classes = [c.strip().replace('\n', '') for c in fp.readlines()]
        return classes
    return classes

def save_to_excel(results_per_sku, results_per_img, save_dir):
    if save_dir in ('', None):
        return
    if not os.path.exists(os.path.dirname(save_dir)):
        print('Directory not exists!', os.path.dirname(save_dir))
        return
    with pd.ExcelWriter(save_dir) as writer:
        results_per_img = pd.DataFrame(results_per_img)
        results_per_sku = pd.DataFrame(results_per_sku)
        results_per_sku.to_excel(writer, sheet_name='SKU', index=False)
        results_per_img.to_excel(writer, sheet_name='IMG', index=False)
    print('Evaluation results saved at', save_dir)

def evaluate_api(gtsrc, detsrc, save_dir=None, classes=None, iou_thr=0.5, mode='bbox', iou_mode='standard'):
    dataset = load_dataset(gtsrc, detsrc)
    classes = load_classes(classes)
    print('Dataset loaded, and found %d pairs for testing' % len(dataset))

    num_gt_det = dict()

    results = {}
    for i, (gts, dets) in enumerate(dataset, 1):
        img = gts['imagePath']
        if mode == 'mask':
            print('[%d/%d] %s' % (i, len(dataset), img))
        gt_shapes = gts['shapes']
        det_shapes = []
        if dets is not None:
            det_shapes = dets['shapes']
        
        num_gt = count_labels(gt_shapes)
        num_det = count_labels(det_shapes)

        ious = None
        if len(gt_shapes) and len(det_shapes):
            ious = calculate_overlaps(gt_shapes, det_shapes, mode=mode, iou_mode=iou_mode)

        num_true = {label: 0 for label in num_gt}
        if ious is not None:
            for j, iou in enumerate(ious):
                gt_label = gt_shapes[j]['label']
                idx = np.argmax(iou)
                iou = iou[idx]
                det_label = det_shapes[idx]['label']
                if iou >= iou_thr and det_label == gt_label:
                    num_true[gt_label] += 1
                # 新增：gt框检出率(不考虑分类)
                if gt_label not in num_gt_det:
                    num_gt_det[gt_label] = 0
                if iou >= iou_thr:
                    num_gt_det[gt_label] += 1

        result, _, _ = calculate_recall_precision(num_gt, num_det, num_true, classes)
        results[img] = result

    for k, v in num_gt_det.items():
        print(k,': ', v)

    num_gts, num_dets, num_trues = {}, {}, {}
    results_per_img = dict(images=[], gts=[], dets=[], true=[], recall=[], precision=[])
    for img, result in results.items():
        _gts = 0
        _dets = 0
        _trues = 0
        for label, count in result.items():
            _gts += count['gts']
            _dets += count['dets']
            _trues += count['true']
            if label not in num_gts:
                num_gts[label] = count['gts']
                num_dets[label] = count['dets']
                num_trues[label] = count['true']
            else:
                num_gts[label] += count['gts']
                num_dets[label] += count['dets']
                num_trues[label] += count['true']
        results_per_img['images'].append(img)
        results_per_img['gts'].append(_gts)
        results_per_img['dets'].append(_dets)
        results_per_img['true'].append(_trues)
        _recall = min(1, _trues / _gts if _gts != 0 else np.nan)
        _precision = min(1, _trues / _dets if _dets != 0 else np.nan)
        results_per_img['recall'].append(_recall)
        results_per_img['precision'].append(_precision)

    _, results, table = calculate_recall_precision(num_gts, num_dets, num_trues, classes)
    print_summary(table)

    # 新增
    results['true(不考虑类别)'] = list()
    results['recall(不考虑类别)'] = list()
    for i, label in enumerate(results['classes']):
        if label in num_gt_det:
            recall_tmp = min(num_gt_det[label] / results['gts'][i] if results['gts'][i] > 0 else np.nan, 1)
            results['true(不考虑类别)'].append(num_gt_det[label])
            results['recall(不考虑类别)'].append(recall_tmp)
        else:
            results['true(不考虑类别)'].append(None)
            results['recall(不考虑类别)'].append(None)

    save_to_excel(results, results_per_img, save_dir)

def print_summary(table):
    format_float_table = []
    for row in table:
        for i in range(len(row)):
            col = row[i]
            if isinstance(col, float):
                col = np.round(col, 4)
                row[i] = col
        format_float_table.append(row)
    table = AsciiTable(format_float_table)
    table.inner_footing_row_border = True
    print(table.table)

def calculate_recall_precision(num_gts, num_dets, num_trues, classes=None):
    if classes is not None:
        for label in classes:
            if label not in num_gts:
                num_gts[label] = 0
                num_trues[label] = 0
        for label in list(num_gts):
            if label not in classes:
                del num_gts[label]
                del num_trues[label]
    else:
        for label in num_dets:
            if label not in num_gts:
                num_gts[label] = 0
                num_trues[label] = 0

    results_sku = {}
    results_df = dict(classes=[], gts=[], dets=[], true=[], recall=[], precision=[])
    headers = list(results_df)
    table = [headers]
    _num_gts, _num_dets, _num_trues = 0, 0, 0
    for label, gts in num_gts.items():
        # if gts == 0:
        #     print((label, gts))
        dets = 0
        if label in num_dets:
            dets = num_dets[label]
        true = min(num_trues[label], dets, gts)
        _num_gts += gts
        _num_dets += dets
        _num_trues += true
        recall = min(true / gts if gts >0 else np.nan, 1)
        precision = min(true / dets if dets >0 else np.nan, 1)
        results_sku[label] = dict(gts=gts, dets=dets, true=true, recall=recall, precision=precision)
        results_df['classes'].append(label)
        results_df['gts'].append(gts)
        results_df['dets'].append(dets)
        results_df['true'].append(true)
        results_df['recall'].append(recall)
        results_df['precision'].append(precision)
        table.append([results_df[header][-1] for header in headers])
    recall = min(_num_trues / _num_gts if _num_gts > 0 else np.nan, 1)
    precision = min(_num_trues / _num_dets if _num_dets > 0 else np.nan, 1)
    if recall > 1 or precision > 1:
        print('EX: r %.4f p %.4f' % (recall, precision))
        print('See', os.path.abspath(__file__))
    table.append(['Summary', _num_gts, _num_dets, _num_trues, recall, precision])
    return results_sku, results_df, table

def unittest(*args, **kwargs):
    evaluate_api(*args, **kwargs)

if __name__ == '__main__':
    gtsrc = r'/data4/wh/bottle_data/val'
    pdsrc = r'/data4/wh/bottle_test/val_v3'
    save_dir = r'./val_v3.xlsx'
    unittest(gtsrc, pdsrc, save_dir, mode='bbox', iou_thr=0.5, iou_mode='standard')

    gtsrc = r'/data4/wh/bottle_test/bskl_test_3LGT'
    pdsrc = r'/data4/wh/bottle_test/bskl_test_v3_pd'
    save_dir = r'./bskl_test_v2.xlsx'
    unittest(gtsrc, pdsrc, save_dir, mode='bbox', iou_thr=0.5, iou_mode='standard')

    gtsrc = r'/data4/wh/bottle_test/hniu_test_3LGT'
    pdsrc = r'/data4/wh/bottle_test/hniu_test_v3_pd'
    save_dir = r'./hniu_test_v3.xlsx'
    unittest(gtsrc, pdsrc, save_dir, mode='bbox', iou_thr=0.5, iou_mode='standard')