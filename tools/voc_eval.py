from argparse import ArgumentParser

import mmcv

from mmdet import datasets
from mmdet.core import eval_map
from mmdet.datasets import build_dataset

def voc_eval(result_file, dataset, iou_thr=0.5, nproc=4):
    det_results = mmcv.load(result_file)
    #det_results = [item[0] for item in det_results]   #有seg输出时，新增该行代码
    annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]
    if hasattr(dataset, 'year') and dataset.year == 2007:
        dataset_name = 'voc07'
    else:
        dataset_name = dataset.CLASSES
    eval_map(
        det_results,
        annotations,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        logger='print',
        nproc=nproc)


def main():
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    parser.add_argument(
        '--nproc',
        type=int,
        default=4,
        help='Processes to be used for computing mAP')
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    #test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
    test_dataset = build_dataset(cfg.data.test)
    voc_eval(args.result, test_dataset, args.iou_thr, args.nproc)


if __name__ == '__main__':
    main()