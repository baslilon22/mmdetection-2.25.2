import os
import json
import shutil
import random
import cv2 as cv
opj = os.path.join



def clamp(x, MIN, MAX):
    if x < MIN:
        return MIN
    elif x > MAX:
        return MAX
    else:
        return x

def convert_to_coco(ann_path, out_file):
    # 生成coco标注，强制转换或过滤非法点; 实例分割->目标分割
    images = []
    annotations = []
    
    #classes = ('daiding_101', 'daiding_102', 'daiding_103')
    #classes = ('rh_box_daiding_101', 'rh_ys_daiding_102', 'rh_yg_daiding_103', 'rh_others_daiding_104', 'rh_ygcm_daiding_105')
    classes = ('fake', 'yl_jml_bhc_500P', 'yl_jml_bhc_600P', 'yl_jml_btxl_500P', 'yl_jml_cc_bdnmbhc_500P', 'yl_jml_cc_bzqtlc_500P', 'yl_jml_cc_kxjymlh_500P', 'yl_jml_dpbhc_750P', 'yl_jml_dpbtxl_750P', 'yl_jml_dpfmyz_750P', 'yl_jml_dplc_750P', 'yl_jml_dplzs_750P', 'yl_jml_dpmlmc_750P', 'yl_jml_dpqmlc_750P', 'yl_jml_fmyz_500P', 'yl_jml_jjnm_500P', 'yl_jml_jk_kqs_570P', 'yl_jml_kqs_550P', 'yl_jml_lbk_550P', 'yl_jml_lc_500P', 'yl_jml_lc_600P', 'yl_jml_lpbhc_1000P', 'yl_jml_lpbtxl_1000P', 'yl_jml_lpfmyz_1000P', 'yl_jml_lplc_1000P', 'yl_jml_lpmlmc_1000P', 'yl_jml_lpqmlc_1000P', 'yl_jml_mdxz_hls_500P', 'yl_jml_mdxz_mts500P', 'yl_jml_mdxz_mts_500P', 'yl_jml_mdxz_nms500P', 'yl_jml_mdxz_nms_500P', 'yl_jml_mdxz_qms_500P', 'yl_jml_mdxz_qpgs500P', 'yl_jml_mdxz_qpgs_500P', 'yl_jml_mdxz_xgs_500P', 'yl_jml_mlmc_500P', 'yl_jml_mlmc_600P', 'yl_jml_others', 'yl_jml_qmlc_500P', 'yl_jml_rsjs_450P', 'yl_jml_sdsbtw_450P', 'yl_jml_sdsnmw_450P', 'yl_jml_sdsyw_450P', 'yl_jml_smt_500P', 'yl_jml_wsss_450P')
    #classes = ('other','sp_ht_5dbc_1.9L','sp_ht_bmc_1.28L','sp_ht_bmc_1.9L','sp_ht_gdlj_1.28L','sp_ht_gdlj_1.9L','sp_ht_hxjy_1.28L','sp_ht_hxjy_1.75L','sp_ht_hxjy_500ml','sp_ht_hxrjy_500ml','sp_ht_jbhy_1.07kg','sp_ht_jbhy_265g','sp_ht_jbhy_315g','sp_ht_jbhy_530g','sp_ht_jbhy_635g','sp_ht_jbhy_715g','sp_ht_jbsc_1.28L','sp_ht_jbsc_1.6L','sp_ht_jbsc_1.9L','sp_ht_jbsc_500ml','sp_ht_jzlj_1.9L','sp_ht_lbjy_500ml','sp_ht_sdhy_1kg','sp_ht_sdhy_260g','sp_ht_sdhy_520g','sp_ht_sdhy_590g','sp_ht_sdhy_6kg','sp_ht_sdhy_700g','sp_ht_wjxjy_1.28L','sp_ht_wjxjy_1.9L','sp_ht_ybsc_500ml')
    #classes = ('mnc_dj_gq', 'mnc_dj_hmz', 'mnc_dj_htj', 'mnc_dj_lsz', 'mnc_dj_mpt', 'mnc_dj_rxb', 'mnc_dj_tj', 'mnc_dj_tsz', 'mnc_dj_wt', 'mnc_dj_yzta', 'mnc_dj_yztg', 'mnc_region_front', 'mnc_region_mid', 'mnc_region_other', 'mnc_sp_sp', 'mnc_sp_xl')

    categories = []
    id = 1
    for item in classes:
        categories.append(dict(id=id, name=item))
        id += 1

    imgs_id = 1
    anno_id = 1

    for filename in os.listdir(ann_path):
        if filename.endswith(".json"):
            with open(opj(ann_path, filename), 'r') as f:
                data_infos = json.load(f)
                img_path = data_infos['imagePath']
                # 读取图片尺寸， 图片和json在同一路径
                img_cv = cv.imread(opj(ann_path, img_path))
                w = img_cv.shape[1]
                h = img_cv.shape[0]
                images.append(dict(
                                id=imgs_id,
                                file_name=img_path,
                                height=h,
                                width=w)
                            )
                
                if len(data_infos['shapes']) == 0:  #无标注
                    print('无标注json: ', filename)
                    """data_anno = dict(
                                    id=anno_id,
                                    image_id=imgs_id,
                                    category_id=1,
                                    bbox=[0,0,0,0],
                                    area=0,
                                    segmentation=[[0,0]],
                                    iscrowd=0)
                    annotations.append(data_anno)
                    anno_id += 1"""
                else:
                    for item in data_infos['shapes']:
                        label = item['label']
                        points = item['points']
                        px, py, poly = [], [], []
                        for p in points:
                            # 截断越界的坐标值
                            p[0] = clamp(p[0], 0, w - 1.0)
                            p[1] = clamp(p[1], 0, h - 1.0)
                            px.append(p[0])
                            py.append(p[1])
                            poly.append(p[0])
                            poly.append(p[1])
                        x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))

                        if len(points) < 2: # 标注点小于2个，过滤
                            continue
                        elif len(points) == 2: # 标注点仅有两个，转为box
                            seg = [x_min,y_min, x_max,y_min, x_max,y_max, x_min,y_max]
                            seg = [seg]
                        else:
                            seg = [poly]
                        # label: 实例分割->目标分割
                        # pos = label.rfind('-')
                        # if pos != -1:
                        #     label = label[:pos]
                        data_anno = dict(
                                        id=anno_id,
                                        image_id=imgs_id,
                                        category_id= classes.index(label) + 1,
                                        bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                                        area=(x_max - x_min) * (y_max - y_min),
                                        segmentation=seg,
                                        iscrowd=0)
                        annotations.append(data_anno)
                        anno_id += 1
            imgs_id += 1
    coco_format_json = dict(
                            images=images,
                            annotations=annotations,
                            categories=categories)
    with open(out_file, 'w') as f:
        json.dump(coco_format_json, f, indent=2)
    print('Convert {} .json file'.format(imgs_id-1))


def show_label(ann_path):
    # 统计label的种类
    labels = set()
    dirs = os.listdir(ann_path)
    for file_name in dirs:
        if file_name.endswith(".json"):
            with open(os.path.join(ann_path, file_name), 'r') as f:
                data = json.load(f)
                for item in data["shapes"]:
                    kind = item['label']
                    #index = kind.rfind('-')
                    #if index != -1:
                    #    kind = kind[:index]
                    labels.add(kind)
    print(len(labels))
    for label in labels:
        print("'"+label+"'", end=", ")

def show_labels_and_num(ann_path):
    # 统计label的种类和数量
    labels = dict()
    dirs = os.listdir(ann_path)
    for file_name in dirs:
        if file_name.endswith(".json"):
            with open(os.path.join(ann_path, file_name), 'r') as f:
                data = json.load(f)
                for item in data["shapes"]:
                    kind = item['label']
                    #index = kind.rfind('-')
                    #if index != -1:
                    #    kind = kind[:index]
                    if kind in labels:
                        labels[kind] += 1
                    else:
                        labels[kind] = 1
    for k,v in labels.items():
        print(k, ": ", v)

def convert_label(ann_path, out_path):
    # 转换标注的类别，并复制图片和json到out_path（无过滤标注）
    dirs = os.listdir(ann_path)
    for file_name in dirs:
        if file_name.endswith(".json"):
            with open(os.path.join(ann_path, file_name), 'r') as f:
                data = json.load(f)
                img_path = data['imagePath']
                for i in range(len(data["shapes"])):
                    label = data["shapes"][i]['label']
                    #index = label.rfind('-')
                    #if index != -1:
                    #    data["shapes"][i]['label'] = label[:index]
                    tmp = label.split('_')
                    if label == 'skbb_yg_etsbddystht_T11':
                        data["shapes"][i]['label'] = 'rh_ys_daiding_102'
                    elif label == 'sk_dzzh_mbhysm_jhx6' or label == 'sk_jy_yxjyx2':
                        data["shapes"][i]['label'] = 'rh_yg_daiding_103'
                    elif label == 'sk_yg_cm' or label == 'sk_yg_kkc':
                        data["shapes"][i]['label'] = 'rh_ygcm_daiding_105'
                    elif tmp[1] =='yg' or tmp[1] == 'ygys':
                        data["shapes"][i]['label'] = 'rh_yg_daiding_103'
                    elif tmp[1] == 'ys':
                        data["shapes"][i]['label'] = 'rh_ys_daiding_102'
                    else:
                        data["shapes"][i]['label'] = 'rh_others_daiding_104'
                #保存文件
                #shutil.copy(opj(ann_path, img_path), out_path)
                with open(os.path.join(out_path, file_name), 'w') as f:
                    json.dump(data, f, indent=2)


def filter_dataset_by_label(in_path, out_path, label_List=['daiding_102', 'daiding_103']):
    # 筛选出包含指定label的图片
    num = 0
    for filename in os.listdir(in_path):
        if filename.endswith(".json"):
            with open(opj(in_path, filename), 'r') as f:
                data_infos = json.load(f)
                img_path = data_infos['imagePath']
                
                if len(data_infos['shapes']) == 0:  #无标注
                    print('无标注:', filename)
                    continue
                else:
                    for item in data_infos['shapes']:
                        label = item['label']
                        pos = label.rfind('-')
                        if pos != -1:
                            label = label[:pos]
                        if label in label_List:
                            # 包含指定标注，复制图片和标注
                            shutil.copy(opj(in_path, img_path), out_path)
                            shutil.copy(opj(in_path, filename), out_path)
                            num += 1
                            break
    print('Collect {} pictures:'.format(num))


def get_images_names(root):
    """支持图片的格式"""
    pic_form = ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']
    filelist = [f for f in os.listdir(root) if os.path.splitext(f)[-1] in pic_form]
    return filelist

def split_dataset(src_dir, val_dir, val_ratio=0.1):
    # 按比例随机划分训练/验证集，即将src目录下的img和.json移动到out目录下
    seed = 2023
    random.seed(seed)

    src_list = get_images_names(src_dir)
    val_size = int(len(src_list) * val_ratio)
    print('训练集: {}, 验证集: {}'.format(len(src_list)-val_size, val_size))

    random.shuffle(src_list)
    for i in range(val_size):
        curr = src_list[i]
        shutil.move(os.path.join(src_dir, curr), os.path.join(val_dir, curr))
        index = curr.rfind('.')
        shutil.move(os.path.join(src_dir, curr[:index] + '.json'), os.path.join(val_dir, curr[:index] + '.json'))

if __name__ == '__main__':
    """
    src = r'F:\DATA\bottle_data\daiding_cola'
    out = r'F:\DATA\bottle_data\daiding_cola_train'
    filter_dataset_by_label(in_path=src, out_path=out, label_List=['daiding_102', 'daiding_103'])
    """
    convert_to_coco(ann_path=r'/data4/wh/JML_Shrank/train_Shrank', out_file=r'/data4/wh/JML_Shrank/annotation/train_Shrank.json')
    convert_to_coco(ann_path=r'/data4/wh/JML_Shrank/train_All', out_file=r'/data4/wh/JML_Shrank/annotation/train_All.json')

    '''
    # 检查各标注其json名与imagePath是否一致, 以及图像与json文件名是否对应
    img_dir = './src'
    dirs = os.listdir(img_dir)
    for file_name in dirs:
        if file_name.endswith(".json"):
            with open(os.path.join(img_dir, file_name), 'r') as f:
                data = json.load(f)
                index = data["imagePath"].rfind('.')
                img_in_json = data["imagePath"][:index]
                json_name = file_name[:-5]
                if img_in_json != json_name:
                    print('json_name:{} img_in_json:{}'.format(json_name, img_in_json))
        elif file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
            index = file_name.rfind('.')
            json_file = file_name[:index] + '.json'
            json_file = os.path.join(img_dir, json_file)
            if not os.path.exists(json_file):
                print('Json not exists: ', json_file)
    '''
    

    # 各文件label转换为coco格式, 注意改classes
    # 食品业
    # convert_to_coco(ann_path='/data4/wh/JML_MP/train', out_file='/data4/wh/JML_MP/annotation/train.json') # 1931
    # convert_to_coco(ann_path='/data4/wh/JML_MP/val', out_file='/data4/wh/JML_MP/annotation/val.json')     # 200
    
    # 饮料业
    #convert_to_coco(ann_path='./train', out_file='./annotation/train.json') #DrinkData: bottle + 酒业 + bottle_test_ok:4703
    #convert_to_coco(ann_path='./val', out_file='./annotation/val.json')
    
    # 海天
    #convert_to_coco(ann_path=r'/data4/wh/haitian/train', out_file=r'/data4/wh/haitian/annotation/train.json') #2119+235=2354
    #convert_to_coco(ann_path=r'/data4/wh/haitian/val', out_file=r'/data4/wh/haitian/annotation/val.json') #235
    
    # 日化
    #convert_to_coco(ann_path=r'/data4/wh/rh_data/train', out_file=r'/data4/wh/rh_data/annotation/train.json') #2636
    #convert_to_coco(ann_path=r'/data4/wh/rh_data/val', out_file=r'/data4/wh/rh_data/annotation/val.json') #200
    
    # Monaco
    #convert_to_coco(ann_path=r'/data4/wh/Monaco/train_img_zb', out_file=r'/data4/wh/Monaco/annotation/train.json') #458
    #convert_to_coco(ann_path=r'/data4/wh/Monaco/test_img_zb', out_file=r'/data4/wh/Monaco/annotation/test.json') #40
    