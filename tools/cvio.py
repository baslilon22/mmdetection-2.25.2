import os
import json
import pickle
import cv2
import threading
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

__all__ = ('cvio', 'multi_thread_running')

class CVIO:
    
    def __init__(self):
        IMGEXT = ['.jpg', '.jpeg', '.png', '.tif']
        IMGEXT = IMGEXT + [ext.upper() for ext in IMGEXT]
        self.IMGEXT = IMGEXT
        self.ANNEXT = ['.json', '.xml', '.pkl', '.npy']
    
    def load_ann(self, ann_path, ann_type='json'):
        assert os.path.splitext(ann_path)[1] in self.ANNEXT
        ann_fun = dict(numpy=self.load_numpy, pkl=self.load_pickle,
                       xml=self.load_xml, voc=self.load_voc, json=self.load_json)
        return ann_fun[ann_type](ann_path)
    
    def load_json(self, src):
        try:
            with open(src, encoding='utf-8') as fp:
                return json.load(fp)
        except:
            with open(src) as fp:
                return json.load(fp)
        
    def load_xml(self, src):
        with open(src, encoding='utf-8') as fp:
            tree = ET.parse(fp)
            root = tree.getroot()
        return tree, root
    
    def load_pickle(self, src):
        with open(src, 'rb') as fp:
            return pickle.load(fp)
    
    def load_numpy(self, src):
        return np.load(src)
    
    def load_voc(self, src):
        tree, root = self.load_xml(src)

        filename = root.find('filename').text
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        bboxes = []
        labels = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)]
            bboxes.append(bbox)
            labels.append(label)

        xmlInfo = {"filename": filename, "width": w, "height": h, "bboxes": bboxes,
                   "labels": labels, "tree": tree, "root": root}
        return xmlInfo
    
    
    def load_img(self, src, mode='cv2'):
        assert os.path.splitext(src)[1] in self.IMGEXT
        mode = mode.lower()
        assert mode in ('cv2', 'cv', 'pil', 'pillow')
        if mode in ('cv', 'cv2'):
            return cv2.imread(src)
        elif mode in ('pillow', 'pil'):
            return Image.open(src)

    def write_img(self, img, dst, mode='cv2'):
        assert os.path.splitext(dst)[1] in self.IMGEXT
        mode = mode.lower()
        assert mode in ('cv2', 'cv', 'pil', 'pillow')
        if mode in ('cv', 'cv2'):
            return cv2.imwrite(dst, img)
        elif mode in ('pillow', 'pil'):
            return Image.fromarray(img).save(dst)
        
    def load_image_list(self, src, recursive=False, silent=True):
        image_list = []
        i = 0
        for root, _, files in os.walk(src):
            for file in files:
                ext = os.path.splitext(file)[1]
                if not ext in self.IMGEXT:
                    continue
                image_list.append(os.path.join(root, file))
                i += 1
                if not silent and i % 10000 == 0 and i >= 10000:
                    print('Load images %d.' % i)
            if not recursive:
                return image_list
        return image_list
    
    def load_ext_list(self, src, ext_type='.json', recursive=False, silent=True):
        ext_list = []
        i = 0
        for root, _, files in os.walk(src):
            for file in files:
                ext = os.path.splitext(file)[1][1:]
                if not ext in ext_type:
                    continue
                ext_list.append(os.path.join(root, file))
                i += 1
                if not silent and i % 10000 == 0 and i >= 10000:
                    print('Load files %d.' % i)
            if not recursive:
                return ext_list
        return ext_list     
    
    def load_img_ann_list(self, src, ann_type='.json', recursive=False, silent=True):
        img_ann_list = []
        ann_ext = '.%s' % ann_type.replace('.', '')
        i = 0
        for root, _, files in os.walk(src):
            for file in files:
                imgext = os.path.splitext(file)[1]
                if not imgext in self.IMGEXT:
                    continue
                img_path = os.path.join(root, file)
                ann_path = os.path.splitext(img_path)[0] + ann_ext
                if not os.path.exists(ann_path):
                    continue
                img_ann_list.append([img_path, ann_path])
                i += 1
                if not silent and i % 10000 == 0 and i >= 10000:
                    print('Load images/annotations %d.' % i)
            if not recursive:
                return img_ann_list
        return img_ann_list         

    def __repr__(self):
        infostr = 'CVIO(IMGEXT(%s), ANNEXT(%s), ' % (self.IMGEXT, self.ANNEXT)
        funs = [f for f in dir(self) if not '__' in f]
        infostr += '%s)' % (funs)
        return infostr

    def write_ann(self, ann_info, destpath, ann_type='json'):
        assert ann_type in ('json', 'pickle')
        if ann_type in '.json':
            with open(destpath, 'w') as fp:
                json.dump(ann_info, fp=fp)
        elif ann_type in 'pickle' or ann_type in '.pkl':
            with open(destpath, 'wb') as fp:
                pickle.dump(ann_info, file=fp)
        else:
            raise 'Not implemented error!'

cvio = CVIO()

def multi_thread_running(targets_args):
    threads = []
    for target, arg in targets_args:
        t = threading.Thread(target=target, args=arg)
        threads.append(t)
    for i, t in enumerate(threads, 1):
        t.setDaemon(True)
        t.start()

    for t in threads:
        t.join()
