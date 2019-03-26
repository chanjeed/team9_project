import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import glob


class AnnotationTransform(object):
    def __init__(self, class_to_index=None, keep_difficult=False):
        self.class_to_index = class_to_index
        self.keep_difficult = keep_difficult

    def __call__(self, xml_file, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        targets = list()

        element = ET.parse(xml_file).getroot()
        # 画像幅
        element_width = int(element.find('size').find('width').text)
        # 画像高さ
        element_height = int(element.find('size').find('height').text)
        # チャネル数
        element_depth = int(element.find('size').find('depth').text)
        for obj in element.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            xmin = (int(bbox.find('xmin').text) - 1) / width
            ymin = (int(bbox.find('ymin').text) - 1) / height
            xmax = (int(bbox.find('xmax').text) - 1) / width
            ymax = (int(bbox.find('ymax').text) - 1) / height

            label_index = self.class_to_index[name]

            targets.append([xmin, ymin, xmax, ymax, label_index])

        return targets  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class DataSetObjectDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 transform=None, target_transform=AnnotationTransform()
                 ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = os.path.join(root, 'annotation')
        self._imgpath = os.path.join(root, 'color-images')
        print(self._imgpath)
        image_path_sets = glob.glob(os.path.join(self._imgpath, '*.png'))
        self.ids = list()
        for image_path in image_path_sets:
            image_id, _ = os.path.splitext(os.path.basename(image_path))
            self.ids.append(image_id)
        print(len(self.ids))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = os.path.join(self._annopath, img_id + '.xml')
        img = cv2.imread(os.path.join(self._imgpath, img_id + '.png'))
        height, width, channels = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            # cv2のchannelsはbgrなのでrgbの順番に変更
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # 画像の次元の順番をHWCからCHWに変更
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(os.path.join(self._imgpath, img_id + '.png'), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = os.path.join(self._annopath, img_id, '.xml')
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
