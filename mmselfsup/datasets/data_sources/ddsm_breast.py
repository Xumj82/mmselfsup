import os
import os.path as osp

import mmcv
import pandas as pd
import numpy as np

from ..builder import DATASOURCES
from .base import BaseDataSource


@DATASOURCES.register_module()
class DdsmBreast(BaseDataSource):


    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')

    def __init__(self,
                 data_prefix,
                 classes=None,
                 ann_file=None,
                 test_mode=False,
                 color_type='color',
                 channel_order='rgb',
                 file_client_args=dict(backend='disk')):
        super.__init__(
            
        )
        self.data_prefix = data_prefix
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args
        self.file_client = None
        self.CLASSES = self.get_classes(classes)
        self.data_infos = self.load_annotations()

    def get_img(self, idx):
        """Get image by index.

        Args:
            idx (int): Index of data.

        Returns:
            Image: PIL Image format.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(db_path = self.data_infos['img_prefix'],**self.file_client_args)

        cc_byte = self.file_client.get(self.data_infos['img_info']['cc_view'])
        mlo_byte = self.file_client.get(self.data_infos['img_info']['mlo_view'])
        
        if cc_byte is None or mlo_byte is None:
            print(self.data_infos['img_id'])

        cc_img = np.frombuffer(cc_byte, np.uint16)
        # mlo_img = np.frombuffer(mlo_byte, np.uint16)
        cc_img = cc_img.reshape(self.data_infos['img_shape'])
        mlo_img = mlo_img.reshape(self.data_infos['img_shape'])

        return (cc_img, mlo_img)

    def load_annotations(self):
        assert self.ann_file ,'ann file cannot be none'
        self.samples = pd.read_csv(self.ann_file)
        data_infos = []
        for idx, row in self.samples.iterrows():
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'cc_view': row['cc_view'], 'mlo_view': row['mlo_view']}
            info['gt_label'] = np.array(1 if row['pathology']=='MALIGNANT' else 0, dtype=np.int64)
            info['img_shape'] = self.img_shape
            info['idx'] = int(idx)
            data_infos.append(info)
        return data_infos