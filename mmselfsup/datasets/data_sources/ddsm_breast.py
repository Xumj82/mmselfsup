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
                 img_shape=(1,1120,896),
                 classes=None,
                 ann_file=None,
                 test_mode=False,
                 color_type='color',
                 channel_order='rgb',
                 file_client_args=dict(backend='lmdb')):
        self.img_shape = img_shape
        super(DdsmBreast, self).__init__(
            data_prefix = data_prefix,
            ann_file = ann_file,
            test_mode = test_mode,
            color_type = color_type,
            channel_order = channel_order,
            file_client_args = file_client_args
        )

    def get_img(self, idx):
        """Get image by index.

        Args:
            idx (int): Index of data.

        Returns:
            Image: PIL Image format.
        """
        reslut = self.data_infos[idx]
        if self.file_client is None:
            self.file_client = mmcv.FileClient(db_path = reslut['img_prefix'],**self.file_client_args)

        cc_byte = self.file_client.get(reslut['img_info']['cc_view'])
        mlo_byte = self.file_client.get(reslut['img_info']['mlo_view'])
        
        if cc_byte is None or mlo_byte is None:
            print(reslut['img_id'])

        cc_img = np.frombuffer(cc_byte, np.uint16)
        mlo_img = np.frombuffer(mlo_byte, np.uint16)
        cc_img = cc_img.reshape(reslut['img_shape']).astype(np.float32)
        mlo_img = mlo_img.reshape(reslut['img_shape']).astype(np.float32)

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