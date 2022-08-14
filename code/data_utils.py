import os
import pandas as pd
import math
import torch
import torchvision
import random
from PIL import Image
import numpy as np


class Dataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]
    
    def __setitem__(self, key, value):
        self.data[key]=value

    def __len__(self):
        return len(self.data)


class DatasetReader:
    def __init__(self, dataset, tokenizer):
        print("preparing {0} dataset ...".format(dataset))
        dir_pth = {
            'csv': '../data/sourcedata/'
        }
        fn = {
            'csv': dataset + '.csv'
        }
        img_roi_dir = {
            'tw15': '../data/img_roi/tw15/',
            'tw17': '../data/img_roi/tw17/'
        }
        csv_fn = os.path.join(dir_pth['csv'], fn['csv'])
        
        data_path = '../data/boa_vocab'
        # Load classes
        self.classes = ['__background__']
        with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())

        # Load attributes
        self.attributes = ['__no_attribute__']
        with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
            for att in f.readlines():
                self.attributes.append(att.split(',')[0].lower().strip())

        self.data = Dataset(self.__read_data__(csv_fn, img_roi_dir[dataset.split('_')[0]], tokenizer))
        

    def __read_data__(self, csv_fn, img_roi_dir, tokenizer):
        csv_df = pd.read_csv(csv_fn)
        
        max_cls_len = 10
        max_roi_num = 36

        all_data = []
        for i in range(0, len(csv_df)):
            text = csv_df.loc[i, 'text']
            label = int(csv_df.loc[i, 'label'])
            aspect = csv_df.loc[i, 'aspect']
            
            img_roi_fn = csv_df.loc[i, 'img'].split('.')[0]
            img_roi_pth = img_roi_dir + img_roi_fn + '.npz'
            img_roi = np.load(img_roi_pth, allow_pickle=True)
            
            objects = img_roi['info'].item()['objects_id']
            attr = img_roi['info'].item()['attrs_id']
            objects_conf = img_roi['info'].item()['objects_conf']
            attr_conf = img_roi['info'].item()['attrs_conf']
            
            img_roi_features = img_roi['x']
            img_roi_features_padded = np.pad(img_roi_features, ((0, max_roi_num-len(img_roi_features)), (0, 0)), 'constant')
            
            attr_thresh = 0.1
            img_roi_cls_list = []
            for i in range(0, len(objects)):
                cls = self.classes[objects[i]+1]  
                if attr_conf[i] > attr_thresh:
                    cls = self.attributes[attr[i]+1] + " " + cls
                img_roi_cls_list.append(cls)
                
            tok_outputs = tokenizer(img_roi_cls_list, add_special_tokens=False, padding='max_length', max_length=max_cls_len)
            img_roi_cls_ids_list = tok_outputs['input_ids']
            img_roi_cls_ids_msk_list = tok_outputs['attention_mask']
            img_roi_cls_ids_list_padded = img_roi_cls_ids_list + [[0] * max_cls_len] * (max_roi_num - len(img_roi_cls_ids_list))
            img_roi_cls_ids_msk_list_padded = img_roi_cls_ids_msk_list + [[0] * max_cls_len] * (max_roi_num - len(img_roi_cls_ids_msk_list))
            img_roi_msk = [1] * len(img_roi_features) + [0] * (max_roi_num-len(img_roi_features))

            # inputs = tokenizer(text, aspect)
            inputs = tokenizer(text)
            
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            assert len(input_ids) == len(attention_mask), "length error"
            
            aspect_idx_list = []
            aspect_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(aspect))
            for start_idx in range(0, len(input_ids) - len(aspect_ids) + 1):
                flag_match = True
                for i in range(0, len(aspect_ids)):
                    cur_idx = i + start_idx
                    if input_ids[cur_idx] != aspect_ids[i]:
                        flag_match = False
                        break
                if flag_match:
                    aspect_idx_list += list(range(start_idx, cur_idx + 1))
                
            aspect_msk = [0] * len(input_ids)
            for idx in aspect_idx_list:
                aspect_msk[idx] = 1

            data = {
                'text': text,
                'aspect': aspect,
                
                'aspect_msk': aspect_msk,
                
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'img_roi_features_padded': img_roi_features_padded,
                'img_roi_cls': img_roi_cls_list,
                'img_roi_cls_ids_padded': img_roi_cls_ids_list_padded,
                'img_roi_cls_ids_msk_padded': img_roi_cls_ids_msk_list_padded,
                'img_roi_msk': img_roi_msk,
                'label': label
            }
            all_data.append(data)
        return all_data


class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='input_ids', shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        if self.shuffle:
            random.shuffle(data)
        
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i * batch_size: (i + 1) * batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text = []
        batch_aspect = []
        
        batch_aspect_msk = []
        
        batch_input_ids = []
        batch_attention_mask = []
        
        batch_img_roi_features_padded = []
        batch_img_roi_cls = []
        batch_img_roi_cls_ids_padded = []
        batch_img_roi_cls_ids_msk_padded = []
        batch_img_roi_msk = []
        
        batch_label = []

        max_len = max([len(t[self.sort_key]) for t in batch_data])
        max_num_roi = 36
        for item in batch_data:
            text, aspect, aspect_msk, input_ids, attention_mask, img_roi_features_padded, img_roi_cls, \
                img_roi_cls_ids_padded, img_roi_cls_ids_msk_padded, img_roi_msk, label = \
                item['text'], item['aspect'], item['aspect_msk'], item['input_ids'], item['attention_mask'], \
                item['img_roi_features_padded'], item['img_roi_cls'], item['img_roi_cls_ids_padded'], \
                item['img_roi_cls_ids_msk_padded'], item['img_roi_msk'], item['label']

            ids_padding = [0] * (max_len - len(input_ids))
            attention_padding = [0] * (max_len - len(input_ids))

            batch_text.append(text)
            batch_aspect.append(aspect)
            
            batch_aspect_msk.append(aspect_msk + ids_padding)
            
            batch_input_ids.append(input_ids + ids_padding)
            batch_attention_mask.append(attention_mask + attention_padding)
            
            batch_img_roi_features_padded.append(img_roi_features_padded)
            batch_img_roi_cls.append(img_roi_cls)
            batch_img_roi_cls_ids_padded.append(img_roi_cls_ids_padded)
            batch_img_roi_cls_ids_msk_padded.append(img_roi_cls_ids_msk_padded)
            batch_img_roi_msk.append(img_roi_msk)
            
            batch_label.append(label)

        return {
            'text': batch_text,
            'aspect': batch_aspect,
            
            'aspect_msk': torch.tensor(batch_aspect_msk, dtype=torch.float),
            
            'input_ids': torch.tensor(batch_input_ids),
            'attention_mask': torch.tensor(batch_attention_mask),
            
            'img_roi_features': torch.tensor(np.array(batch_img_roi_features_padded)),
            'img_roi_cls': batch_img_roi_cls,
            'img_roi_cls_ids': torch.tensor(batch_img_roi_cls_ids_padded),
            'img_roi_cls_ids_msk': torch.tensor(batch_img_roi_cls_ids_msk_padded),
            'img_roi_msk': torch.tensor(batch_img_roi_msk),
            
            'label': torch.tensor(batch_label)
        }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
