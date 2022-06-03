
# coding: utf-8
"""Dataset Loader for Memory Dialogs.

Author(s): noctli, skottur
"""

import json
import logging
import os
import pickle
import re
from itertools import chain

import numpy as np
import torch
import torch.utils.data
import tqdm

from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image


# from train import SPECIAL_TOKENS, MODEL_INPUTS, PADDED_INPUTS
# SPECIAL_TOKENS = ["<bos>", "<eos>", "<user>", "<system>", "<video>", "<pad>"]
# SPECIAL_TOKENS_DICT = {
#     "bos_token": "<bos>",
#     "eos_token": "<eos>",
#     "additional_special_tokens": ["<user>", "<system>", "<video>", "<cap>"],
#     "pad_token": "<pad>",
# }
MODEL_INPUTS = ["input_ids", "token_type_ids", "lm_labels"]
PADDED_INPUTS = ["input_ids", "token_type_ids", "lm_labels"]
MEMORY_BREAK = "<MM_BREAK>"
ANCHOR_TOKENS = ["<USER>", "<SYSTEM>", "<MM>", "<SOAC>", "<SOAR>", "<SOR>"]


class MemoryDialogDataset(Dataset):
    def __init__(self, transform, comet_root, coco_root, coco_annotations, dialog_files = [], memory_files = [], split='train'):
        
        self.transform = transform
        self.split = split
        self.memory_files = memory_files
        self.comet_root = comet_root

        if split == 'train':
            self.dialogs = json.load(open(os.path.join(comet_root, dialog_files[0]), 'r', encoding='utf-8'))
        elif split=='val':
            self.dialogs = json.load(open(os.path.join(comet_root, dialog_files[1]), 'r', encoding='utf-8'))
        else:
            self.dialogs = json.load(open(os.path.join(comet_root, dialog_files[2]), 'r', encoding='utf-8'))

        #Mapping of memories to their information
        self.memories_to_data = self._parse_memory_graphs()

        #COCO api for accessing images dynamically
        self.coco = COCO(coco_annotations)
        self.coco_root = coco_root

    def _parse_memory_graphs(self):
        #First merge all memory graphs // There are two files in dataset
        memories_to_data = {}
        repeated_memories = 0
        for file_path in self.memory_files:
            with open(os.path.join(self.comet_root, file_path), 'r', encoding='utf-8') as input_file:
                graph_data = json.load(input_file)
                for graph in graph_data:
                    for memory in graph['memories']:
                        if memory['memory_id'] in memories_to_data:
                            repeated_memories += 1
                        else:
                            memories_to_data[memory['memory_id']] = memory

        print(f"There are {repeated_memories} repeated memories in the dataset")
        print(f"Read {len(memories_to_data)} memories loaded")

        return memories_to_data
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, index):
        instance = self.dialogs[index]
        input_ids = []
        split_str = instance['predict'].split(MEMORY_BREAK)
        memory_ids = [int(element.rsplit(" ", 1)[-1]) for element in split_str[:-1]]
        images  = []
        #Load images
        for mem_id in memory_ids:
            image_inf = self.coco.loadImgs(self.memories_to_data[mem_id]['media'][0]['media_id'])
            image = Image.open(os.path.join(self.coco_root, image_inf[0]['file_name'])).convert('RGB')
            image = self.transform(image)
            images.append(image)
            
        
        return images, instance['predict'], instance['target']

def collate_fn(batch):
    image_list, predict_list, target_list = [], [], []
    #For the case of no image on the context
    max_num_images = min(8, max([len(element[0]) for element in batch]))
    max_num_images = 1 if max_num_images == 0 else max_num_images
    images_mask = []
    
    #Hardcoding this is not optimal
    image_shape = (3, 480, 480)
    
    for images, predict, target in batch:
        current_images = len(images)
        current_mask = torch.cat((torch.ones(current_images), torch.zeros(max_num_images-current_images)), 0)
        for i in range(current_images, max_num_images):
            images.append(torch.zeros(image_shape))
        image_list.append(torch.stack(images[:max_num_images], dim=0))
        predict_list.append(predict)
        target_list.append(target)
        images_mask.append(current_mask)
        
    return torch.stack(image_list, dim=0), predict_list, target_list, torch.stack(images_mask, dim=0)

