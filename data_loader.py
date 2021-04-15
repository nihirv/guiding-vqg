"""Loads question answering data and feeds it to the models.
"""
import json
import h5py
import numpy as np
import torch
import torch.utils.data as data
from transformers import BertTokenizer
from tokenizers.processors import TemplateProcessing


class VQGDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, dataset, tokenizer, max_examples=None,
                 indices=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            dataset: hdf5 file with questions and images.
            images: hdf5 file with questions and imags.
            transform: image transformer.
            max_examples: Used for debugging. Assumes that we have a
                maximum number of training examples.
            indices: List of indices to use.
        """
        self.dataset = dataset
        self.max_examples = max_examples
        self.indices = indices
        self.tokenizer = tokenizer
        self.max_input_len = 14
        self.max_q_len = 26

        self.cat2name = sorted(
            json.load(open("data/processed/cat2name.json", "r")))

    def __getitem__(self, index):
        """Returns one data pair (image and caption).
        """
        if not hasattr(self, 'images'):
            annos = h5py.File(self.dataset, 'r')
            self.questions = annos['questions']
            self.answer_types = annos['answer_types']
            self.image_indices = annos['image_indices']
            self.images = annos['images']
            self.image_ids = annos["image_ids"]
            self.obj_labels = annos["rcnn_obj_labels"]
            self.cap_labels = annos["rcnn_cap_labels"]
            self.object_features = annos["rcnn_features"]
            self.object_locations = annos["rcnn_locations"]

        if self.indices is not None:
            index = self.indices[index]

        question = self.questions[index]  # natural language. Let's tokenize
        category = self.answer_types[index]
        obj_label = list(self.obj_labels[index])  # natural language
        cap_label = list(self.cap_labels[index])  # natural language

        category_word = [self.cat2name[category]]  # english word

        input_concat = category_word + obj_label + cap_label
        input_concat = [word for word in input_concat if word != "<EMPTY>"]
        input_string = " ".join(input_concat)

        encoded_inputs = self.tokenizer(input_string)  # 1 + 5 + 5 + 2 + 2 (2 for extra BPE, 2 for SOS/EOS)
        encoded_question = self.tokenizer(question)

        encoded_input_id = torch.tensor(encoded_inputs["input_ids"])
        encoded_input_attention_mask = torch.tensor(encoded_inputs["attention_mask"])
        len_diff = self.max_input_len - len(encoded_input_id)
        if len_diff < 0:
            encoded_input_id = encoded_input_id[:self.max_input_len]
            encoded_input_attention_mask = encoded_input_attention_mask[:self.max_input_len]
        else:
            pads = torch.tensor([self.tokenizer.pad_token_id] * len_diff)
            pads_for_attn_mask = torch.ones_like(pads)
            encoded_input_id = torch.cat((encoded_input_id, pads), dim=-1)
            encoded_input_attention_mask = torch.cat((encoded_input_attention_mask, pads_for_attn_mask), dim=-1)

        encoded_question_id = torch.tensor(encoded_question["input_ids"])
        encoded_question_attention_mask = torch.tensor(encoded_question["attention_mask"])
        len_diff = self.max_q_len - len(encoded_question_id)
        if len_diff < 0:
            encoded_question_id = encoded_question_id[:self.max_q_len]
            encoded_question_attention_mask = encoded_question_attention_mask[:self.max_q_len]
        else:
            pads = torch.tensor([self.tokenizer.pad_token_id] * len_diff)
            pads_for_attn_mask = torch.ones_like(pads)
            encoded_question_id = torch.cat((encoded_question_id, pads), dim=-1)
            encoded_question_attention_mask = torch.cat((encoded_question_attention_mask, pads_for_attn_mask), dim=-1)

        rcnn_features = torch.from_numpy(self.object_features[index])
        rcnn_locations = torch.from_numpy(self.object_locations[index])

        image_index = self.image_indices[index]
        image = self.images[image_index]
        image_id = self.image_ids[index]

        return image_id, torch.from_numpy(image), encoded_input_id, encoded_input_attention_mask, encoded_question_id, encoded_question_attention_mask, rcnn_features, rcnn_locations

    def __len__(self):
        if self.max_examples is not None:
            return self.max_examples
        if self.indices is not None:
            return len(self.indices)
        annos = h5py.File(self.dataset, 'r')
        return annos['questions'].shape[0]


def collate_fn(data):

    image_ids, images, encoded_input_ids, encoded_input_attention_masks, encoded_question_ids, encoded_question_attention_masks, rcnn_features, rcnn_locations = list(zip(*data))

    images = torch.stack(images)
    input_ids = torch.stack(encoded_input_ids).long()
    input_attention_masks = torch.stack(encoded_input_attention_masks).long()
    question_ids = torch.stack(encoded_question_ids).long()
    question_attention_masks = torch.stack(encoded_question_attention_masks).long()
    rcnn_features = torch.stack(rcnn_features)
    rcnn_locations = torch.stack(rcnn_locations)

    return {"images": images,
            "image_ids": image_ids,
            "question_ids": question_ids,
            "question_attention_masks": question_attention_masks,
            "input_ids": input_ids,
            "input_attention_masks": input_attention_masks,
            "rcnn_features": rcnn_features,
            "rcnn_locations": rcnn_locations
            }


def get_loader(dataset, tokenizer, batch_size, sampler=None,
               shuffle=True, num_workers=1, max_examples=None,
               indices=None):

    vqg = VQGDataset(dataset, tokenizer, max_examples=max_examples,
                     indices=indices)
    data_loader = torch.utils.data.DataLoader(dataset=vqg,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
