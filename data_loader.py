"""Loads question answering data and feeds it to the models.
"""
import json
import h5py
import numpy as np
import torch
from torch._C import dtype
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
        self.max_inference_len = 8
        self.max_q_len = 26

        self.cat2name = sorted(
            json.load(open("data/processed/cat2name.json", "r")))

    def tokenize_and_pad(self, tokens, max_len):
        encoded = self.tokenizer(tokens)
        encoded_id = torch.tensor(encoded["input_ids"])
        encoded_attention_mask = torch.tensor(encoded["attention_mask"])

        len_diff = max_len - len(encoded_id)
        if len_diff < 0:
            encoded_id = encoded_id[:max_len]
            encoded_attention_mask = encoded_attention_mask[:max_len]
        else:
            pads = torch.tensor([self.tokenizer.pad_token_id] * len_diff)
            pads_for_attn_mask = torch.ones_like(pads)
            encoded_id = torch.cat((encoded_id, pads), dim=-1)
            encoded_attention_mask = torch.cat((encoded_attention_mask, pads_for_attn_mask), dim=-1)

        return encoded_id, encoded_attention_mask

    def __getitem__(self, index):
        """Returns one data pair (image and caption).
        """
        # if not hasattr(self, 'images'):
        #     annos = h5py.File(self.dataset, 'r')
        #     self.questions = annos['questions']
        #     self.answer_types = annos['answer_types']
        #     self.image_indices = annos['image_indices']
        #     self.images = annos['images']
        #     self.image_ids = annos["image_ids"]
        #     self.obj_labels = annos["rcnn_obj_labels"]
        #     self.cap_labels = annos["rcnn_cap_labels"]
        #     self.object_features = annos["rcnn_features"]
        #     self.object_locations = annos["rcnn_locations"]

        # if self.indices is not None:
        #     index = self.indices[index]

        # question = self.questions[index]  # natural language. Let's tokenize
        # category = self.answer_types[index]
        # obj_label = list(self.obj_labels[index])  # natural language
        # cap_label = list(self.cap_labels[index])  # natural language

        # category_word = [self.cat2name[category]]  # english word

        # cat_obj_labels = category_word + obj_label
        # cat_obj_labels = [word for word in cat_obj_labels if word != "<EMPTY>"]
        # cat_obj_labels = " ".join(cat_obj_labels)
        # encoded_cat_obj_ids, encoded_cat_obj_attn_mask = self.tokenize_and_pad(cat_obj_labels, self.max_inference_len)

        # input_concat = category_word + obj_label + cap_label
        # input_concat = [word for word in input_concat if word != "<EMPTY>"]
        # input_string = " ".join(input_concat)
        # encoded_input_id, encoded_input_attention_mask = self.tokenize_and_pad(input_string, self.max_input_len)

        # encoded_question_id, encoded_question_attention_mask = self.tokenize_and_pad(question, self.max_q_len)

        # rcnn_features = torch.from_numpy(self.object_features[index])
        # rcnn_locations = torch.from_numpy(self.object_locations[index])

        # image_index = self.image_indices[index]
        # image = self.images[image_index]
        # image_id = self.image_ids[index]

        image_id = 0
        image = np.random.randn(512)
        encoded_input_id = torch.tensor(list(range(self.max_input_len)))
        encoded_input_attention_mask = torch.tensor([0]*self.max_input_len)
        encoded_question_id = torch.tensor(list(range(self.max_q_len)))
        encoded_question_attention_mask = torch.tensor([0]*self.max_q_len)
        encoded_cat_obj_ids = torch.tensor(list(range(self.max_inference_len)))
        encoded_cat_obj_attn_mask = torch.tensor([0]*self.max_inference_len)
        rcnn_features = torch.tensor(0)
        rcnn_locations = torch.tensor(0)

        return image_id, torch.from_numpy(image), encoded_input_id, encoded_input_attention_mask, encoded_question_id, encoded_question_attention_mask, encoded_cat_obj_ids, encoded_cat_obj_attn_mask, rcnn_features, rcnn_locations

    def __len__(self):
        return 300
        # if self.max_examples is not None:
        #     return self.max_examples
        # if self.indices is not None:
        #     return len(self.indices)
        # annos = h5py.File(self.dataset, 'r')
        # return annos['questions'].shape[0]


def collate_fn(data):

    image_ids, images, encoded_input_ids, encoded_input_attention_masks, encoded_question_ids, encoded_question_attention_masks, encoded_cat_obj_ids, encoded_cat_obj_attn_mask, rcnn_features, rcnn_locations = list(
        zip(*data))

    images = torch.stack(images).float()
    input_ids = torch.stack(encoded_input_ids).long()
    input_attention_masks = torch.stack(encoded_input_attention_masks).long()
    question_ids = torch.stack(encoded_question_ids).long()
    question_attention_masks = torch.stack(encoded_question_attention_masks).long()
    inference_ids = torch.stack(encoded_cat_obj_ids).long()
    inference_attention_masks = torch.stack(encoded_cat_obj_attn_mask).long()
    rcnn_features = torch.stack(rcnn_features)
    rcnn_locations = torch.stack(rcnn_locations)

    return {"images": images,
            "image_ids": image_ids,
            "question_ids": question_ids,
            "question_attention_masks": question_attention_masks,
            "input_ids": input_ids,
            "input_attention_masks": input_attention_masks,
            "inference_ids": inference_ids,
            "inference_attention_masks": inference_attention_masks,
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
