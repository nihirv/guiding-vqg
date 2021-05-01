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
        self.max_input_len = 25
        self.max_legal_len = 17
        self.max_oqa_inference_len = 11
        self.max_q_len = 26
        self.max_cap_len = 30

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

    # *args are inputs, each input being a list of strings
    def build_inputs(self, len, *args):
        args_list = []
        for arg in args:
            args_list += arg
        # args_list is now a 1d list of strings which contains all tokens in all *args
        args_list = list(dict.fromkeys(args_list))  # unique words only. dict() preserves ordering which means we can easily extract out the category in the main.py file
        args_list = [word for word in args_list if word != "<EMPTY>"]
        args_string = " ".join(args_list)
        args_ids, args_attn_mask = self.tokenize_and_pad(args_string, len)
        return args_ids, args_attn_mask

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
            self.object_features = annos["object_features"]
            self.obj_labels = annos["obj_labels"]
            self.captions = annos["captions"]
            self.caption_labels_from_object = annos["caption_labels_from_object"]
            self.objects_from_qa_labels = annos["objects_from_qa_labels"]
            self.qa_labels_from_object = annos["qa_labels_from_object"]

        if self.indices is not None:
            index = self.indices[index]

        question = self.questions[index]  # natural language. Let's tokenize
        encoded_question_id, encoded_question_attention_mask = self.tokenize_and_pad(question, self.max_q_len)

        caption = self.captions[index]
        encded_caption_id, encoded_caption_attention_mask = self.tokenize_and_pad(caption, self.max_cap_len)

        category = self.answer_types[index]
        category_label = [self.cat2name[category]]                   # ['binary']
        obj_label = list(self.obj_labels[index])                    # ['trousers', 'swimming', 'racket', 'jeans', 'footwear', 'furniture', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>']
        co_label = list(self.caption_labels_from_object[index])     # ['court', 'holding', 'racket', 'on', 'a']
        oqa_label = list(self.objects_from_qa_labels[index])        # ['footwear', 'trousers', 'racket']
        qao_label = list(self.qa_labels_from_object[index])         # ['yes', 'play', 'can', 'he', '?']

        # all_train_inputs # category label, object labels, co, oqa, qao, (caption)
        # legal_inputs=inference_inputs # category label, object labels, co, (caption)
        # qa_inference_inputs # category label, oqa, co, (caption)

        all_train_input_ids, all_train_input_attn_mask = self.build_inputs(self.max_input_len, category_label, obj_label, co_label, oqa_label, qao_label)
        legal_input_ids, legal_input_attn_mask = self.build_inputs(self.max_legal_len, category_label, obj_label, co_label)
        qa_inference_input_ids, qa_inference_input_attn_mask = self.build_inputs(self.max_oqa_inference_len, category_label, oqa_label, co_label)

        object_features_full_vector = self.object_features[index]  # 2054-d
        object_features = object_features_full_vector[:, :2048]  # 2048-d
        object_locations = object_features_full_vector[:, 2048:]  # 6-d
        object_features = torch.from_numpy(object_features)
        object_locations = torch.from_numpy(object_locations)

        image_index = self.image_indices[index]
        image = self.images[image_index]
        image_id = self.image_ids[index]

        return image_id, torch.from_numpy(image), encoded_question_id, encoded_question_attention_mask, all_train_input_ids, all_train_input_attn_mask, legal_input_ids, legal_input_attn_mask, qa_inference_input_ids, qa_inference_input_attn_mask, object_features, object_locations

    def __len__(self):
        if self.max_examples is not None:
            return self.max_examples
        if self.indices is not None:
            return len(self.indices)
        annos = h5py.File(self.dataset, 'r')
        return annos['questions'].shape[0]


def collate_fn(data):

    image_ids, images, encoded_question_id, encoded_question_attention_mask, all_train_input_ids, all_train_input_attn_mask, legal_input_ids, legal_input_attn_mask, qa_inference_input_ids, qa_inference_input_attn_mask, object_features, object_locations = list(
        zip(*data))

    images = torch.stack(images).float()
    question_ids = torch.stack(encoded_question_id).long()
    question_attention_masks = torch.stack(encoded_question_attention_mask).long()
    input_ids = torch.stack(all_train_input_ids).long()
    input_attention_masks = torch.stack(all_train_input_attn_mask).long()
    legal_ids = torch.stack(legal_input_ids).long()
    legal_attention_masks = torch.stack(legal_input_attn_mask).long()
    qa_inference_ids = torch.stack(qa_inference_input_ids).long()
    qa_inference_attention_masks = torch.stack(qa_inference_input_attn_mask).long()
    object_features = torch.stack(object_features)
    object_locations = torch.stack(object_locations)

    return {"images": images,
            "image_ids": image_ids,
            "question_ids": question_ids,
            "question_attention_masks": question_attention_masks,
            "input_ids": input_ids,
            "input_attention_masks": input_attention_masks,
            "legal_ids": legal_ids,
            "legal_attention_masks": legal_attention_masks,
            "qa_inference_ids": qa_inference_ids,
            "qa_inference_attention_masks": qa_inference_attention_masks,
            "object_features": object_features,
            "object_locations": object_locations
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
