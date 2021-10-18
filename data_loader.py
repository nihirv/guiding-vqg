from collections import OrderedDict
from tokenizers.processors import TemplateProcessing
from transformers import BertTokenizer
import torch.utils.data as data
from torch._C import dtype
import torch
import numpy as np
import h5py
import json
"""Loads question answering data and feeds it to the models.
"""
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
        self.max_variational_len = 25
        self.max_generative_len = 38
        self.max_oqa_inference_len = 11
        self.max_q_len = 26
        self.max_qa_len = 30
        self.max_cap_len = 30
        self.max_obj_len = 36
        self.PAD_token = self.tokenizer.pad_token  # '[PAD]'
        self.STOP_WORDS = ["?", ".", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "it", "its", "itself", "them", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
                           "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
        self.cat2name = sorted(
            json.load(open("data/processed/cat2name.json", "r")))

        self.multiple_references = self.get_multiple_questions()


    def get_multiple_questions(self):
        with open("/data2/zwang/data/vqg/v2_OpenEnded_mscoco_train2014_questions.json") as f1:
            question_file1 = json.load(f1)
        with open("/data2/zwang/data/vqg/v2_OpenEnded_mscoco_val2014_questions.json") as f2:
            question_file2 = json.load(f2)  

        final_dict = {}
        for entry in question_file1['questions']:
            image_id = entry['image_id']
            question = entry['question']
            if image_id not in final_dict:
                final_dict[image_id] = [question]
            else:
                final_dict[image_id].append(question)

        for entry in question_file2['questions']:
            image_id = entry['image_id']
            question = entry['question']
            if image_id not in final_dict:
                final_dict[image_id] = [question]
            else:
                final_dict[image_id].append(question)

        return final_dict


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
        # args_list.append("[CLS]")
        for arg in args:
            args_list += arg
            # args_list.append("[SEP]")
        # args_list is now a 1d list of strings which contains all tokens in all *args
        # args_list = list(dict.fromkeys(args_list))  # unique words only. dict() preserves ordering which means we can easily extract out the category in the main.py file
        args_list = [word for word in args_list if word != "<EMPTY >"]
        args_string = " ".join(args_list)
        # print(args_string)
        args_ids, args_attn_mask = self.tokenize_and_pad(args_string, len)
        return args_ids, args_attn_mask

    def filter_stop_words(self, list_to_filter):
        return [item for item in list_to_filter if item not in self.STOP_WORDS]

    def pad_features(self, rank2_feature):
        len_difference = self.max_obj_len - rank2_feature.shape[0]  # e.g. 36 - 20 = 16
        zeros = np.zeros((len_difference, rank2_feature.shape[1]))  # shape[1] => either 768, 2048 or 5
        rank2_feature = np.concatenate((rank2_feature, zeros), axis=0)
        return rank2_feature

    def remove_duplicate_objects_and_pad_features(self, labels, embeddings, features, locations):
        labels_dict = OrderedDict((x, labels.index(x)) for x in labels)
        return_labels = list(labels_dict.keys())
        keep_indexes = list(labels_dict.values())
        embeddings = np.take(embeddings, keep_indexes, axis=0)  # [T, 768]
        features = np.take(features, keep_indexes, axis=0)  # [T, 2048]
        locations = np.take(locations, keep_indexes, axis=0)  # [T, 5]
        obj_len_diff = self.max_obj_len - len(list(return_labels))
        pads = [self.PAD_token] * obj_len_diff
        return_labels = list(return_labels) + pads
        embeddings = self.pad_features(embeddings)
        features = self.pad_features(features)
        locations = self.pad_features(locations)
        return return_labels, embeddings, features, locations

    def __getitem__(self, index):
        """Returns one data pair(image and caption).
        """
        if not hasattr(self, "images"):
            annos = h5py.File(self.dataset, "r")
            # print(dict(annos).keys())
            self.questions = annos["questions"]
            self.answer_types = annos["answer_types"]
            self.image_indices = annos["image_indices"]
            self.images = annos["images"]
            self.image_ids = annos["image_ids"]
            self.answers = annos["answers"]
            self.object_labels = annos["bottom_up_obj_labels"]
            self.object_features = annos["bottom_up_obj_features"]
            self.object_locations = annos["bottom_up_obj_locations"]
            self.obj_embeddings = annos["bottom_up_obj_embeddings"]

        if self.indices is not None:
            index = self.indices[index]

        obj_embedding = self.obj_embeddings[index]
        obj_label = list(self.object_labels[index])     # ["toilet", "wall", "floor", "toilet paper", "base", "bathroom" ...]
        question = self.questions[index]             # Is this bathroom clean?
        answer = self.answers[index]                 # yes
        category = self.answer_types[index]          # 3
        category_label = self.cat2name[category]     # binary
        category = torch.Tensor([category]).long()

        qa = question.lower()[:-1].split() + ["[SEP]"] + answer.split()
        qa = " ".join(qa)
        encoded_question_id, encoded_question_attention_mask = self.tokenize_and_pad(question, self.max_q_len)
        encoded_qa_id, encoded_qa_attention_mask = self.tokenize_and_pad(qa, self.max_qa_len)

        object_features = self.object_features[index]  # 2048-d
        object_locations = self.object_locations[index]  # 5-d

        # obj_label, obj_embedding, object_features, object_locations = self.remove_duplicate_objects_and_pad_features(obj_label, obj_embedding, object_features, object_locations)
        object_features = torch.from_numpy(object_features)
        object_locations = torch.from_numpy(object_locations)

        image_index = self.image_indices[index]
        image = self.images[image_index]
        image_id = self.image_ids[index]
        category_only_id, category_only_attn_mask = self.tokenize_and_pad(category_label, 3)

        if image_id in self.multiple_references:
            multiple_reference = self.multiple_references[image_id]
        else:
            multiple_reference = [question]
        # multiple_question_id = [self.tokenize_and_pad(q, self.max_q_len)[0] for q in multiple_reference]

        return image_id, torch.from_numpy(image), category, \
            object_features, object_locations, torch.from_numpy(obj_embedding), \
            encoded_question_id, encoded_question_attention_mask, \
            encoded_qa_id, encoded_qa_attention_mask, \
            multiple_reference, self.cat2name, obj_label

    def __len__(self):
        if self.max_examples is not None:
            return self.max_examples
        if self.indices is not None:
            return len(self.indices)
        annos = h5py.File(self.dataset, "r")
        return annos["questions"].shape[0]

def collate_fn(data):
    image_ids, images, categories, \
        object_features, object_locations, object_embeddings, \
        encoded_question_id, encoded_question_attention_mask, \
        encoded_qa_id, encoded_qa_attention_mask, multiple_question_ids, cat2name, obj_label = list(zip(*data))
    images = torch.stack(images).float()
    categories = torch.stack(categories).long()
    object_features = torch.stack(object_features)
    object_locations = torch.stack(object_locations)
    object_embeddings = torch.stack(object_embeddings)
    question_ids = torch.stack(encoded_question_id).long()
    question_attention_masks = torch.stack(encoded_question_attention_mask).long()
    qa_ids = torch.stack(encoded_qa_id).long()
    qa_attention_masks = torch.stack(encoded_qa_attention_mask).long()

    # print(question_ids)
    # print(multiple_question_ids)s

    # multiple_question_ids = torch.stack(multiple_question_ids).long()
    return {"image_ids": image_ids,
            "images": images,
            "categories": categories,
            "object_features": object_features,
            "object_locations": object_locations,
            "object_embeddings": object_embeddings,
            "question_ids": question_ids,
            "question_attention_masks": question_attention_masks,
            "qa_ids": qa_ids,
            "qa_attention_masks": qa_attention_masks,
            "multiple_question_ids": multiple_question_ids,
            "cat2name": cat2name,
            "obj_label": obj_label
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