"""Transform all the IQ VQA dataset into a hdf5 dataset.
"""

import base64
import csv
from functools import total_ordering
import pickle
import sys
from PIL import Image
from numpy.core.fromnumeric import mean
from numpy.lib.type_check import imag
from torch._C import dtype
from torch.nn.functional import cosine_embedding_loss
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import argparse
import json
import h5py
import numpy as np
import os
import progressbar
import copy
from tqdm import tqdm
import cv2
import re
import nltk
from scipy.spatial import distance

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

csv.field_size_limit(sys.maxsize)

STOP_WORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "it", "its", "itself", "them", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
              "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tokenize(sentence):
    """Tokenizes a sentence into words.

    Args:
        sentence: A string of words.

    Returns:
        A list of words.
    """
    if len(sentence) == 0:
        return []
    sentence = re.sub('\.+', r'.', sentence)
    sentence = re.sub('([a-z])([.,!?()])', r'\1 \2 ', sentence)
    sentence = re.sub('\s+', ' ', sentence)

    tokens = nltk.tokenize.word_tokenize(
        sentence.strip().lower())
    return tokens


class EncoderCNN(nn.Module):
    """Generates a representation for an image input.
    """

    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer.
        """
        super(EncoderCNN, self).__init__()
        self.cnn = models.resnet18(pretrained=True).to(device)
        modules = list(self.cnn.children())[:-1]
        self.cnn = nn.Sequential(*modules)

    def forward(self, images):
        """Extract the image feature vectors.
        """
        features = self.cnn(images).squeeze()
        return features


def create_answer_mapping(annotations, ans2cat):
    """Returns mapping from question_id to answer.

    Only returns those mappings that map to one of the answers in ans2cat.

    Args:
        annotations: VQA annotations file.
        ans2cat: Map from answers to answer categories that we care about.

    Returns:
        answers: Mapping from question ids to answers.
        image_ids: Set of image ids.
    """
    answers = {}
    image_ids = set()
    for q in annotations['annotations']:
        question_id = q['question_id']
        answer = q['multiple_choice_answer']
        if answer in ans2cat:
            answers[question_id] = answer
            image_ids.add(q['image_id'])
    return answers, image_ids


def read_image_features_tsv(tsv_in_file, FIELDNAMES=['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']):
    image_feature_data = {}
    reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
    for i, item in tqdm(enumerate(reader)):
        item['image_id'] = int(item['image_id'])
        item['image_h'] = int(item['image_h'])
        item['image_w'] = int(item['image_w'])
        item['num_boxes'] = int(item['num_boxes'])
        for field in ['boxes', 'features']:
            try:
                item[field] = np.frombuffer(
                    base64.b64decode(item[field].encode() + b'==='),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
            except:
                print("Error processing file {}".format(item["image_id"]))
                continue

        boxes = copy.deepcopy(item["boxes"])
        boxes[:, [0, 2]] /= item['image_w']
        boxes[:, [1, 3]] /= item['image_h']

    # Normalized box areas
        areas = (boxes[:, 2] - boxes[:, 0]) * \
            (boxes[:, 3] - boxes[:, 1])

        image_feature_data[item['image_id']] = {
            "normalized_boxes_area": np.c_[boxes, areas],
            "features": item["features"]
        }
    return image_feature_data


def extract_labels_from_scores(mean_similarity_array, label_list, set_k):

    top_args = list(np.argsort(-mean_similarity_array))
    if len(top_args) < set_k:
        diff = set_k-len(top_args)
        empty = [-1] * diff
        top_args.extend(empty)

    top_args = np.array(top_args[:set_k])

    k_object_labels = []
    for idx in top_args:
        if idx == -1:
            k_object_labels.append("<EMPTY>")
        else:
            k_object_labels.append(label_list[idx])

    return k_object_labels


def process_string_objects(x):
    x = json.loads(x)
    object_set = set()
    for objs_info in x:
        obj = objs_info["class"]
        object_set.add(obj)
    x = object_set
    return list(x)


def filter_if_not_in_glove(token_list, glove_keys):
    filtered_token_list = []
    to_remove_from_tokens = []
    for t, label in enumerate(token_list):
        if label in glove_keys:
            filtered_token_list.append(label)

    return filtered_token_list


def find_similar_vectors(object_vectors, object_labels, x_vectors, x_labels, k=5):
    similarity_array = np.zeros((len(object_vectors), len(x_vectors)), dtype=np.float32)

    for o, obj_vector in enumerate(object_vectors):
        for t, token_vector in enumerate(x_vectors):
            d = distance.cosine(obj_vector, token_vector)
            similarity_array[o, t] = d

    mean_similarity_cap = np.mean(similarity_array, axis=0)  # [len(tokens)]
    mean_similarity_obj = np.mean(similarity_array, axis=1)  # [len(object_labels)]

    k_x_labels = extract_labels_from_scores(mean_similarity_cap, x_labels, k)
    k_object_labels = extract_labels_from_scores(mean_similarity_obj, object_labels, k)

    return k_object_labels, k_x_labels


def save_dataset(image_dir, questions, annotations, raw_data_dir, ans2cat, output,
                 im_size=224, train_or_val="train", set_k=5, num_objects=36):
    # Load the data.
    with open(annotations) as f:
        annos = json.load(f)
    with open(questions) as f:
        questions = json.load(f)

    print("Loading Glove Vectors...")
    glove_file = open("data/glove.6B.300d.txt", "r", encoding="utf8")
    glove_vectors_list = [word_and_vector.strip() for word_and_vector in glove_file.readlines()]
    glove_vectors = {obj.split()[0]: np.asarray(obj.split()[1:], dtype=np.float) for obj in glove_vectors_list}
    print("Loaded Glove Vectors!")

    cnn = EncoderCNN().to(device)

    objects_filepaths = [
        "{}/val.label.tsv".format(raw_data_dir),
        "{}/test.label.tsv".format(raw_data_dir),
        "{}/train.label.tsv".format(raw_data_dir)
    ]

    caption_features_filepaths = [
        "{}/pred.coco_caption.val.beam5.max20.odlabels.tsv".format(raw_data_dir),
        "{}/pred.coco_caption.test.beam5.max20.odlabels.tsv".format(raw_data_dir),
        "{}/pred.coco_caption.train.beam5.max20.odlabels.tsv".format(raw_data_dir)
    ]

    print("Loading object file...")
    d = []
    set_keys = set()
    for filepath in objects_filepaths:
        print("Loading file:", filepath)
        with open(filepath, "r") as f:
            for i, line in enumerate(tqdm(f)):
                # if i > 100:
                #     break
                line = tuple(line.split("\t"))
                d.append((int(line[0]), line[1]))

    df = pd.DataFrame(d, columns=["id", "objects"])
    df["objects"] = df["objects"].apply(lambda x: process_string_objects(x))
    df_objects = df.set_index("id")
    set_keys.update(df_objects.index)
    print("Loaded object file!")

    print("Loading captions/features file...")
    caption_features = []
    d = []
    for filepath in caption_features_filepaths:
        print("Loading file:", filepath)
        with open(filepath, "r") as f:
            for i, line in enumerate(tqdm(f, total=113287)):

                line = tuple(line.split("\t"))
                id = int(line[0])
                caption_features = json.loads(line[1])[0]
                captions = caption_features["caption"]
                image_features = np.array(caption_features["image_features"])
                d.append((id, captions, image_features))

    print("Building captions/features dataframe...")
    df = pd.DataFrame(d, columns=["id", "captions", "image_features"])
    print("Building captions/features dataframe...")
    del d
    df_caption_features = df.set_index("id")
    print(df_caption_features.head())
    print("Loaded captions/features file!")

    # df = pd.merge(df_objects, df_caption_features, on="id").set_index("id")

    # Get the mappings from qid to answers.
    qid2ans, image_ids = create_answer_mapping(annos, ans2cat)
    total_questions = len(list(qid2ans.keys()))
    total_images = len(image_ids)
    print("Number of images to be written: %d" % total_images)
    print("Number of QAs to be written: %d" % total_questions)

    string_dt = h5py.string_dtype(encoding='utf-8')
    h5file = h5py.File(output, "w")
    d_questions = h5file.create_dataset(
        "questions", (total_questions,), dtype=string_dt)
    d_indices = h5file.create_dataset(
        "image_indices", (total_questions,), dtype='i')
    d_images = h5file.create_dataset(
        "images", (total_images, 512), dtype='f')
    d_answer_types = h5file.create_dataset(
        "answer_types", (total_questions,), dtype='i')
    d_image_ids = h5file.create_dataset(
        "image_ids", (total_questions,), dtype='i')
    d_object_features = h5file.create_dataset(
        "object_features", (total_questions, num_objects, 2054), dtype='f')
    d_obj_labels = h5file.create_dataset(
        "obj_labels", (total_questions, 2*set_k), dtype=string_dt
    )
    d_captions = h5file.create_dataset(
        "captions", (total_questions,), dtype=string_dt
    )
    d_caption_labels_from_object = h5file.create_dataset(
        "caption_labels_from_object", (total_questions, set_k), dtype=string_dt
    )
    d_objects_from_qa_labels = h5file.create_dataset(
        "objects_from_qa_labels", (total_questions, 3), dtype=string_dt
    )
    d_qa_labels_from_object = h5file.create_dataset(
        "qa_labels_from_object", (total_questions, set_k), dtype=string_dt
    )

    # Create the transforms we want to apply to every image.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224,
                                     scale=(1.00, 1.2),
                                     ratio=(0.75, 1.3333333333333333)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # Iterate and save all the questions and images.
    # bar = progressbar.ProgressBar(maxval=total_questions)
    # bar.start()
    i_index = 0
    q_index = 0
    done_img2idx = {}
    found_images = set()
    not_found_images = set()
    for entry in tqdm(questions['questions']):
        image_id = entry['image_id']
        question_id = entry['question_id']
        if image_id not in image_ids:
            continue
        if question_id not in qid2ans:
            continue
        if image_id not in done_img2idx:

            try:
                path = "COCO_%s2014_%d.jpg" % (train_or_val, image_id)
                image = Image.open(os.path.join(
                    image_dir, path)).convert('RGB')
                image_for_predictor = cv2.imread(os.path.join(
                    image_dir, path))
            except IOError:
                try:
                    path = "COCO_%s2014_%012d.jpg" % (train_or_val, image_id)
                    image = Image.open(os.path.join(
                        image_dir, path)).convert('RGB')
                    image_for_predictor = cv2.imread(os.path.join(
                        image_dir, path))
                except:
                    print("COULD NOT FIND IMAGE {}".format(path))
                    continue
            image = transform(image)

            image_features = cnn(image.unsqueeze(0).to(device))
            image_features = image_features.squeeze(0).detach().cpu().numpy()
            d_images[i_index] = image_features
            done_img2idx[image_id] = i_index
            i_index += 1

        question = entry["question"]
        answer = qid2ans[question_id]
        caption = df_caption_features.loc[image_id]["captions"]
        # object_labels = df.loc[image_id]["objects"]
        object_labels = df_objects.loc[image_id]["objects"]

        temp_object_labels = []
        for label in object_labels:
            # label can be multiple words (e.g. sports ball). Let's take the average vector
            split_label = label.lower().split()
            temp_object_labels.extend(split_label)
        object_labels = list(set(temp_object_labels))
        object_labels = filter_if_not_in_glove(object_labels, glove_vectors.keys())

        object_label_vectors = []
        for label in object_labels:
            object_label_vectors.append(glove_vectors[label])

        set_obj_labels = set()
        # related caption tokens
        caption_tokens = list(set(tokenize(caption.lower().strip())))
        caption_tokens = filter_if_not_in_glove(caption_tokens, glove_vectors.keys())
        caption_vectors = [glove_vectors[label] for label in caption_tokens]
        obj_cap_labels, similar_cap_labels = find_similar_vectors(object_label_vectors, object_labels, caption_vectors, caption_tokens, k=set_k)
        set_obj_labels.update(obj_cap_labels)
        d_caption_labels_from_object[q_index] = np.array(similar_cap_labels)

        # related question tokens
        qa_token_string = question + " " + answer
        qa_tokens = list(set(tokenize(qa_token_string.lower().strip())))
        qa_tokens = filter_if_not_in_glove(qa_tokens, glove_vectors.keys())
        qa_vectors = [glove_vectors[label] for label in qa_tokens]
        obj_qa_labels, similar_qa_labels = find_similar_vectors(object_label_vectors, object_labels, qa_vectors, qa_tokens, k=3)
        d_objects_from_qa_labels[q_index] = np.array(obj_qa_labels)
        obj_qa_labels, similar_qa_labels = find_similar_vectors(object_label_vectors, object_labels, qa_vectors, qa_tokens, k=set_k)
        set_obj_labels.update(obj_qa_labels)
        d_qa_labels_from_object[q_index] = np.array(similar_qa_labels)

        obj_pad_len = 2*set_k
        set_obj_labels = list(set_obj_labels)
        while len(set_obj_labels) < obj_pad_len:
            set_obj_labels.append("<EMPTY>")

        d_obj_labels[q_index] = np.array(set_obj_labels)
        d_image_ids[q_index] = image_id
        d_questions[q_index] = question
        d_captions[q_index] = caption

        answer = qid2ans[question_id]
        d_answer_types[q_index] = int(ans2cat[answer])
        d_indices[q_index] = done_img2idx[image_id]

        truncated_image_features = df_caption_features.loc[image_id]["image_features"][:num_objects]
        d_object_features[q_index] = truncated_image_features

        q_index += 1
        # bar.update(q_index)
    h5file.close()
    print("Number of images written: %d" % i_index)
    print("Number of QAs written: %d" % q_index)
    print("Number of images found ({}) vs not found ({})".format(
        len(found_images), len(not_found_images)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Inputs.
    parser.add_argument('--image-dir', type=str, default='/data/lama/mscoco/images/train2014/',
                        help='directory for resized images')
    parser.add_argument('--questions', type=str,
                        default='/data/nv419/VQG_DATA/raw/v2_OpenEnded_mscoco_'
                        'train2014_questions.json',
                        help='Path for train annotation file.')
    parser.add_argument('--annotations', type=str,
                        default='/data/nv419/VQG_DATA/raw/v2_mscoco_'
                        'train2014_annotations.json',
                        help='Path for train annotation file.')
    parser.add_argument('--cat2ans', type=str,
                        default='/data/nv419/VQG_DATA/raw/iq_dataset.json',
                        help='Path for the answer types.')
    parser.add_argument('--raw_data_dir', type=str,
                        default="/data/nv419/VQG_DATA/raw",
                        help="Path for the raw data files. Should include *.label.tsv and pred.coco_caption.* etc")

    # Outputs.
    parser.add_argument('--output', type=str,
                        default='/data/nv419/VQG_DATA/processed/iq_dataset.hdf5',
                        help='directory for resized images.')
    parser.add_argument('--cat2name', type=str,
                        default='/data/nv419/VQG_DATA/processed/cat2name.json',
                        help='Location of mapping from category to type name.')

    # Hyperparameters.
    parser.add_argument('--im_size', type=int, default=224,
                        help='Size of images.')
    parser.add_argument('--max-q-length', type=int, default=20,
                        help='maximum sequence length for questions.')
    parser.add_argument('--max-a-length', type=int, default=4,
                        help='maximum sequence length for answers.')

    # Train or Val?
    parser.add_argument('--val', type=bool, default=False,
                        help="whether we're working iwth the validation set or not")
    args = parser.parse_args()

    ans2cat = {}
    with open(args.cat2ans) as f:
        cat2ans = json.load(f)
    cats = sorted(cat2ans.keys())
    with open(args.cat2name, 'w') as f:
        json.dump(cats, f)
    for cat in cat2ans:
        for ans in cat2ans[cat]:
            ans2cat[ans] = cats.index(cat)

    train_or_val = "train"
    if args.val == True:
        train_or_val = "val"

    save_dataset(args.image_dir, args.questions, args.annotations, args.raw_data_dir, ans2cat, args.output,
                 im_size=224, train_or_val=train_or_val, set_k=5)
    print(('Wrote dataset to %s' % args.output))
    # Hack to avoid import errors.
