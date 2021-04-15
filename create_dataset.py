"""Transform all the IQ VQA dataset into a hdf5 dataset.
"""

import base64
import csv
import pickle
import sys
from PIL import Image
from numpy.core.fromnumeric import mean
from numpy.lib.type_check import imag
from torch.nn.functional import cosine_embedding_loss
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models

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


def save_dataset(image_dir, questions, annotations, ans2cat, output,
                 im_size=224, train_or_val="train", set_k=5, num_objects=36):
    # Load the data.
    with open(annotations) as f:
        annos = json.load(f)
    with open(questions) as f:
        questions = json.load(f)

    glove_file = open("data/glove.6B.300d.txt", "r", encoding="utf8")
    glove_vectors_list = [word_and_vector.strip() for word_and_vector in glove_file.readlines()]
    glove_vectors = {obj.split()[0]: np.asarray(obj.split()[1:], dtype=np.float) for obj in glove_vectors_list}

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    object_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    cnn = EncoderCNN().to(device)

    # Loading captions
    captions = pickle.load(open("data/vqa/captions_{}".format(train_or_val), "rb"))

    # Load RCNN features
    print("Loading RCNN features. This may take a while")
    image_feature_data = {}

    filepaths = [
        "data/updown/trainval/karpathy_test_resnet101_faster_rcnn_genome.tsv",
        "data/updown/trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.0",
        "data/updown/trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1",
        'data/updown/trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv'
    ]

    for filepath in filepaths:
        print("Loading file...")
        with open(filepath, "r") as tsv_in_file:
            ifd = read_image_features_tsv(tsv_in_file)
            image_feature_data.update(ifd)
    print("Loaded RCNN features!")

    # Get the mappings from qid to answers.
    qid2ans, image_ids = create_answer_mapping(annos, ans2cat)
    total_questions = len(list(qid2ans.keys()))
    total_images = len(image_ids)
    print("Number of images to be written: %d" % total_images)
    print("Number of QAs to be written: %d" % total_questions)

    string_dt = h5py.special_dtype(vlen=str)
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
    d_rcnn_features = h5file.create_dataset(
        "rcnn_features", (total_questions, num_objects, 2048), dtype='f')
    d_rcnn_locations = h5file.create_dataset(
        "rcnn_locations", (total_questions, num_objects, 5), dtype='f')
    d_rcnn_obj_labels = h5file.create_dataset(
        "rcnn_obj_labels", (total_questions, set_k), dtype=string_dt
    )
    d_rcnn_cap_labels = h5file.create_dataset(
        "rcnn_cap_labels", (total_questions, set_k), dtype=string_dt
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

        object_detectection = predictor(image_for_predictor)
        object_label_class_idxs = object_detectection["instances"].pred_classes
        object_labels = list(set([object_classes[item] for item in object_label_class_idxs]))

        question = entry["question"]
        captionz = " ".join(captions[image_id])
        answer = qid2ans[question_id]
        caption_question_ans = captionz + " " + question + " " + answer
        tokens = list(set(tokenize(caption_question_ans.lower().strip())))
        tokens = [token for token in tokens if token not in STOP_WORDS]

        new_object_labels = []
        for label in object_labels:
            # label can be multiple words (e.g. sports ball). Let's take the average vector
            split_label = label.split()
            new_object_labels.extend(split_label)

        object_labels = list(set(new_object_labels))

        object_label_vectors = []
        for label in object_labels:
            object_label_vectors.append(glove_vectors[label])

        token_vectors = []
        to_remove_from_tokens = []
        for t, label in enumerate(tokens):
            if label in glove_vectors:
                token_vectors.append(label)
            else:
                to_remove_from_tokens.append(t)

        for x in sorted(to_remove_from_tokens, reverse=True):
            del tokens[x]

        token_vectors = [glove_vectors[label] for label in tokens]

        # TODO: extract object vectors for object_label_vectors

        # construct a [len(object_labels), len(tokens)] matrix
        similarity_array = np.zeros((len(object_label_vectors), len(tokens)), dtype=np.float32)

        for o, obj_vector in enumerate(object_label_vectors):
            for t, token_vector in enumerate(token_vectors):
                d = distance.cosine(obj_vector, token_vector)
                similarity_array[o, t] = d

        mean_similarity_cap = np.mean(similarity_array, axis=0)  # [len(tokens)]
        mean_similarity_obj = np.mean(similarity_array, axis=1)  # [len(object_labels)]

        k_cap_labels = extract_labels_from_scores(mean_similarity_cap, tokens, set_k)
        k_object_labels = extract_labels_from_scores(mean_similarity_obj, object_labels, set_k)

        d_rcnn_obj_labels[q_index] = np.array(k_object_labels)
        d_rcnn_cap_labels[q_index] = np.array(k_cap_labels)
        d_questions[q_index] = question

        answer = qid2ans[question_id]
        d_answer_types[q_index] = int(ans2cat[answer])
        d_indices[q_index] = done_img2idx[image_id]
        d_image_ids[q_index] = image_id

        rcnn_features_zeros = np.zeros((num_objects, 2048), dtype=np.float32)
        rcnn_normalised_boxes = np.zeros((num_objects, 5), dtype=np.float32)

        try:
            relevant_image_feature_object = image_feature_data[image_id]
            found_images.add(image_id)
        except:
            # print("Skipping file {} due to an error. Most like the file could not be found.".format(
            #     image_id))
            not_found_images.add(image_id)
            continue

        image_features, normalised_boxes = relevant_image_feature_object[
            "features"], relevant_image_feature_object["normalized_boxes_area"]
        len_features = image_features.shape[0]
        if len_features > num_objects:
            image_features = image_features[:num_objects]
            normalised_boxes = normalised_boxes[:num_objects]
            rcnn_features_zeros = image_features
            rcnn_normalised_boxes = normalised_boxes
        else:
            rcnn_features_zeros[:len_features] = image_features
            rcnn_normalised_boxes[:len_features] = normalised_boxes

        d_rcnn_features[q_index] = rcnn_features_zeros
        d_rcnn_locations[q_index] = rcnn_normalised_boxes

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
    parser.add_argument('--image-dir', type=str, default='data/vqa/train2014',
                        help='directory for resized images')
    parser.add_argument('--questions', type=str,
                        default='data/vqa/v2_OpenEnded_mscoco_'
                        'train2014_questions.json',
                        help='Path for train annotation file.')
    parser.add_argument('--annotations', type=str,
                        default='data/vqa/v2_mscoco_'
                        'train2014_annotations.json',
                        help='Path for train annotation file.')
    parser.add_argument('--cat2ans', type=str,
                        default='data/vqa/iq_dataset.json',
                        help='Path for the answer types.')

    # Outputs.
    parser.add_argument('--output', type=str,
                        default='data/processed/iq_dataset.hdf5',
                        help='directory for resized images.')
    parser.add_argument('--cat2name', type=str,
                        default='data/processed/cat2name.json',
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

    save_dataset(args.image_dir, args.questions, args.annotations, ans2cat, args.output,
                 im_size=224, train_or_val=train_or_val, set_k=5)
    print(('Wrote dataset to %s' % args.output))
    # Hack to avoid import errors.
