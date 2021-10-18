# %%
import argparse
import os
from PIL import Image
import cv2
from detectron2.config.config import get_cfg
import torch
from torch import nn
from torchvision import models, transforms
from main import TrainVQG
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224,
                                 scale=(1.00, 1.2),
                                 ratio=(0.75, 1.3333333333333333)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])


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


def load_image(image_path):
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    object_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes

    cnn = EncoderCNN().to(device)

    image_for_cnn = Image.open(image_path).convert("RGB")
    image_for_cnn = transform(image_for_cnn)
    image_features = cnn(image_for_cnn.unsqueeze(0).to(device))

    image_for_predictor = cv2.imread(image_path)
    object_detectection = predictor(image_for_predictor)
    object_label_class_idxs = object_detectection["instances"].pred_classes
    object_labels = list(set([object_classes[item] for item in object_label_class_idxs]))
    split_object_labels = []
    for label in object_labels:
        # a label can be multiple words (e.g. sports ball). We want to have these as individal words (e.g. ["sports", "ball"])
        split_label = label.split()
        split_object_labels.extend(split_label)
    split_object_labels = list(set(split_object_labels))

    return image_features, " ".join(split_object_labels)


def inference(trainer: TrainVQG, image_features, object_labels):
    model.eval()

    encoded_inputs = trainer.tokenizer(object_labels)
    encoded_input_id = torch.tensor([encoded_inputs["input_ids"]])
    encoded_input_attention_mask = torch.tensor([encoded_inputs["attention_mask"]])
    image_features = image_features.unsqueeze(0)

    decoded_inputs = [trainer.tokenizer.decode(to_decode) for to_decode in encoded_input_id][0]
    decoded_sentence = trainer.model.decode_greedy(image_features, encoded_input_id, encoded_input_attention_mask)[0]
    curr_input = trainer.filter_special_tokens(decoded_inputs)
    generated_q = trainer.filter_special_tokens(decoded_sentence)
    print("Category:\t", curr_input.split()[0])
    print("KW inputs:\t", " ".join(curr_input.split()[1:]))
    print("Generated:\t", generated_q)

    return generated_q


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_rootdir", type=str, default="lightning_logs",
                        help="Root location of model checkpoints")
    parser.add_argument("--model_v_num", type=int, default=0,
                        help="model version number to load in")
    parser.add_argument("--image_path", type=str, default="/data/lama/mscoco/images/val2014/COCO_val2014_000000213224.jpg", help="path to load image to do inference against")

    args = parser.parse_args()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    args.device = device

    model_path_dir = os.path.join(args.model_rootdir, "version_{}".format(args.model_v_num), "checkpoints")

    model_ckpt = None
    for file in os.listdir(model_path_dir):
        if file.endswith(".ckpt"):
            model_ckpt = file  # returns the latest checkpoint in case of multiple files

    image_features, object_labels = load_image(args.image_path)
    model = TrainVQG.load_from_checkpoint(os.path.join(model_path_dir, model_ckpt)).to(device)
    model.args.device = device
    inference(model, image_features, object_labels)
