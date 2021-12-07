# Please run these two commands first!
# ! pip install ftfy regex tqdm
# ! pip install git+https://github.com/openai/CLIP.git

import cv2
import numpy as np
import pickle
import os
import itertools
import re
import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from gensim.models import KeyedVectors
import random
from sklearn.model_selection import train_test_split
# random.seed(517)

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


# Global Path Vairables
ROOT_DIR =  "drive/MyDrive/DecorAssist/IKEA/" # modify this
TEXT_DIR = ROOT_DIR + "text_data/"
ITEMS_DIR = ROOT_DIR + "images/all_items/"
ROOMS_DIR = ROOT_DIR + "images/room_scenes/"

# Global Parameter Variables
MAX_SEQUENCE_LENGTH = 100
NUM_WORDS_TOKENIZER = 50000
EMBEDDING_DIM = 300
BATCH_SIZE = 32
POSITIVE_SIZE = 1000 # We might only use a subset of the positive pairs
TRAIN_TEST_RATIO = 0.33

# Model Hyperparameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.001 # 0.001
HIDDEN_DIM = 64 # 64
N_LAYERS = 8 # 2
EPOCHS = 5
CLIP = 5
DROPOUT = 0.5
TRAIN_WITH_ROOM_IMAGES = True


########################################################################
# Clip Model Configuration
########################################################################

import clip
# clip.available_models()

model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size


# CLIP has some layers explicitly parameterized using fp16 values. We need to
# convert them back to fp32 in order to use automatic mixed-precision training
def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp32"""

    def _convert_weights_to_fp32(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp32)
convert_weights(model)


########################################################################
# Helper Functions
########################################################################
def preprocess_img(path):
  img = cv2.imread(path)
  try:
    img = cv2.resize(img, (224, 224))
  except:
    print(path)
  img = img.astype(np.float32) / 255
  img = np.reshape(img, (3, 224 ,224))
  return img


def read_pickle(fn):
	with open(fn, "rb") as f:
		return pickle.load(f)


def clip_tokenize(texts):
  truncated_texts = [' '.join(text.split()[:50]) for text in texts]
  return clip.tokenize(truncated_texts)


# Train-val split that does not share products between training and validation sets.
def generate_product_limited_samples(products, all_positive_pairs, random_state=None):
    """
    Generates positive and negative examples for the given products using shared
    occurence in rooms to indicate whether two products are compatible.

    products: A sequence of product IDs; ALL positive and negative pairs must
        contain only these product IDs.
    all_positive_pairs: A set of product ID pairs that are positive.

    Returns: A tuple (x, y), where x is a sequence of product ID pairs and y is
        the array of [0 or 1] labels indicating presence in all_positive_pairs.
    """
    product_set = set(products)
    within_positive_pairs = [p for p in sorted(all_positive_pairs) if p[0] in product_set and p[1] in product_set]
    negative_pairs = random_negative_sampling(products, all_positive_pairs, count=len(within_positive_pairs), random_state=random_state)
    x = within_positive_pairs + negative_pairs
    y = np.array([1] * len(within_positive_pairs) + [0] * len(negative_pairs))
    if random_state is not None: np.random.seed(random_state)
    indices = np.random.permutation(np.arange(len(x)))
    return [x[i] for i in indices], y[indices]


def random_negative_sampling(products, all_positive_pairs, count=None, random_state=None):
  selected_negative_pairs = []
  if random_state is not None: random.seed(random_state)
  while len(selected_negative_pairs) < (count or len(all_positive_pairs)):
    random_pair = tuple(random.sample(products, 2))
    if random_pair in all_positive_pairs:
      continue
    else:
      selected_negative_pairs.append(random_pair)
  return selected_negative_pairs


def preprocess_info(properties):
    base = properties["type"] + " " + properties["desc"]
    base = base.replace("View more product information", "")
    base = re.sub(product_names, '', base)
    return re.sub(r'\s+', ' ', base)


class CoLearnFurnitureImagePairsDataset(Dataset):
    """Dataset containing pairs of furniture items."""

    def __init__(self, pairs, room_ids, labels, item_path=ITEMS_DIR, room_path=ROOMS_DIR):
        """
        Args:
            image_path (string): Path to the directory containing images.
            pairs (list of tuples of strings): Pairs of image IDs to be used as training samples.
            labels (array of integers): Labels for the training samples.
        """
        super(CoLearnFurnitureImagePairsDataset, self).__init__()
        self.image_ids = list(set(x for pair in pairs for x in pair))
        self.room_ids = room_ids
        self.index_mapping = {image_id: i for i, image_id in enumerate(self.image_ids)}
        self.item_images = [preprocess_img(item_path + image_id + ".jpg") for image_id in tqdm.tqdm_notebook(self.image_ids)]
        self.room_images = [preprocess_img(room_path + room_id + ".jpg") for room_id in tqdm.tqdm_notebook(self.room_ids)]
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
        idx = idx.tolist()

      if isinstance(idx, (list, tuple)):
        x1, x2, x3, y = zip(*[self[i] for i in idx])
        return torch.stack(x1), torch.stack(x2), torch.stack(x3), torch.from_numpy(np.array(y))

      pair = self.pairs[idx]
      return self.item_images[self.index_mapping[pair[0]]],\
              self.item_images[self.index_mapping[pair[1]]],\
              self.room_images[idx],\
              self.labels[idx]


########################################################################
# Load Venkat's data
########################################################################
# {room image url -> string of room category}; e.g.: 'ikea-town-and-country__1364308377063-s4.jpg': 'Living Room'
room_categories = read_pickle(TEXT_DIR + "categories_dict.p")
# {item image ID -> string of item category}; e.g.: '291.292.29': 'Footstool',
item_categories = read_pickle(TEXT_DIR + "categories_images_dict.p")
# {item image id -> dict of descriptions}; e.g. '202.049.06': {'color': 'Grey,black','desc': 'View more product information Concealed press studs keep the quilt in place','img': 'images/objects/202.049.06.jpg','name': 'GURLI','size': '120x180 cm','type': 'Throw'},
item_property = read_pickle(TEXT_DIR + "products_dict.p")
# {item image url -> {description, name}}; e.g: '/static/images/902.592.50.jpg': {'desc': 'The high pile dampens sound and provides a soft surface to walk on.','name': 'GSER'},
item_to_description = read_pickle(TEXT_DIR + "img_to_desc.p")
# {item image url -> list of corresponding room image url}; e.g.: 'images/001.509.85.jpg': ['images/room_scenes/ikea-wake-up-and-grow__1364335362013-s4.jpg','images/room_scenes/ikea-wake-up-and-grow-1364335370196.jpg'],
item_to_rooms_map = read_pickle(TEXT_DIR + "item_to_room.p")
item_to_rooms_map = {item_url.split("/")[-1].split(".jpg")[0] : val for item_url, val in item_to_rooms_map.items()}
# {room image url -> list of items}; e.g.: 'ikea-work-from-home-in-perfect-harmony__1364319311386-s4.jpg': ['desk','chair']
room_to_item_categories = read_pickle(TEXT_DIR + "room_to_items.p")

room_to_items = {}
unavailable_scenes = set()
for item_id, room_url_list in item_to_rooms_map.items():
    if not os.path.exists(ITEMS_DIR + item_id + ".jpg"):
        print(ITEMS_DIR + item_id + ".jpg" + " does not exist")
        continue

    for room_url in room_url_list:
        room_id = room_url.split("/")[-1].split(".jpg")[0]
        if not os.path.exists(ROOT_DIR + "images/room_scenes/" + room_url):
            if room_id not in unavailable_scenes:
                # print(room_url + " does not exist")
                unavailable_scenes.add(room_id)

        if room_id not in room_to_items:
            room_to_items[room_id] = [item_id]
        else:
            room_to_items[room_id].append(item_id)

product_names = "(" + "|".join(list(set([value["name"] for value in item_property.values()]))) + ")"
item_to_info = {key: preprocess_info(value) for key, value in item_property.items()} # remove view more info

with open(ROOT_DIR + "train_sets_reweighted.pkl", "rb") as f:
  train_sets = pickle.load(f)

with open(ROOT_DIR + "val_data_reweighted.pkl", "rb") as f:
  val_set = pickle.load(f)

# add empty image
for i in range(len(train_sets)):
  train_sets[i] = (train_sets[i][0], train_sets[i][1], ['empty' if v is None else v for v in train_sets[i][2]])

# The reason for using an if-else here is that
# we want the model to get rid of the reliance on room images in later epochs
train_pairs = train_sets[0][0]
y_train = train_sets[0][1]

if TRAIN_WITH_ROOM_IMAGES:
  train_room_ids = train_sets[0][2]
else:
  train_room_ids = ["empty" for _ in range(len(train_sets[0][2] ))]

val_pairs = val_set[0]
y_val = val_set[1]


########################################################################
# Modeling
########################################################################

class CLIPIKEA_Colearn(nn.Module):
    def __init__(self, clip_model, embedding_dim, n_out):
        super(CLIPIKEA_Colearn, self).__init__()

        self.clip_model = clip_model
        self.combined_fc1 = nn.Linear(embedding_dim * 5, 256)
        self.output_fc = nn.Linear(256, n_out)


    def embedder(self, txt, img):
      with autocast(enabled=False):
        txt_emb = self.clip_model.encode_text(txt)
        img_emb = self.clip_model.encode_image(img)
      return txt_emb, img_emb


    def classifier(self, txt_emb_1, txt_emb_2, img_emb_1, img_emb_2, img_emb_3):
      all_emb = torch.cat((txt_emb_1, txt_emb_2, img_emb_1, img_emb_2, img_emb_3), 1)
      x_comb = F.relu(self.combined_fc1(all_emb))
      out = self.output_fc(x_comb)
      return out


    def forward(self, txt_1, txt_2, img_1, img_2, img_3):
      batch_size = txt_1.size(0)

      txt_emb_1, img_emb_1 = self.embedder(txt_1, img_1)
      txt_emb_2, img_emb_2 = self.embedder(txt_2, img_2)
      img_emb_3 = self.clip_model.encode_image(img_3)

      out = self.classifier(txt_emb_1, txt_emb_2, img_emb_1, img_emb_2, img_emb_3)
      return out

output_size = 1 # only output a single sigmoid value # y.shape[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

full_model = CLIPIKEA_Colearn(model, 512, output_size)
full_model.to(device)
full_model.load_state_dict(torch.load(ROOT_DIR + "co-learn.p")) # https://drive.google.com/file/d/1qlDZqN3GRVMfFjS9K-66VkcN-eJSkvH9/view?usp=sharing



########################################################################
# Evaluation
########################################################################
val_image_ids = sorted(list(set([x for pair in val_pairs for x in pair])))
ranking_pairs = list(itertools.combinations(val_image_ids, 2)) # n choose 2

X_rank_image = CoLearnFurnitureImagePairsDataset(pairs=ranking_pairs,
                                         room_ids=["empty" for _ in range(len(ranking_pairs))],
                                         labels=np.zeros(len(ranking_pairs)))

X_rank_text_premise = [item_to_info[id] for id, _ in ranking_pairs]
X_rank_text_hypothesis = [item_to_info[id] for _, id in ranking_pairs]
X_rank_text_premise = clip_tokenize(X_rank_text_premise)
X_rank_text_hypothesis = clip_tokenize(X_rank_text_hypothesis)

img_ranking_data = X_rank_image
text_ranking_data = TensorDataset(X_rank_text_premise, X_rank_text_hypothesis, torch.zeros(len(ranking_pairs)))

text_ranking_loader = DataLoader(text_ranking_data, batch_size=BATCH_SIZE)
img_ranking_loader = DataLoader(img_ranking_data, batch_size=BATCH_SIZE)
full_model.eval()
ranking_results = []

with torch.no_grad():
    for lstm, cnn in tqdm.tqdm_notebook(zip(text_ranking_loader, img_ranking_loader), total=len(text_ranking_loader)):
      lstm_inp1, lstm_inp2, _ = lstm
      cnn_inp1, cnn_inp2, cnn_inp3, _ = cnn
      lstm_inp1, lstm_inp2 = lstm_inp1.to(DEVICE), lstm_inp2.to(DEVICE)
      cnn_inp1, cnn_inp2, cnn_inp3 = cnn_inp1.to(DEVICE), cnn_inp2.to(DEVICE), cnn_inp3.to(DEVICE)
      model.zero_grad()
      output = full_model(lstm_inp1, lstm_inp2, cnn_inp1, cnn_inp2, cnn_inp3)
      score = output.squeeze().cpu().detach().numpy().tolist()
      ranking_results.append(score)
ranking_results = np.concatenate(ranking_results).reshape(len(val_image_ids), len(val_image_ids))
print(ranking_results.shape)
