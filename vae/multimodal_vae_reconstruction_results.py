import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import gensim
import os
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import gensim.downloader
import random

nltk.download('stopwords')
nltk.download('wordnet')

batch_size = 128
learning_rate = 1e-3 # IKEA
num_epochs = 150
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# IKEA
#from google.colab import drive
#drive.mount('/content/drive')
dataset = "/Users/david/Documents/GitHub/DecorAssistant/dataset"

images_folder = os.path.join(dataset, "images/all_items")
image_names = []

for root, dirs, files in os.walk(images_folder, topdown=False):
    for name in files:
        if name[0] != 'i' and name[-5] != ")":
            image_names.append(name)

fn = os.path.join(dataset, "text_data/products_dict.p")
with open(fn, "rb") as f:
		desc = pickle.load(f)
        
descdict = {}
for name in image_names:
    d = desc[name[:-4]]["desc"]
    if d[:4] == 'View':
        d = d[30:]
    descdict[name] = desc[name[:-4]]["type"] + ' ' + d

def preprocess(corpus):
    corpus = [''.join([i.lower() for i in line if i not in string.punctuation])for line in corpus]
    corpus = [line.split(" ") for line in corpus]
    stopwords = nltk.corpus.stopwords.words('english')
    porter_stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    corpus = [[i for i in line if i not in stopwords] for line in corpus]
    return corpus
    
corpus = list(descdict.values())
corpus = preprocess(corpus)
glove = gensim.downloader.load('glove-wiki-gigaword-50')
corpus_vec = []
for line in corpus:
    line_vec = []
    for word in line:
        try:
            line_vec.append(glove[word].tolist())
        except:
            continue
    corpus_vec.append(sum(line_vec, []))
descvecdict = {}
for i,key in enumerate(list(descdict.keys())):
  descvecdict[key] = corpus_vec[i]

maxlen = max(len(x) for x in corpus_vec)

for key in descvecdict:
  descvecdict[key] += [0.0]*(maxlen-len(descvecdict[key]))

class trainset(torch.utils.data.Dataset):
    def __init__(self, image, text, transform):
        self.images = image
        self.texts = text
        self.transform = transform

    def __getitem__(self, index):
        imgidx = self.images[index]
        img = self.transform(Image.open(os.path.join(images_folder, imgidx)))
        #img = torch.Tensor(np.array(Image.open(os.path.join(images_folder, imgidx))))
        text = torch.Tensor(self.texts[imgidx])
        return img, text

    def __len__(self):
        return len(self.images)

transform=transforms.Compose([transforms.Resize(28), transforms.CenterCrop(28), transforms.ToTensor()])
dataset = trainset(image_names, descvecdict, transform)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

class VAE(nn.Module):
  def __init__(self, img_channels=3, feature_dim=32*20*20, z_dim=256, maxlen = maxlen): # IKEA
    super(VAE, self).__init__()
    self.image_encoder_conv1 = nn.Conv2d(img_channels, 16, 5)
    self.image_encoder_conv2 = nn.Conv2d(16, 32, 5) 
    self.image_encoder_fc1 = nn.Linear(feature_dim, z_dim // 2)
    self.image_encoder_fc2 = nn.Linear(feature_dim, z_dim // 2)
    self.text_encoder_fc1 = nn.Linear(maxlen, 512)
    self.text_encoder_fc2 = nn.Linear(512, 512)
    self.text_encoder_fc3 = nn.Linear(512, z_dim // 2)
    self.text_encoder_fc4 = nn.Linear(512, z_dim // 2)
    self.decoder_fc = nn.Linear(z_dim, feature_dim)
    self.decoder_conv1 = nn.ConvTranspose2d(32, 16, 5)
    self.decoder_conv2 = nn.ConvTranspose2d(16, img_channels, 5)
      
  def image_encoder(self, x):
    x = F.relu(self.image_encoder_conv1(x))
    x = F.relu(self.image_encoder_conv2(x))
    x = x.view(-1, 32*20*20)
    mu = self.image_encoder_fc1(x)
    logVar = self.image_encoder_fc2(x)
    return mu, logVar
  
  def text_encoder(self, x):
    x = F.relu(self.text_encoder_fc1(x))
    x = F.relu(self.text_encoder_fc2(x))
    mu = F.relu(self.text_encoder_fc3(x))
    logVar = F.relu(self.text_encoder_fc4(x))
    return mu, logVar

  def reparameterize(self, mu, logVar):
    std = torch.exp(logVar / 2)
    eps = torch.randn_like(std)
    return mu + std * eps, std, eps
  
  def decoder(self, z):
    x = F.relu(self.decoder_fc(z))
    x = x.view(-1, 32, 20, 20)
    x = F.relu(self.decoder_conv1(x))
    x = torch.sigmoid(self.decoder_conv2(x))
    return x
  
  def forward(self, image, text):
    mu1, logVar1 = self.image_encoder(image)
    mu2, logVar2 = self.text_encoder(text)
    mu = torch.cat((mu1, mu2), dim = 1)
    logVar = torch.cat((logVar1, logVar2), dim = 1)
    z, std, eps = self.reparameterize(mu, logVar)
    out = self.decoder(z)
    return out, mu, std, eps, logVar

vae = VAE(maxlen = maxlen).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

# for epoch in range(num_epochs):
#   for idx, data in enumerate(data_loader, 0):
#     imgs, texts = data
#     imgs = imgs.to(device)
#     texts = texts.to(device)
#     print(imgs)
#     # print(texts)
#     out, mu, std, eps, logVar = vae(imgs, texts)
#     kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
#     loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#   print(f'Epoch {epoch}: Loss {loss}')

vae = torch.load('multimodal_vae.pth')
vae.eval()

for i in range(10):
  with torch.no_grad():
    for data in random.sample(list(data_loader), 1):
      imgs, texts = data
      imgs = imgs.to(device)
      texts = texts.to(device)
      img = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
      plt.subplot(121)
      plt.imshow(np.squeeze(img))
      out, mu, std, eps, logVar = vae(imgs, texts)
      outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
      plt.subplot(122)
      plt.imshow(np.squeeze(outimg))
      plt.show()
      print(f'Bottleneck: {mu + std * eps}')
