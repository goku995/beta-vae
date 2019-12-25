import torch
import torch.optim as optim
import multiprocessing
import time, json
import preprocess as prep
import models
import utils
from dataset import *
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as transforms

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    train_loss = 0

    for batch_idx, (imgs, caps, caplens) in enumerate(train_loader):
        imgs = imgs.to(device)
        # imgs = F.interpolate(imgs, size=(64, 64))
        optimizer.zero_grad()
        output, mu, logvar = model(imgs)
        loss = model.loss(output, imgs, mu, logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()), epoch, batch_idx,
                len(train_loader), 100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    print('Train set Average loss:', train_loss)
    return train_loss


def test(model, device, test_loader, return_images=0, log_interval=None):
    model.eval()
    test_loss = 0

    # two np arrays of images
    original_images = []
    rect_images = []

    with torch.no_grad():
        for batch_idx, (imgs, caps, caplens, _) in enumerate(test_loader):
            imgs = imgs.to(device)
            # imgs = F.interpolate(imgs, size=(64, 64))
            output, mu, logvar = model(imgs)
            loss = model.loss(output, imgs, mu, logvar)
            test_loss += loss.item()

            if return_images > 0 and len(original_images) < return_images:
                original_images.append(imgs[0].cpu())
                rect_images.append(output[0].cpu())

            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx, len(test_loader),
                    100. * batch_idx / len(test_loader), loss.item()))

    test_loss /= len(test_loader)
    print('Test set Average loss:', test_loss)

    if return_images > 0:
        return test_loss, original_images, rect_images

    return test_loss


# parameters
BATCH_SIZE = 256
TEST_BATCH_SIZE = 10
EPOCHS = 400

batch_size = 64
workers = 0 

LATENT_SIZE = 100
LEARNING_RATE = 1e-3

USE_CUDA = True
PRINT_INTERVAL = 100
LOG_PATH = './logs/log.pkl'
MODEL_PATH = './checkpoints/'
COMPARE_PATH = './comparisons/'

checkpoint = "checkpoint_32.pth.tar"
data_folder = '../../../caption_dataset/flickr30k_files/'
dataset_name = 'flickr30k_5_cap_per_img_5_min_word_freq'  

annotation_path = "../../../flickr30k_entities/annotation_data.json"
sentence_path = "../../../flickr30k_entities/sentence_data.json"

use_cuda = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)
print('num cpus:', multiprocessing.cpu_count())

unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

with open('{}/WORDMAP_{}.json'.format(data_folder, dataset_name), 'r') as j:
    word_map = json.load(j)

rev_word_map = {v: k for k, v in word_map.items()}

checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder_optimizer = checkpoint['decoder_optimizer']
encoder = checkpoint['encoder']
encoder_optimizer = checkpoint['encoder_optimizer']

decoder = decoder.to(device)
encoder = encoder.to(device)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(CaptionDataset(data_folder, dataset_name, 'TRAIN', annotation_path, sentence_path, transform=transforms.Compose([normalize])), 
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            num_workers=workers, 
                                            pin_memory=True)

val_loader = torch.utils.data.DataLoader(CaptionDataset(data_folder, dataset_name, 'VAL', annotation_path, sentence_path, transform=transforms.Compose([normalize])), 
                                            batch_size=1, 
                                            shuffle=True, 
                                            num_workers=workers, 
                                            pin_memory=True)

# # training code
# train_ids, test_ids = prep.split_dataset()
# print('num train_images:', len(train_ids))
# print('num test_images:', len(test_ids))

# data_train = prep.ImageDiskLoader(train_ids)
# data_test = prep.ImageDiskLoader(test_ids)

# kwargs = {'num_workers': multiprocessing.cpu_count(),
#           'pin_memory': True} if use_cuda else {}

# train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(data_test, batch_size=TEST_BATCH_SIZE, shuffle=True, **kwargs)

print('latent size:', LATENT_SIZE)

# model = models.BetaVAE(latent_size=LATENT_SIZE).to(device)
model = models.DFCVAE(latent_size=LATENT_SIZE).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if __name__ == "__main__":

    start_epoch = model.load_last_model(MODEL_PATH) + 1
    train_losses, test_losses = utils.read_log(LOG_PATH, ([], []))

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch, PRINT_INTERVAL)
        test_loss, original_images, rect_images = test(model, device, val_loader, return_images=5)

        save_image(original_images + rect_images, COMPARE_PATH + str(epoch) + '.png', padding=0, nrow=len(original_images))

        train_losses.append((epoch, train_loss))
        test_losses.append((epoch, test_loss))
        utils.write_log(LOG_PATH, (train_losses, test_losses))

        model.save_model(MODEL_PATH + '%03d.pt' % epoch)
