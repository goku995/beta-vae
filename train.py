import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing
import time, json
import preprocess as prep
import models
import utils
from dataset import *
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence
from datetime import datetime
import matplotlib.pyplot as plt

# parameters
BATCH_SIZE = 256
TEST_BATCH_SIZE = 10
EPOCHS = 400

batch_size = 32
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

now = datetime.now()
writer = SummaryWriter('./runs/vae_{}'.format(now.strftime("%d_%H_%M")))

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def train(model, encoder, decoder, criterion, device, train_loader, optimizer, epoch, log_interval):

    model.train()
    train_loss = 0

    for batch_idx, (imgs, caps, caplens) in enumerate(train_loader):
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        enc_image,  global_features = encoder(imgs)
        orig_preds, _, _, _, decode_lengths, _ = decoder(enc_image, global_features, caps, caplens)

        # imgs = F.interpolate(imgs, size=(64, 64))
        optimizer.zero_grad()

        rec_imgs, mu, logvar = model(imgs)
        r_enc_image,  r_global_features = encoder(rec_imgs)
        rec_preds, _, _, _, _, _ = decoder(r_enc_image, r_global_features, caps, caplens)

        orig_caps = pack_padded_sequence(orig_preds, decode_lengths, batch_first=True)[0]
        rec_caps = pack_padded_sequence(rec_preds, decode_lengths, batch_first=True)[0]

        # print("{}\n, {}".format(orig_caps, rec_caps))
        loss = model.loss(rec_imgs, imgs, mu, logvar) + criterion(rec_caps, orig_caps)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()), epoch, batch_idx,
                len(train_loader), 100. * batch_idx / len(train_loader), loss.item()), flush=True)

    train_loss /= len(train_loader)
    print('Train set Average loss:', train_loss, flush=True)
    return train_loss


def test(model, encoder, decoder, criterion, device, test_loader, epoch, return_images=0, log_interval=None):
    model.eval()
    test_loss = 0

    # two np arrays of images
    original_images = []
    rect_images = []

    with torch.no_grad():
        for batch_idx, (imgs, caps, caplens, _) in enumerate(test_loader):
            imgs = imgs.to(device)
            # imgs = F.interpolate(imgs, size=(64, 64))
            rec_imgs, mu, logvar = model(imgs)
            loss = model.loss(rec_imgs, imgs, mu, logvar)
            test_loss += loss.item()
            
            if epoch % 5 == 0 and batch_idx%100 == 0:
                orig_img = unorm(imgs.squeeze(0))
                orig_img = orig_img.permute(1, 2 , 0).cpu().numpy()

                fig = plt.figure(figsize=(10, 10))

                rows = 1
                cols = 2

                fig.add_subplot(rows, cols, 1)
                plt.imshow(orig_img)
                plt.title("original image {}".format(batch_idx))
                # writer.add_figure("original image", fig, epoch, True)
                # plt.clf()
                
                rec_img = unorm(rec_imgs.squeeze(0))
                rec_img = rec_img.permute(1, 2 , 0).cpu().numpy()

                # fig = plt.figure(figsize=(10, 10))
                
                fig.add_subplot(rows, cols, 2)
                plt.imshow(rec_img)
                plt.title("reconstructed image {}".format(batch_idx))
                
                writer.add_figure("vae image", fig, epoch, True)
                # plt.clf()
                
            
            if return_images > 0 and len(original_images) < return_images:
                original_images.append(imgs[0].cpu())
                rect_images.append(rec_imgs[0].cpu())

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

print('latent size:', LATENT_SIZE)

# model = models.BetaVAE(latent_size=LATENT_SIZE).to(device)
model = models.DFCVAE(latent_size=LATENT_SIZE).to(device)

encoder.eval()
decoder.eval()

criterion = nn.MSELoss().to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if __name__ == "__main__":

    start_epoch = model.load_last_model(MODEL_PATH) + 1
    train_losses, test_losses = utils.read_log(LOG_PATH, ([], []))

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss = train(model, encoder, decoder, criterion, device, train_loader, optimizer, epoch, PRINT_INTERVAL)
        test_loss, original_images, rect_images = test(model, encoder, decoder, criterion, device, val_loader, epoch, return_images=5)

        # save_image(original_images + rect_images, COMPARE_PATH + str(epoch) + '.png', padding=0, nrow=len(original_images))

        # train_losses.append((epoch, train_loss))
        # test_losses.append((epoch, test_loss))
        # utils.write_log(LOG_PATH, (train_losses, test_losses))

        # model.save_model(MODEL_PATH + '%03d.pt' % epoch)
