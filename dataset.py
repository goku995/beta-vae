import os
import re
import h5py
import json
import torch
from torch.utils.data import Dataset
import torch.optim
import torch.utils.data

class CaptionDataset(Dataset):

    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, annotation_path, sentence_path, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """

        self.split = split

        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(
            data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')

        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # for flickr30k_entities sentences and bounding box annotations
        with open(os.path.join(data_folder, self.split + '_ANNOTATIONS_' + data_name + '.json'), 'r') as j:
            self.annotations = json.load(j)

        # load word map from a JSON
        with open(os.path.join(data_folder, 'WORDMAP_' + data_name + '.json'), 'r') as j:
            self.word_map = json.load(j)

        self.rev_word_map = {v: k for k, v in self.word_map.items()}

        # bounding box annotations
        with open(annotation_path, 'r') as j:
            self.annotation_data = json.load(j)

        # sentence phrases
        with open(sentence_path, 'r') as j:
            self.sentence_data = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

        # Maximum sentence length
        self.max_len = 50

    def __getitem__(self, i):

        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)

        if self.transform is not None:
            img = self.transform(img)

        file_key = self.annotations[i // self.cpi]
        annotation = self.annotation_data[file_key]
        # print("Annotations", annotation, flush=True)

        sentences = self.sentence_data[file_key]
        sentence_map = sentences[i % self.cpi]
        # print(sentence_map)

        sentence = sentence_map['sentence'].lower()
        sentence = re.sub(r'[^\w\s]', '', sentence)
        sentence = sentence.split()

        # for sent in sentences:
        #    print(sent['sentence'])

        # print("caption_data ======== ", sentence_map['sentence'], flush=True)
        sentence_encoding = torch.LongTensor([self.word_map['<start>']] + [self.word_map.get(word, self.word_map['<unk>'])
                                            for word in sentence] + [self.word_map['<end>']] + [self.word_map['<pad>']] * (self.max_len - len(sentence)))

        caption = torch.LongTensor(self.captions[i])
        caption_idx = [w.item() for w in caption if w.item() not in {
            self.word_map['<start>'], self.word_map['<end>'], self.word_map['<pad>']}]
        caption_sentence = ' '.join(
            self.rev_word_map[idx] for idx in caption_idx)

        # print("caption_data ======== ", caption_sentence, flush=True)
        # print(sentence_encoding, caption)

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
