# -*- coding: utf-8 -*-
import os
import sys
import glob
import pickle

import numpy as np
from PIL import Image
import torch
from torch.utils import data

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
import utils

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

import multiprocessing


# ----------------------------------------
# Functions for IIT-V2C Database Integration
# ----------------------------------------

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations.
    NAME = None  # Override in sub-classes
    ROOT_DIR = None  # Root project directory

    # --------------------
    # Training Parameters
    # Learning rate

    # Saved model path
    CHECKPOINT_PATH = os.path.join('checkpoints')

    # --------------------
    # Size for Vocabulary
    VOCAB_SIZE = None

    # Size for video observation window
    WINDOW_SIZE = 30

    # --------------------
    # Parameters for dataset, tf.dataset configuration
    # Path to currently used dataset
    DATASET_PATH = os.path.join('datasets')  # Override in sub-classes

    # Maximum command sentence length
    MAXLEN = 10

    # Buffer size for tf.dataset shuffling
    BUFFER_SIZE = 1000

    # --------------------
    # Parameters for NLP parsing settings
    # Word frequency
    FREQUENCY = None

    # Whether to use bias vector, which is related to the log
    # probability of the distribution of the labels (words) and how often they occur.
    USE_BIAS_VECTOR = True

    # Special tokens to be added into vocabulary
    START_WORD = '<sos>'
    END_WORD = '<eos>'
    UNK_WORD = None

    def __init__(self):
        """Set values of computed attributes."""
        # Workers used for dataset object...
        if os.name is 'nt':
            self.WORKERS = 0
        else:
            self.WORKERS = multiprocessing.cpu_count()

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        print("-" * 30)
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print()


# Configuration for hperparameters
class TrainConfig(Config):
    """Configuration for training with IIT-V2C.
    """
    NAME = 'v2c_IIT-V2C'
    ROOT_DIR = ROOT_DIR
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints')
    # DATASET_PATH = os.path.join(ROOT_DIR, 'datasets', 'IIT-V2C')
    DATASET_PATH = 'H:\\IIT-V2C-image'
    # DATASET_PATH = 'E:\\dataset\\IIT-V2C-image'
    MAXLEN = 10


def load_annotations(dataset_path=os.path.join('datasets', 'IIT-V2C'),
                     annotation_file='train.txt'):
    """Helper function to parse IIT-V2C dataset.
    """

    def get_frames_no(init_frame_no, end_frame_no):
        frames = []
        for i in range(init_frame_no, end_frame_no + 1, 1):
            frames.append(i)
        return frames

    # Read annotations
    v2c_json = []
    annotations = {}
    with open(os.path.join(dataset_path, annotation_file), 'r', encoding='utf-8') as f:
        i = 0
        annotation = []
        for line in f:
            v2c_dict = {}
            v2c_dict['videoID'] = ''
            v2c_dict['enCap'] = []
            v2c_dict['chCap'] = []
            v2c_dict['action'] = []

            line = line.strip()
            i += 1
            annotation.append(line)

            if i % 6 == 0:
                # Test cases
                # print(annotation)
                # assert annotation[-1] == ''
                # assert len(annotation[1].split(' ')) == 2
                # Collect Video Name, Annotated Sub-Video Clip id
                video_fname, video_id = '_'.join(annotation[0].split('_')[:-1]), annotation[0].split('_')[-1]

                # Collect init frame no ~ end frame no 
                # Populate frames and commands
                init_frame_no, end_frame_no = int(annotation[1].split(' ')[0]), int(annotation[1].split(' ')[1])
                frames = get_frames_no(init_frame_no, end_frame_no)
                command = annotation[2].strip().split(' ')
                videoID = annotation[0].strip()
                command_en = annotation[2].strip()
                command_ch = annotation[3].strip()
                command_action = annotation[4].strip()
                if command_en == '' or command_ch == '' or command_action == '':
                    print(annotation[0].strip())
                    print('标签出错！！！')
                    while 1:
                        pass

                if video_fname not in annotations:
                    annotations[video_fname] = [[video_id, frames, command]]
                else:
                    annotations[video_fname].append([video_id, frames, command])

                v2c_dict['videoID'] = videoID
                v2c_dict['enCap'].append(command_en)
                v2c_dict['chCap'].append(command_ch)
                v2c_dict['action'].append(command_action)
                v2c_json.append(v2c_dict)

                annotation = []

    return annotations, v2c_json


def summary(annotations):
    """Helper function for IIT-V2C dataset summary.
    """
    num_clips = 0
    num_frames = 0
    for video_path in annotations.keys():
        annotations_by_video = annotations[video_path]
        num_clips += len(annotations_by_video)
        for annotation in annotations_by_video:
            num_frames += len(annotation[1])
    print('# videos in total:', len(annotations))
    print('# sub-video clips in annotation file:', num_clips)
    print('# frames in total:', num_frames)


def clipsname_captions(annotations,
                       padding_words=True):
    """Get (clip_name, target) pairs from annotation.
    """
    # Parse all (inputs, captions) pair
    clips_name, captions = [], []
    for video_fname in annotations.keys():
        annotations_by_clip = annotations[video_fname]
        for annotation in annotations_by_clip:
            # Clip name
            clip_name = video_fname + '_' + annotation[0]

            # Get command caption
            if padding_words:
                target = '<sos> ' + ' '.join(annotation[2]) + ' <eos>'
            else:
                target = ' '.join(annotation[2])

            captions.append(target)
            clips_name.append(clip_name)

    return clips_name, captions


# ----------------------------------------
# For Custom Feature Extraction
# ----------------------------------------

def avi2frames(dataset_path,
               in_folder='avi_video',
               out_folder='images'):
    """Convert avi videos from IIT-V2C dataset into images.
    WARNING: SLOW + TIME CONSUMING process! Final images take LARGE DISK SPACE!
    """
    import cv2
    videos_path = glob.glob(os.path.join(dataset_path, in_folder, '*.avi'))
    for video_path in videos_path:
        video_fname = video_path.strip().split('\\')[-1][:-4]
        save_path = os.path.join(dataset_path, out_folder, video_fname)
        print('Saving:', save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # OpenCV video feed
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        count = 0  # Start frame index as 0, as stated by the author
        cap.set(cv2.CAP_PROP_FPS, 15)  # Force fps into 15, as stated by the author
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

        while success:
            cv2.imwrite(os.path.join(save_path, 'frame%d.png' % count), frame)  # Save into loseless *.png format
            count += 1
            success, frame = cap.read()
    return True


def imgspath_targets_v1(annotations,
                        max_frames=30,
                        dataset_path=os.path.join('datasets', 'IIT-V2C'),
                        folder='images',
                        synthetic_frame_path=os.path.join('imagenet_frame.png'),
                        padding_words=True):
    """Get training/test image-command pairs.
    Version v2 strategy: Same as original IIT-V2C strategy, have a number 
    of max_frames images per sample. Cut images larger than max_frames, populate
    sample if no. images is smaller than max_frames.
    """

    def get_frames_path(frames_no,
                        video_fname,
                        max_frames=30,
                        dataset_path=os.path.join('datasets', 'IIT-V2C'),
                        folder='images',
                        synthetic_frame_path=os.path.join('imagenet_frame.png')):
        """Helper func to parse image path from numbers.
        """
        # Cut additional images by getting min loop factor
        num_frames = len(frames_no)
        loop_factor = min(num_frames, max_frames)
        imgs_path = []

        # -----------------------------
        if num_frames >= max_frames:
            frames_sampling = np.linspace(float(frames_no[0]), float(frames_no[-1]), max_frames)  # 从 - pi到pi取100个值
            for i in range(max_frames):
                frames_sampling_int = round(frames_sampling[i])
                img_path = os.path.join(dataset_path, folder, video_fname, 'frame{}.png'.format(frames_sampling_int))
                if os.path.isfile(img_path):  # Check if frame exists
                    imgs_path.append(img_path)

        # -----------------------------
        else:
            for i in range(loop_factor):
                img_path = os.path.join(dataset_path, folder, video_fname, 'frame{}.png'.format(frames_no[i]))
                if os.path.isfile(img_path):  # Check if frame exists
                    imgs_path.append(img_path)

        # Add synthetically made imagenet frame
        while len(imgs_path) < max_frames:
            imgs_path.append(synthetic_frame_path)

        if len(imgs_path) < max_frames:
            print('出现帧数小于30的情况. len(imgs_path) =', len(imgs_path))
            print('imgs_path =', imgs_path)
            while True:
                pass

        return imgs_path

    # Parse all (inputs, targets) pair
    inputs, targets = [], []
    for video_fname in annotations.keys():
        annotations_by_clip = annotations[video_fname]
        for annotation in annotations_by_clip:
            # Clip name
            clip_name = video_fname + '_' + annotation[0]

            # Get all images of the current clip
            frames_path = get_frames_path(annotation[1],
                                          video_fname,
                                          max_frames,
                                          dataset_path,
                                          folder,
                                          synthetic_frame_path)

            # Get command caption
            if padding_words:
                target = '<sos> ' + ' '.join(annotation[2]) + ' <eos>'
            else:
                target = ' '.join(annotation[2])

            inputs.append({clip_name: frames_path})
            targets.append(target)

    return inputs, targets


# ----------------------------------------
# Functions for torch.data.Dataset
# ----------------------------------------

def parse_dataset(config,
                  annotation_file,
                  vocab=None,
                  numpy_features=True):
    """Parse IIT-V2C dataset and update configuration.
    """

    # Load annotation 1st
    annotations, v2c_json = load_annotations(config.DATASET_PATH, annotation_file)

    # Pre-extracted features saved as numpy
    if numpy_features:
        clips, captions = clipsname_captions(annotations)
        # clips = [os.path.join(config.DATASET_PATH, list(config.BACKBONE.keys())[0], x + '.npy') for x in clips]

        clips = [os.path.join(config.DATASET_PATH, 'resnet50_keras_feature_no_sub_mean', x + '.npy') for x in clips]
        # clips = [os.path.join(config.DATASET_PATH, 'resnet50-uniform-sampling', x + '.npy') for x in clips]
        # clips = [os.path.join(config.DATASET_PATH, 'resnet152', x + '.npy') for x in clips]

    # Use images
    else:
        clips, captions = imgspath_targets_v1(annotations,
                                              max_frames=config.WINDOW_SIZE,
                                              dataset_path=config.DATASET_PATH,
                                              folder='images',
                                              synthetic_frame_path=os.path.join(config.ROOT_DIR, 'datasets',
                                                                                'imagenet_frame.png')
                                              )

    # Build vocabulary
    if vocab is None:
        vocab = utils.build_vocab(captions,
                                  frequency=config.FREQUENCY,
                                  start_word=config.START_WORD,
                                  end_word=config.END_WORD,
                                  unk_word=config.UNK_WORD)
    # Reset vocab_size
    config.VOCAB_SIZE = len(vocab)  # 一共 133 个单词，包含<sos> <eos>

    if config.MAXLEN is None:  # Maximum command sentence length
        config.MAXLEN = utils.get_maxlen(captions)

        # Process text tokens
    targets = utils.texts_to_sequences(captions, vocab)  # 将单词转换为词表中的索引
    targets = utils.pad_sequences(targets, config.MAXLEN, padding='post')
    targets = targets.astype(np.int64)

    # ----------------------------------------------------
    # embedding_type = "word2vec"
    # embedding_type = "glove"
    # if embedding_type == "word2vec":
    #     embedding_weights = get_word2vec(vocab)
    # elif embedding_type == "glove":
    #     embedding_weights = get_glove_embedding(vocab)
    # else:
    #     pass

    # ----------------------------------------------------

    return clips, targets, vocab, config


class FeatureDataset(data.Dataset):
    """Create an instance of IIT-V2C dataset with (features, targets) pre-extracted,
    or with (imgs_path, targets)
    """

    def __init__(self,
                 inputs,
                 targets,
                 numpy_features=True,
                 transform=None):
        self.inputs, self.targets = inputs, targets  # Load annotations
        self.numpy_features = numpy_features
        self.transform = transform

    def parse_clip(self,
                   clip):
        """Helper function to parse images {clip_name: imgs_path} into a clip. 
        """
        Xv = []
        clip_name = list(clip.keys())[0]
        imgs_path = clip[clip_name]
        for img_path in imgs_path:
            img = self._imread(img_path)
            Xv.append(img)
        Xv = torch.stack(Xv, dim=0)
        return Xv, clip_name

    def _imread(self, path):
        """Helper function to read image.
        """
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Pre-extracted numpy features
        if self.numpy_features:
            Xv = np.load(self.inputs[idx])
            clip_name = self.inputs[idx].split('/')[-1]
        # Image dataset
        else:
            Xv, clip_name = self.parse_clip(self.inputs[idx])
        S = self.targets[idx]
        return Xv, S, clip_name


class FeatureDataset_3D(data.Dataset):
    """Create an instance of IIT-V2C dataset with (features, targets) pre-extracted,
    or with (imgs_path, targets)
    """

    def __init__(self,
                 inputs,
                 targets,
                 numpy_features=True,
                 transform=None):
        self.inputs, self.targets = inputs, targets  # Load annotations
        self.numpy_features = numpy_features
        self.transform = transform

    def parse_clip(self,
                   clip):
        """Helper function to parse images {clip_name: imgs_path} into a clip.
        """
        Xv = []
        clip_name = list(clip.keys())[0]
        imgs_path = clip[clip_name]
        for img_path in imgs_path:
            img = self._imread(img_path)
            Xv.append(img)

        Xv1 = Xv[0:16]
        Xv2 = Xv[14:]

        Xv1 = torch.stack(Xv1, dim=0).unsqueeze(0)  # shape = [1, 16, 3, 112, 112]
        Xv2 = torch.stack(Xv2, dim=0).unsqueeze(0)

        Xv1 = Xv1.permute(0, 2, 1, 3, 4).contiguous()  # shape = [1, 3, 16, 112, 112]
        Xv2 = Xv2.permute(0, 2, 1, 3, 4).contiguous()

        Xv12 = torch.cat([Xv1, Xv2], dim=0)

        return Xv12, clip_name

    def _imread(self, path):
        """Helper function to read image.
        """
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Pre-extracted numpy features
        if self.numpy_features:
            Xv = np.load(self.inputs[idx])
            clip_name = self.inputs[idx].split('/')[-1]
        # Image dataset
        else:
            Xv, clip_name = self.parse_clip(self.inputs[idx])
        S = self.targets[idx]
        return Xv, S, clip_name


# ----- 构建词向量 -------------
def get_glove_embedding(vocab):
    '''
    生成glove词向量
    :return: 根据词表生成词向量
    '''

    path = os.path.abspath('../../datasets')

    if not os.path.exists(path + "/glove_embedding_mr.npy"):  # 如果已经保存了词向量，就直接读取
        if not os.path.exists(path + "/glove_word2vec.txt"):
            glove_file = datapath(path + '/glove.840B.300d.txt')
            # 指定转化为word2vec格式后文件的位置
            tmp_file = get_tmpfile(path + "/glove_word2vec.txt")
            from gensim.scripts.glove2word2vec import glove2word2vec
            glove2word2vec(glove_file, tmp_file)
        else:
            tmp_file = get_tmpfile(path + "/glove_word2vec.txt")
        print("Reading Glove Embedding...")
        wvmodel = KeyedVectors.load_word2vec_format(tmp_file)
        tmp = []
        for word, index in vocab.word2idx.items():
            try:
                tmp.append(wvmodel.get_vector(word))
            except:
                pass
        mean = np.mean(np.array(tmp))
        std = np.std(np.array(tmp))
        print(mean, std)
        vocab_size = len(vocab)  # self.n_vocab
        embed_size = 300
        embedding_weights = np.random.normal(mean, std, [vocab_size, embed_size])  # 正太分布初始化方法
        for word, index in vocab.word2idx.items():
            try:
                embedding_weights[index, :] = wvmodel.get_vector(word)
            except:
                pass
        np.save(path + "/glove_embedding_mr.npy", embedding_weights)  # 保存生成的词向量
    else:
        embedding_weights = np.load(path + "/glove_embedding_mr.npy")  # 载入生成的词向量

    return embedding_weights


def get_word2vec(vocab):
    '''
    生成word2vec词向量
    :return: 根据词表生成的词向量
    '''

    path = os.path.abspath('../../datasets')

    if not os.path.exists(path + "/word2vec_embedding_mr.npy"):  # 如果已经保存了词向量，就直接读取
        print("Reading word2vec Embedding...")
        wvmodel = KeyedVectors.load_word2vec_format(path + "/GoogleNews-vectors-negative300.bin.gz", binary=True)
        tmp = []
        for word, index in vocab.word2idx.items():
            try:
                tmp.append(wvmodel.get_vector(word))
            except:
                pass
        mean = np.mean(np.array(tmp))
        std = np.std(np.array(tmp))
        print(mean, std)
        vocab_size = len(vocab)  # self.n_vocab
        embed_size = 300
        embedding_weights = np.random.normal(mean, std, [vocab_size, embed_size])  # 正太分布初始化方法
        for word, index in vocab.word2idx.items():
            try:
                embedding_weights[index, :] = wvmodel.get_vector(word)
            except:
                pass
        np.save(path + "/word2vec_embedding_mr.npy", embedding_weights)  # 保存生成的词向量
    else:
        embedding_weights = np.load(path + "/word2vec_embedding_mr.npy")  # 载入生成的词向量

    return embedding_weights


# ----- 构建词向量 -------------

# ----- 生成包含中文的 json 文件----.
def generate_json_v2c():
    config = TrainConfig()
    # annotation_file = 'train-en-ch.txt'
    annotation_file = 'test-en-ch.txt'
    annotation_file_name = annotation_file.strip().split('.')[0]

    annotations, v2c_json = load_annotations(config.DATASET_PATH, annotation_file)
    print('Lenth-v2c_json', len(v2c_json))
    # save_path = ROOT_DIR

    import json
    filename = annotation_file_name + '.json'
    with open(filename, 'w', encoding='utf-8') as file_obj:
        json.dump(v2c_json, file_obj)
    print('保存：'+filename)

    # clips, targets, vocab, config = parse_dataset(config, annotation_file)
    # config.display()
    # train_dataset = FeatureDataset(clips, targets)
    # train_loader = data.DataLoader(train_dataset,
    #                                batch_size=16,
    #                                shuffle=True,
    #                                num_workers=0)
    # bias_vector = vocab.get_bias_vector() if config.USE_BIAS_VECTOR else None

    pass


# ----- 生成包含中文的 json 文件----

if __name__ == '__main__':
    pass
    generate_json_v2c()

# --------------------------------------------------------------------------------------------------
# file1 = open('glove_word2vec-init.txt', 'r', encoding='utf-8')  # 打开要去掉空行的文件
# file2 = open('glove_word2vec-2.txt', 'w', encoding='utf-8')  # 生成没有空行的文件
# i = 0
# s = 0
# for line in file1.readlines():  # 2196017 300\n
#     i = i + 1
#     if line == '2196017 300\n':
#         line = '2196016 300\n'
#         print(line)
#     if line == '\n':
#         s = s + 1
#         line = line.strip("\n")
#     file2.write(line)
#
# print('i =', i)
# print('s =', s)
# print('输出成功....')
# file1.close()
# file2.close()

# import os
# word_i = 0
# space_i = 0
# with open('glove_word2vec-init.txt', 'r+', encoding='utf-8') as f, open('glove_word2vec.txt', 'w+', encoding='utf-8') as new_f:
#     f_list = list(set(f.readlines()))
#     # print(f_list)
#     for i in f_list:
#         word_i = word_i + 1
#         print('word_i =', word_i)
#         if i == '\n':
#             space_i = space_i + 1
#             print('space_i =', space_i)
#             f_list.remove(i)
#     # print(f_list)
#     new_f.writelines(f_list)
#
# print('word_i =', word_i)
# print('space_i =', space_i)
# print('输出成功....')
# f.close()
# new_f.close()
# os.rename('new_staff.txt', 'staff.txt')
