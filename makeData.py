# encoding:utf-8
import json
import os
from os.path import join
import glob
from PIL import Image, ImageFilter
import cv2
import numpy as np
import pickle
import random
from tqdm import tqdm

input_dir = '/home/kai/dataset_for_research/0*'
base_output_dir = '/home/ohta/workspace/eye_tracking/data/'

# 被験者ごとにインスタンスを作りgeneratorからデータを出力
class Subject(object):
    def __init__(self, subject_path, limit=20):
        self.subject_path = subject_path
        self.limit = limit
        self.gen_num = 0
        self.open_each_json()
# 隣接のframeが似ているため,frameの順番をランダムにする
        self.frames_dict = {f:i for i, f in enumerate(self.frames_list)}
        random.shuffle(self.frames_list)
    
    def open_each_json(self):
        self.face_json = self.open_json('appleFace.json')
        self.leye_json = self.open_json('appleLeftEye.json')
        self.reye_json = self.open_json('appleRightEye.json')
        self.info_json = self.open_json('info.json')
        self.frames_list = self.open_json('frames.json')
        self.dotInfo_json = self.open_json('dotInfo.json')

    def open_json(self, json_filename):
        json_path = join(self.subject_path, json_filename)
        with open(json_path) as f:
            json_file = json.load(f)
        return json_file
# generatorをfor文で回す
    def generator(self):
        for f in self.frames_list:
            # limitの回数以上に出力しない
            if self.gen_num >= self.limit:
                raise StopIteration
            if self.is_valid(self.frames_dict[f]):
                f_path = join(self.subject_path, 'frames', f)
                img = cv2.imread(f_path, 0)
                if img is None:
                    break
# 全体の画像の縦横を判定
                self.frame_size = img.shape
                input_data = self.create_input_data(img)
                self.save_image(input_data, 'all', f)
                eyes_data = self.create_eyes_data(img, self.frames_dict[f])
                self.save_image(eyes_data[0], 'leye', f)
                self.save_image(eyes_data[1], 'reye', f)
                label_data = self.create_label_data(self.frames_dict[f])
                self.gen_num += 1
                yield (input_data, eyes_data, label_data)
    def create_input_data(self, img):
# 縦横比をそのままにする。余白は黒
        base_img = np.zeros((128, 128))
        if self.frame_size == (640, 480):
            img = cv2.resize(img, (96, 128))
            base_img[:, 16:112] = img
        if self.frame_size == (480, 640):
            img = cv2.resize(img, (128, 96))
            base_img[16:112, :] = img
        return base_img
    def create_eyes_data(self, img, i):
        leye_data = self.create_eye_data(img, self.leye_json, i)
        reye_data = self.create_eye_data(img, self.reye_json, i)
        data = np.array([leye_data, reye_data])
        return data
    def create_eye_data(self, img, eye_js, i):
        eye_img = self.crop_eye(img, eye_js, i)
        try:
            eye_img = cv2.resize(eye_img, (32, 32))
        except:
            print(self.subject_path)
            print(i)
        return eye_img
    def create_label_data(self, i):
        x = self.dotInfo_json['XCam'][i]
        y = self.dotInfo_json['YCam'][i]
        data = np.array([x, y])
        self.save_label(data)
        return np.array([x, y])
# eye_jsonのXYはfaceの相対値であることに注意
    def crop_eye(self, img, eye_js, i):
        (f_L, f_U, _, _) = extract_bb(self.face_json, i)
        (e_L, e_U, e_R, e_D) = extract_bb(eye_js, i)
        eye_left = int(f_L + e_L)
        eye_up = int(f_U + e_U)
        eye_right = int(f_L + e_R)
        eye_down = int(f_U + e_D)
# 画像の範囲をクリッピングが超える場合がある
        if eye_up < 0:
            eye_up = 0
        if eye_down > self.frame_size[0]:
            eye_down = self.frame_size[0]
        if eye_left < 0:
            eye_left = 0
        if eye_right > self.frame_size[1]:
            eye_right = self.frame_size[1]
        cropped_img = img[eye_up:eye_down, eye_left:eye_right]
        return cropped_img
        
    def is_valid(self, i):
        if (self.face_json['IsValid'][i] == 1) and \
                (self.leye_json['IsValid'][i] == 1) and \
                (self.reye_json['IsValid'][i] == 1):
            return True
        return False
# 被験者ごとにtrain,test,valに分類されている
    def what_Dataset(self):
        return self.info_json['Dataset']
    def save_image(self, img, part, fname):
        subject_number = os.path.basename(self.subject_path)
        subject_dir = join(base_output_dir,
                           self.info_json['Dataset'],
                           part,
                           subject_number)
        os.makedirs(subject_dir, exist_ok=True)
        cv2.imwrite(join(subject_dir, fname), img)
    def save_label(self, label):
        subject_number = os.path.basename(self.subject_path) + '.txt'
        label_dir = join(base_output_dir,
                          self.info_json['Dataset'],
                          'label')
        os.makedirs(label_dir, exist_ok=True)
        with open(join(label_dir, subject_number), mode='a') as f:
            np.savetxt(f, label, delimiter=',')
        
        
def extract_bb(js, i):
    left = js['X'][i]
    up = js['Y'][i]
    right = left + js['W'][i]
    down = up + js['H'][i]
    return (left, up, right, down)

class Dataset(object):
    def __init__(self, mode):
        self.mode = mode
        self.output_dir = join(base_output_dir, self.mode)
        self.input = []
        self.eyes = []
        self.label = []
    def input_sub(self, subject):
        for sub in tqdm(subject.generator()):
            self.input.append(sub[0])
            self.eyes.append(sub[1])
            self.label.append(sub[2])
    def debug(self, subject):
        for sub in subject.generator():
            (input_data, eyes_data, label_data) = sub
            cv2.imwrite('input.jpg', input_data)
            cv2.imwrite('left_eye.jpg', eyes_data[0])
            cv2.imwrite('right_eye.jpg', eyes_data[1])
            break
    def convert(self):
        self.input = np.array(self.input)
        self.input = self.input.reshape(-1, 128, 128, 1).astype('float32') / 255.
        self.eyes = np.array(self.eyes)
        self.eyes = self.eyes.reshape(-1, 32*32*1*2).astype('float32') / 255.
        self.label = np.array(self.label).astype('float32')
    def save_data(self):
        self.convert()
        np.savez_compressed(join(self.output_dir, 'input.npz'), self.input)
        np.savez_compressed(join(self.output_dir, 'eyes.npz'), self.eyes)
        np.savez_compressed(join(self.output_dir, 'label.npz'), self.label)
    def make_batch(self, batch_size=100, shuffle=True):
        self.convert()
        data_size =  self.input.shape[0]
        num_batches = int(data_size / batch_size)
        output_path = join(self.output_dir, 'batches')
        os.makedirs(output_path, exist_ok=True)
        if shuffle == True:
            indices = np.random.permutation(data_size)
            self.input = self.input[indices]
            self.eyes = self.eyes[indices]
            self.label = self.label[indices]
        for i in range(num_batches):
            input_batch = self.input[i*batch_size:(i+1)*batch_size]
            eyes_batch = self.eyes[i*batch_size:(i+1)*batch_size]
            label_batch = self.label[i*batch_size:(i+1)*batch_size]
            batch = [input_batch, eyes_batch, label_batch]
            with open(join(output_path, (str(i)+'.pickle')), 'wb') as f:
                pickle.dump(batch, f)

        
def batch_iter():
    train_dir = join(base_output_dir, 'train', 'batches')
    dir_list = os.listdir(train_dir)
    num_batches = len(dir_list)
    def train_generator():
        batch_list = glob.glob(train_dir + '/*')
        for batch_path in batch_list:
            with open(batch_path, 'rb') as f:
                batch = pickle.load(f)
            yield batch
    return num_batches, train_generator()

if __name__ == '__main__':
    num_batches, generator = batch_iter()
    print(num_batches)
    for batch in generator:
        print(batch)
        break
    """
    subject_path_list = glob.glob(input_dir)
    train_set = Dataset('train')
    test_set = Dataset('test')
    val_set = Dataset('val')
    for subject_path in subject_path_list:
        sub = Subject(subject_path)
        dataset_kind = sub.what_Dataset()
        if dataset_kind == 'train':
            train_set.input_sub(sub)
        if dataset_kind == 'test':
            test_set.input_sub(sub)
        if dataset_kind == 'val':
            val_set.input_sub(sub)
    train_set.make_batch()
    test_set.make_batch()
    val_set.make_batch()
    """

