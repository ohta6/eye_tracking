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
import argparse
from sklearn.preprocessing import StandardScaler

# 被験者ごとにiteratorからデータを出力
class Subject_iter(object):
    def __init__(self, subject_path, args): #limit=20, is_std=True, is_img_saved=False):
        self.subject_path = subject_path
        self.output_dir = args.output_dir
        self.limit = args.limit
        self.is_std = args.is_std
        self.is_img_saved = args.is_img_saved
        self.gen_num = 0
        self.open_each_json()
# 隣接のframeが似ているため,frameの順番をランダムにする
        self.frames_dict = {f:i for i, f in enumerate(self.frames_list)}
        random.shuffle(self.frames_list)
    def __iter__(self):
        for f in self.frames_list:
            # limitの回数以上に出力しない
            if self.gen_num >= self.limit:
                raise StopIteration
            if self.is_valid(self.frames_dict[f]):
                f_path = join(self.subject_path, 'frames', f)
                img = cv2.imread(f_path, 0)
                if img is None:
                    break
                # 画像の標準化
                if self.is_std:
                    sc = StandardScaler()
                    img = sc.fit_transform(img)
# 全体の画像の縦横を判定
                self.frame_size = img.shape
                input_data = self.create_input_data(img)
                eyes_data = self.create_eyes_data(img, self.frames_dict[f])
                label_data = self.create_label_data(self.frames_dict[f])
# 画像を保存するか？
                if self.is_img_saved and not self.is_std:
                    self.save_image(input_data, 'all', f)
                    self.save_image(eyes_data[0], 'leye', f)
                    self.save_image(eyes_data[1], 'reye', f)
                self.gen_num += 1
                yield (input_data, eyes_data[0], eyes_data[1], label_data)

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
        subject_dir = join(self.output_dir,
                           self.info_json['Dataset'],
                           part,
                           subject_number)
        os.makedirs(subject_dir, exist_ok=True)
        cv2.imwrite(join(subject_dir, fname), img)
    def save_label(self, label):
        subject_number = os.path.basename(self.subject_path) + '.txt'
        label_dir = join(self.output_dir,
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
    def __init__(self, mode, args):
        self.mode = mode
        self.output_dir = join(args.output_dir, self.mode)
        self.batch_size = args.batch_size
        self.is_shuffle = args.is_shuffle
        self.is_std = args.is_std
        self.input = []
        self.leye = []
        self.reye = []
        self.label = []
    def input_sub(self, subject_iter):
        for sub in tqdm(subject_iter):
            self.input.append(sub[0])
            self.leye.append(sub[1])
            self.reye.append(sub[2])
            self.label.append(sub[3])
    def debug(self, subject):
        for sub in subject.generator():
            (input_data, eyes_data, label_data) = sub
            cv2.imwrite('input.jpg', input_data)
            cv2.imwrite('left_eye.jpg', eyes_data[0])
            cv2.imwrite('right_eye.jpg', eyes_data[1])
            break
    def convert(self):
        self.input = np.array(self.input).reshape(-1, 128, 128, 1).astype('float32')
        self.leye = np.array(self.leye).reshape(-1, 32*32*1).astype('float32')
        self.reye = np.array(self.reye).reshape(-1, 32*32*1).astype('float32')
        if not self.is_std:
            self.input = self.input / 255.
            self.leye = self.leye / 255.
            self.reye = self.reye / 255.
        self.label = np.array(self.label).astype('float32')
    def make_batch(self):
        self.convert()
        data_size =  self.input.shape[0]
        num_batches = int(data_size / self.batch_size)
        output_path = join(self.output_dir, 'batches')
        os.makedirs(output_path, exist_ok=True)
        if self.is_shuffle:
            indices = np.random.permutation(data_size)
            self.input = self.input[indices]
            self.leye = self.leye[indices]
            self.reye = self.reye[indices]
            self.label = self.label[indices]
        for i in range(num_batches):
            input_batch = self.input[i*self.batch_size:(i+1)*self.batch_size]
            leye_batch = self.leye[i*self.batch_size:(i+1)*self.batch_size]
            reye_batch = self.reye[i*self.batch_size:(i+1)*self.batch_size]
            label_batch = self.label[i*self.batch_size:(i+1)*self.batch_size]
            batch = [input_batch, leye_batch, reye_batch, label_batch]
            with open(join(output_path, (str(i)+'.pickle')), 'wb') as f:
                pickle.dump(batch, f)

        
if __name__ == '__main__':

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="make dataset for capsule network.")
    parser.add_argument('-i', '--input_dir', default='/home/kai/dataset_for_research')
    parser.add_argument('-o', '--output_dir', default='/home/ohta/workspace/eye_tracking/data/')
    parser.add_argument('--limit', default=20, type=int)
    parser.add_argument('--batch_size', default=30, type=int)
    parser.add_argument('--is_std', action='store_true')
    parser.add_argument('--is_img_saved', action='store_true')
    parser.add_argument('--is_shuffle', action='store_true')

    args = parser.parse_args()
    input_dir = join(args.input_dir, '0*')
    subject_path_list = glob.glob(input_dir)

    train_set = Dataset('train', args)
    test_set = Dataset('test', args)
    val_set = Dataset('val', args)
    for subject_path in subject_path_list:
        sub_iter = Subject_iter(subject_path, args)
        dataset_kind = sub_iter.what_Dataset()
        if dataset_kind == 'train':
            train_set.input_sub(sub_iter)
        if dataset_kind == 'test':
            test_set.input_sub(sub_iter)
        if dataset_kind == 'val':
            val_set.input_sub(sub_iter)
    train_set.make_batch()
    test_set.make_batch()
    val_set.make_batch()

