# encoding:utf-8
import json
import os
from os.path import join
import glob
import math
from PIL import Image, ImageFilter
import cv2
import dlib
import numpy as np
import pickle
import random
from tqdm import tqdm
import argparse
from sklearn.preprocessing import StandardScaler
from makeData import Subject_iter
from imutils import face_utils

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
class Subject_iter_eye(Subject_iter):
    def __init__(self, subject_path, args, mode='r'):
        super().__init__(subject_path, args)
        self.mode = mode
        self.eyeball_size = 24
        FOV = 54.5
        self.px_f = 320/math.tan(math.radians(FOV/2))
        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                                 [1.330353, 7.122144, 6.903745],
                                 [-1.330353, 7.122144, 6.903745],
                                 [-6.825897, 6.760612, 4.402142],
                                 [5.311432, 5.485328, 3.987654],
                                 [1.789930, 5.393625, 4.413414],
                                 [-1.789930, 5.393625, 4.413414],
                                 [-5.311432, 5.485328, 3.987654],
                                 [2.005628, 1.409845, 6.165652],
                                 [-2.005628, 1.409845, 6.165652],
                                 [2.774015, -2.080775, 5.048531],
                                 [-2.774015, -2.080775, 5.048531],
                                 [0.000000, -3.116408, 6.097667],
                                 [0.000000, -7.415691, 4.070434]])
    def __iter__(self):
        for f in self.frames_list:
            # limitの回数以上に出力しない
            if self.gen_num >= self.limit:
                raise StopIteration
            if self.is_valid(self.frames_dict[f]):
                f_path = join(self.subject_path, 'frames', f)
                img = cv2.imread(f_path)
                if img is None:
                    break
                # 画像の標準化
                if self.is_std:
                    sc = StandardScaler()
                    img = sc.fit_transform(img)
                self.frame_size = img.shape

                try:
                    parts = self.detect_landmark(img)
                except NoDetectionError:
                    continue
                rvec, tvec = self.estimate_head_pose(parts)
                label_data = self.create_label_data(self.frames_dict[f], rvec, tvec, mode=self.mode)
                head_pose = self.get_head_pose_vec(rvec, tvec)
                if self.mode == 'r':
                    eye_data = self.create_eye_data(img, self.reye_json, self.frames_dict[f])
                else:
                    eye_data = self.create_eye_data(img, self.leye_json, self.frames_dict[f])
                #if isinstance(label_data, (np.ndarray, np.generic)):
# 画像を保存するか？
                if self.is_img_saved and not self.is_std:
                    self.save_image(eye_data, self.mode, f)
                self.gen_num += 1
                yield (eye_data, head_pose, label_data)

    def create_label_data(self, i, rvec, tvec, mode='r'):
        x = self.dotInfo_json['XCam'][i]
        y = self.dotInfo_json['YCam'][i]
        ex, ey, ez = self.get_eye_xyz(rvec, tvec, mode=mode)
        x = (x - ex) / ez * 100
        y = (y - ey) / ez * 100
        data = np.array([x, y])
        self.save_label(data)
        return data
    def estimate_head_pose(self, shape): 
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])
        centerX = self.frame_size[1]/2
        centerY = self.frame_size[0]/2
        camera_matrix = np.float32([[self.px_f, 0, centerX],
                                    [0, self.px_f, centerY],
                                    [0, 0, 1]])
        dist_coeffs = np.zeros((4,1)).astype(np.float32)
        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts,
                                                        camera_matrix, dist_coeffs,
                                                        cv2.SOLVEPNP_EPNP)
        return rotation_vec, translation_vec

    def get_eye_xyz(self, rvec, tvec, mode='r'):
        RT = self.rvec2rmat(rvec, tvec)
        if mode=='r':
            eye_world = (self.object_pts[4] + self.object_pts[5]) / 2
        else:
            eye_world = (self.object_pts[6] + self.object_pts[7]) / 2
        eye_world = np.append(eye_world, 1)
        eye_camera = np.dot(RT, eye_world)
        return eye_camera

    def get_head_pose_vec(self, rvec, tvec):
        rvec = rvec.reshape([3])
        tvec = tvec.reshape([3])
        head_pose = np.concatenate([rvec, tvec])
        return head_pose
    def rvec2rmat(self, rvec, tvec):
        x,y,z = rvec
        Rx = np.float32([[1, 0, 0],
                        [0, np.cos(x), -np.sin(x)],
                        [0, np.sin(x), np.cos(x)]])
        Ry = np.float32([[np.cos(y), 0, np.sin(y)],
                        [0, 1, 0],
                        [-np.sin(y), 0, np.cos(y)]])
        Rz = np.float32([[np.cos(z), -np.sin(z), 0],
                        [np.sin(z), np.cos(z), 0],
                        [0, 0, 1]])
        R = np.dot(np.dot(Rx, Ry), Rz)
        RT = np.concatenate([R, tvec], axis=1)
        return RT

    def detect_landmark(self, img):
        dets = DETECTOR(img[:, :, ::-1])
        if len(dets) > 0:
            parts = PREDICTOR(img, dets[0])
            parts = face_utils.shape_to_np(parts)
            return parts
        else:
            raise NoDetectionError


class NoDetectionError(Exception):
    pass

class Dataset_eye(object):
    def __init__(self, mode, args):
        self.mode = mode
        self.output_dir = join(args.output_dir, self.mode)
        self.batch_size = args.batch_size
        self.is_shuffle = args.is_shuffle
        self.is_std = args.is_std
        self.eye = []
        self.head = []
        self.label = []
    def input_sub(self, subject_iter):
        for sub in tqdm(subject_iter):
            self.eye.append(sub[0])
            self.head.append(sub[1])
            self.label.append(sub[2])
    def debug(self, subject):
        for sub in subject.generator():
            (eyes_data, label_data) = sub
            cv2.imwrite('eye.jpg', eyes_data)
            break
    def convert(self):
        self.eye = np.array(self.eye).astype('float32')
        self.head = np.array(self.head).astype('float32')
        if not self.is_std:
            self.eye = self.eye / 255.
        self.label = np.array(self.label).astype('float32')
    def make_batch(self):
        self.convert()
        data_size =  self.eye.shape[0]
        num_batches = int(data_size / self.batch_size)
        output_path = join(self.output_dir, 'batches')
        os.makedirs(output_path, exist_ok=True)
        if self.is_shuffle:
            indices = np.random.permutation(data_size)
            self.eye = self.eye[indices]
            self.head = self.head[indices]
            self.label = self.label[indices]
        for i in range(num_batches):
            eye_batch = self.eye[i*self.batch_size:(i+1)*self.batch_size]
            head_batch = self.head[i*self.batch_size:(i+1)*self.batch_size]
            label_batch = self.label[i*self.batch_size:(i+1)*self.batch_size]
            batch = [eye_batch, head_batch, label_batch]
            with open(join(output_path, (str(i)+'.pickle')), 'wb') as f:
                pickle.dump(batch, f)

if __name__ == '__main__':

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="make dataset for cnn network.")
    parser.add_argument('-i', '--input_dir', default='/home/kai/dataset_for_research')
    parser.add_argument('-o', '--output_dir', default='/home/ohta/workspace/eye_tracking/eye_data/')
    parser.add_argument('--limit', default=20, type=int)
    parser.add_argument('--batch_size', default=30, type=int)
    parser.add_argument('--is_std', action='store_true')
    parser.add_argument('--is_img_saved', action='store_true')
    parser.add_argument('--is_shuffle', action='store_true')

    args = parser.parse_args()
    input_dir = join(args.input_dir, '0*')
    subject_path_list = glob.glob(input_dir)

    train_set = Dataset_eye('train', args)
    test_set = Dataset_eye('test', args)
    val_set = Dataset_eye('val', args)
    for subject_path in subject_path_list:
        sub_iter = Subject_iter_eye(subject_path, args, mode='r')
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

