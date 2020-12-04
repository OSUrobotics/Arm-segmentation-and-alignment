#!/usr/bin/env python3
###############
# Author: Yi Herng Ong, Chang Ju Lee
# Purpose: import kinova jaco j2s7s300 into Mujoco environment
#
# ( "/home/leelcz/Graduate_project/mujoco_kinova_rendering/j2s7s300/j2s7s300.xml")
#
###############
import gym
import numpy as np
from mujoco_py import MjViewer, load_model_from_path, MjSim, MjRenderContextOffscreen
from mujoco_py.generated import const
from PID_Kinova_MJ import *
import math
import matplotlib.pyplot as plt
import cv2
from random import random
from scipy import optimize
import time
import sys
import os
import inspect

# table range(x:117~584, y:0~480) for the testing video env

class Kinova_MJ(object):
	def __init__(self):
		# self._model = load_model_from_path("/home/graspinglab/NearContactStudy/MDP/kinova_description/j2s7s300.xml")
		self._model = load_model_from_path("j2s7s300/j2s7s300.xml")
		self._sim = MjSim(self._model)

		self._viewer = MjRenderContextOffscreen(self._sim, 0)
		#self._viewer = MjViewer(self._sim)
		self._timestep = 0.0001
		self._sim.model.opt.timestep = self._timestep

		self._torque = [0,0,0,0,0,0,0,0,0,0]
		self._velocity = [0,0,0,0,0,0,0,0,0,0]

		self._jointAngle = [0,0,0,0,0,0,0,0,0,0]
		self._positions = [] # ??
		self._numSteps = 0
		self._simulator = "Mujoco"
		self._experiment = "" # ??
		self._currentIteration = 0

	def set_step(self, seconds):
		self._numSteps = seconds / self._timestep

	def run_mujoco(self,thetas = [2, 1, 0.1, 0.75, 4.62, 4.48, 4.88, 0.0, 0.0, 0.0]):
		self._sim.data.qpos[0:10] = thetas[:] 		#first 10 - first 7 are joint angles, next 3 are finger pose
		self._sim.forward()
		#if fl ==0:
		#	self._viewer.render(640, 480, 0)
		#	img = np.asarray(self._viewer.read_pixels(640, 480, depth=False)[::-1, :, :], dtype=np.uint8)
		#else:
		#	self._viewer.render(640, 480, -1)
		#	img = np.asarray(self._viewer.read_pixels(640, 480, depth=False)[::-1, :, :], dtype=np.uint8)

	#img = self._sim.render(width=640, height=480, camera_name="camera2")

		self._viewer.render(1920, 1080, 0)
		img = np.asarray(self._viewer.read_pixels(1920, 1080, depth=False)[::-1, :, :], dtype=np.uint8)
		#img = self._sim.render(width=640, height=480, camera_name="camera2")
		return img

	#def init_camera(self):
	#	self._viewer.cam.lookat[:] = [-0.05348492, - 0.46381618, - 0.01835867]
	#	self._viewer.cam.azimuth = 172.29128760000003
	#	self._viewer.cam.elevation = -89.0
	#	self._viewer.cam.distance = 0.7788944976208755

	#def reset_camera(self,input):
	#	self._viewer.cam.lookat[:] = input[0]
	#	self._viewer.cam.azimuth = input[1]
	#	self._viewer.cam.elevation = input[2]
	#	self._viewer.cam.distance = input[3]

	#def moving_camera(self,d_position = [],d_rotation = []):
		# Make the free camera look at the scene
	#	d_position = np.array(d_position)
	#	d_rotation = np.array(d_rotation)
	#	if len(d_position) == 3:
	#		dx = d_position[0]/100
	#		dy = d_position[1]/100
	#		dz = d_position[2]/100
	#		action = const.MOUSE_MOVE_H
	#		self._viewer.move_camera(action, dx, dy)
	#		action = const.MOUSE_MOVE_V
	#		action = const.MOUSE_ZOOM
	#		self._viewer.move_camera(action, 0, dz)

	#	if len(d_rotation) == 2:
	#		drx = d_rotation[0]
	#		dry = d_rotation[1]
	#		action = const.MOUSE_ROTATE_H
	#		self._viewer.move_camera(action, drx, dry)
	#		action = const.MOUSE_ROTATE_V
	#		self._viewer.move_camera(action, drx, dry)


#def degreetorad(degree):
#    rad = degree / (180 / math.pi)
#    return rad


def read_ang_data(filename):
    f = open(filename, "r")
    if f.mode == 'r':
        angle_data = f.read()
    data = [float(angle_data.split()[i]) for i in range(1, len(angle_data.split()), 2)]
    return data


def read_ang_data_v(filename): #for the data form2
	f = open(filename, "r")
	if f.mode == 'r':
		angle_data = f.read()
	data = [[float(i) for i in line.split(',')] for line in angle_data.split()]
	return data



def grab_cuting(target, K, sim_img, real, nwmask=[], save_fl=0):
    # img = cv2.imread(target+'/real.jpg')
    img = real
    mask = np.zeros(img.shape[:2], np.uint8)
    mask_ch = np.zeros(img.shape[:2], np.uint8)
    test_mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    gray_mjimage = cv2.cvtColor(sim_img, cv2.COLOR_BGR2GRAY)
    _, mj_seg = cv2.threshold(gray_mjimage, 0, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    mask_erod = cv2.dilate(mj_seg, kernel, iterations=12)  # using dia as erod since the color

    h, w = mj_seg.shape
    # combine_mask = np.zeros((h, w, 3), np.uint8)
    for ii in range(h):
        for j in range(w):
            if mask_erod[ii][j] == 0:
                # combine_mask[ii][j][0] = 255
                mask[ii][j] = 1
            elif mask_erod[ii][j] == 255 and mj_seg[ii][j] == 0:
                mask[ii][j] = 3

    for ii in range(h):
        for j in range(w):
            if mask[ii][j] == 1:
                mask_ch[ii][j] = 0
            elif mask[ii][j]==3:
                mask_ch[ii][j] = 50
            elif mask[ii][j]==0:
                mask_ch[ii][j] = 100



    mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask[:, :, np.newaxis]
    if save_fl == 1:
        #target = "output"
        plt.imsave(target + "/Grab" + str(K) + ".jpg", img)
        plt.imsave(target + "/Grab_mask" + str(K) + ".jpg", mask_ch)
        plt.imsave(target + "/Grab_mjmask" + str(K) + ".jpg", mask_erod)
    return mask, mj_seg


def overlap_image(realim, mj, realgray, i, target):
    h, w = realim.shape
    combine_mask = np.zeros((h, w, 3), np.uint8)

    for ii in range(h):
        for j in range(w):
            if realim[ii][j] == 1:
                combine_mask[ii][j][0] = 255
            if mj[ii][j] == 0:
                combine_mask[ii][j][1] = 255
            combine_mask[ii][j][2] = realgray[ii][j]

    #cv2.imshow('My Image2', combine_mask)
    plt.imsave(target + "/contour{}.jpg".format(i), combine_mask)


def cal_score(inmask,insimu):
    #real mask 0 1 (arm)
    #mj mask 0 (arm) 255
    h,w = inmask.shape
    score = 0
    sim_score = 0
    mask_score = 0
    for x in range(h):
        for y in range(w):
            if insimu[x][y] == 0:
                sim_score +=1
            if inmask[x][y] == 1:
                mask_score +=1
            if insimu[x][y] == 0 and inmask[x][y] == 1:
                score += 1
    return score

def corp_margin(img):
	img2 = img.sum(axis=2)
	(row, col) = img2.shape
	out_image = img.copy()
	row_top = 0
	pix_ma = img2.sum(axis=1)[0]
	for sr in range(0, row):
		if img2.sum(axis=1)[sr] > pix_ma:
			row_top = sr + 1
			break
	cutting = img[0:row_top + 1].copy()
	new_img = img[row_top:-1]
	out_image[0:row - row_top - 1] = new_img.copy()
	out_image[row - row_top - 2:-1] = cutting.copy()
	return out_image

#def seg_process(d_pos, d_rot, filename, sim, target, imag_save_fl=0, rotation_fl=0, indata=[]):
def seg_process(filename, sim, input="real_image" ,output = "output3", imag_save_fl=0):
    #print(indata)
    #if len(indata) != 0:
    #    sim.reset_camera(indata)
    data = read_ang_data_v(filename)
    #FING1 = 70
    #FING2 = 72
    #FING3 = 73
    #d_p = []  # x - right, y + down, z + zoom in
    #d_rp = []
    #if rotation_fl == 1:
    #    d_rp = d_rot
    #    sim.moving_camera(d_p, d_rp)
    #elif rotation_fl == 0:
    #    d_p = d_pos
    #    sim.moving_camera(d_p, d_rp)

    # d_p = [-0.01, 0.01, 0.01]
    score = []
    data = np.array(data)
    #for i in range(int(len(data)/2)):
    for i in range(int(len(data)/2)): #ang = bb[i].copy()
        k = i*2
        #ang.append(FING1)
        #ang.append(FING2)
        #ang.append(FING3)
        #ang = np.array(ang) / (180 / math.pi)
        #ang[0] = -1 * ang[0] + math.pi / 2
        #ang[1] = math.pi - ang[1]  # -1+pi
        #ang[2] = ang[2]
        #ang[3] = ang[3]
        #ang[4] = ang[4]
        #ang[5] = ang[5]
        #ang[6] = ang[6]
        print(i,k)

        data[k][0] = -data[k][0] + math.pi + 0.12
        data[k][1] = - data[k][1] + math.pi
        data[k][4] = data[k][4] - 0.1
        #if i % 2 == 0:
        ori_mask = sim.run_mujoco(data[k])
        ori_mask = corp_margin(ori_mask)
        realimg = cv2.imread(input + '/real{}.jpg'.format(i+1))
        realim, mj = grab_cuting(output, i, ori_mask, realimg, save_fl=1)
        realgray = cv2.cvtColor(realimg, cv2.COLOR_BGR2GRAY)
        if imag_save_fl == 1:
            overlap_image(realim, mj, realgray, i, output)
        score.append(cal_score(realim, mj))
    #score_m = sum(score) / len(score)
    #print(score, score_m)

        #print(sim._viewer.cam.lookat[:].copy())
        #print(sim._viewer.cam.azimuth)
        #print(sim._viewer.cam.elevation)
        #print(sim._viewer.cam.distance)

    return 0#score_m * -1


def main(angle_data,out_path):
    sim = Kinova_MJ()
    #sim.init_camera()
    #sim.run_mujoco(fl=1)
    # d_pos_rot = [0, 0, 0.03, 0, 0]
    #target = "testdata5"
    # filename = target+"/angle_data.txt"
    #filename = target + "/angle_sequence.csv"
    real_in = "real_image"
    filename = angle_data

    print("working...")
    # d_pos_rot = [-0, 0, 0,0,0] # x - up, y + left, z + zoom in (hand)
    #resetin = []
    #resetin.append(sim._viewer.cam.lookat[:].copy())
    #resetin.append(sim._viewer.cam.azimuth)
    #resetin.append(sim._viewer.cam.elevation)
    #resetin.append(sim._viewer.cam.distance)
    #d_pos = [1, -1, 0]
    #d_pos = [0, 0, 0]
    #d_rot = [0, 0]
    #_ = seg_process(d_pos, [], filename, sim, real_in, rotation_fl=0, imag_save_fl=1, indata=resetin)

    score = seg_process(filename, sim, imag_save_fl=1)
    #resetin = []
    #resetin.append(sim._viewer.cam.lookat[:].copy())
    #resetin.append(sim._viewer.cam.azimuth)
    #resetin.append(sim._viewer.cam.elevation)
    #resetin.append(sim._viewer.cam.distance)
    #score = seg_process([], d_rot, filename, sim, real_in, rotation_fl=1, imag_save_fl=1,indata=resetin)
    print(score)


if __name__ == '__main__':
    a = sys.argv[1]
    file_path = "output3"
    try:
        os.makedirs(file_path)
    except FileExistsError:
        pass
    print(a)
    main(a,file_path)





