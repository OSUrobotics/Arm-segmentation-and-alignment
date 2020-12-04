#!/usr/bin/env python3

###############
#Author: Yi Herng Ong, Chang Ju Lee
#Purpose: import kinova jaco j2s7s300 into Mujoco environment
#
#("/home/graspinglab/NearContactStudy/MDP/jaco/jaco.xml")
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
#table range(x:117~584, y:0~480)


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


def degreetorad(degree):
	rad = degree/(180/math.pi)
	return rad


def read_ang_data(filename):# for the data form1
    f = open(filename, "r")
    if f.mode == 'r':
        angle_data = f.read()
    data = [float(angle_data.split()[i]) for i in range(1,len(angle_data.split()),2)]
    return data

def read_ang_data_v(filename): #for the data form2
	f = open(filename, "r")
	if f.mode == 'r':
		angle_data = f.read()
	data = [[float(i) for i in line.split(',')] for line in angle_data.split()]
	return data

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
	return new_img

def view_main():
	sim = Kinova_MJ()
	#sim.init_camera()
	#sim.run_mujoco()
	#sim.run_mujoco([2, 1, 0.1, 0.75, 4.62, 4.48, 4.88, 0.0, 0.0, 0.0])

	#ang[0] = -1 * ang[0] + math.pi / 2
	#ang[1] = math.pi - ang[1]  # -1+pi
	#ang[2] = ang[2]
	#ang[3] = ang[3]
	#ang[4] = ang[4]
	#ang[5] = ang[5]
	#ang[6] = ang[6]
	#ang = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

	#ang[0] = -0.257819
	#ang[1] = -12.501069
	#ang[2] = -3.083040
	#ang[3] = -1.163000
	#ang[4] =  -0.010763
	#ang[5] =  2.487189
	#ang[6] =  0.075274
	#ang[7] =  0.000000
	#ang[8] =  0.000000
	#ang[9] =  0.000000

	#ang[0] = -1 * ang[0]
	#ang[1] = math.pi/2 + ang[1]  # -1+pi
	#ang[2] = -ang[2]
	#ang[3] = -ang[3]
	#ang[4] = ang[4]
	#ang[5] = ang[5]
	#ang[6] = ang[6]
	#ang[7] = 0
	#ang[8] = 0
	#ang[9] = 0
	#math.pi / 2 - (12.501069 % (2 * math.pi) )

	#[ 0]:  5.233082
	#2, 1, 0.1, 0.75, 4.62, 4.48, 4.88, 0.0, 0.0, 0.0

	#ang = [0.257819, math.pi / 4 -12.501069 , 3.083040 , 1.163000, 4.62-0.010763, 4.48, 4.88, 0.0, 0.0, 0.0]
	#ang = [0.257819, 1 - 12.501069, 3.083040-0.1,  1.163000-0.75, 4.62-0.010763 , 4.48+2.487189, 4.88, 0.0, 0.0, 0.0]

	#ang1 = [4.941322-math.pi/2, math.pi/2 + 2.842190, 6.280905, 0.758276, 4.630653, 4.496469, 5.032572, 1.127277, 1.127277,
	#	   1.127277]
	#ori_mask1 = sim.run_mujoco(ang1)
	#plt.imsave("img_view1.png", ori_mask1)
	#ang = [5.233082-math.pi/2, math.pi/2 + 4.626677, 6.268495, 1.098733, 4.316920, 4.769392, 6.449859, 0.188496, 0.188496,
	#	   0.188496]
	#ori_mask2 = sim.run_mujoco(ang)
	#plt.imsave("img_view2.png", ori_mask2)
	#ang = [5.475519 - math.pi / 2, math.pi/2 + 4.084754, 6.325216, 0.762528, 4.574362, 4.674051, 6.238493, 0.188496,
	#	   0.188496,  0.18896]
	#ori_mask3 = sim.run_mujoco(ang)
	#plt.imsave("img_view3.png", ori_mask3)
	#print("done")

	#ang = [5.223545 - math.pi / 2, math.pi/2 + 4.625666, 6.267092, 1.105346, 4.312319, 4.769968, 6.442017, 1.113725,
	#	   1.113725,  1.113725]
	#ori_mask4 = sim.run_mujoco(ang)
	#plt.imsave("img_view4.png", ori_mask4)
	#print("done")

	#ang = [5.224459-math.pi/2 , math.pi/2 + 4.625699, 6.267181, 1.104913, 4.312989, 4.769700, 6.442746, 0.188496, 0.188496,
	#	   0.188496]
	#ori_mask5 = sim.run_mujoco(ang)
	#plt.imsave("img_view5.png", ori_mask5)
	#print("done")

	#ang = [5.060067 - math.pi / 2, math.pi/2 + 3.117000, 6.290349, 0.758301, 4.621203, 4.530230, 5.296271, 0.188496,
	#	   0.188496,
	#	   0.188496]
	#ori_mask6 = sim.run_mujoco(ang)
	#plt.imsave("img_view6.png", ori_ask6)
	#print("done")

	#ang = [5.060067 - math.pi / 2, 0 , 6.290349, 0.758301, 4.621203, 4.530230, 5.296271, 0.188496,
	#	   0.188496,
	#	   0.188496]
	#ori_mask6 = sim.run_mujoco(ang)
	#plt.imsave("img_view11.png", ori_mask6)
	#print("done")

	#ang = [5.060067 - math.pi / 2, - math.pi/2 , 6.290349, 0.758301, 4.621203, 4.530230, 5.296271, 0.188496,
	#   0.188496,
	#	   0.188496]
	#ori_mask6 = sim.run_mujoco(ang)
	#plt.imsave("img_view12.png", ori_mask6)
	#print("done")

	#ang = [5.060067 - math.pi / 2, math.pi/2 , 6.290349, 0.758301, 4.621203, 4.530230, 5.296271, 0.188496,
	#	   0.188496,
	#	   0.188496]
	#ori_mask6 = sim.run_mujoco(ang)
	#plt.imsave("img_view13.png", ori_mask6)
	#print("done")

	#ang = [5.060067 - math.pi / 2, math.pi, 6.290349, 0.758301, 4.621203, 4.530230, 5.296271, 0.188496,
	#	   0.188496,
	#	   0.188496]
	#ori_mask6 = sim.run_mujoco(ang)
	#plt.imsave("img_view14.png", ori_mask6)
	#print("done")

	data = read_ang_data_v("angle_data_su.txt")	#print(data[1])
	print(1)#len(data)
	for i in range(len(data)):  #len(data)
		data[i][0] = -data[i][0] + math.pi+0.12
		data[i][1] = - data[i][1] + math.pi
		data[i][4] =  data[i][4] - 0.1
		#print(data[1])
		if i % 5 == 0:
			ori_mask3 = sim.run_mujoco(data[i])
			ori_mask3 = corp_margin(ori_mask3)
			plt.imsave("outtest/img_view"+str(i+8118)+".jpg", ori_mask3)


	# d_pos_rot = [0, 0, 0.03, 0, 0]
	#real_in = "real_image"
	#filename = angle_data
	#print("working...")
	#print(target)
	# d_pos_rot = [-0, 0, 0,0,0] # x - up, y + left, z + zoom in (hand)
	#resetin = []
	#resetin.append(sim._viewer.cam.lookat[:].copy())
	#resetin.append(sim._viewer.cam.azimuth)
	#resetin.append(sim._viewer.cam.elevation)
	#resetin.append(sim._viewer.cam.distance)

	#resetin = []
	#resetin.append([-0.05814825 - 0.45016306 - 0.01190703])
	#resetin.append(-1987.7087124)
	#resetin.append(-89.0)
	#resetin.append(0.7788944976208755)

	#log1
	#d_pos = [0.3794864 , 0.52786403, 0.85616615]
	#d_rot = [1.01242558, 3.95838427]

	#log2
	#d_pos = [0.3794864, 0,0]
	#d_rot = [1.01242558, 3.95838427]

	#log3
	#d_pos = [1.018034,-5.018034,-2.618034]
	#d_rot = [0,0]

	#log4
	#d_pos = [-0.95519511,  3.44125577,  1.07168055]
	#d_rot = [7.02141309, 5.29992549]
	#d_pos = [0,-3,0]
	#d_rot = [0,0]

	#_ = seg_process(d_pos,[], filename, sim,  real_in,rotation_fl =1,imag_save_fl = 1,indata = resetin)
	#score = seg_process([],d_rot, filename, sim, real_in,rotation_fl =2,imag_save_fl = 1,indata = resetin)
	#print(score)

if __name__ == '__main__':
	view_main()







