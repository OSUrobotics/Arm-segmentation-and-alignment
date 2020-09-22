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

	def run_mujoco(self,thetas = [2, 1, 0.1, 0.75, 4.62, 4.48, 4.88, 0.0, 0.0, 0.0],fl=0):
		self._sim.data.qpos[0:10] = thetas[:] 		#first 10 - first 7 are joint angles, next 3 are finger pose
		self._sim.forward()
		if fl ==0:
			self._viewer.render(640, 480, 0)
			img = np.asarray(self._viewer.read_pixels(640, 480, depth=False)[::-1, :, :], dtype=np.uint8)
		else:
			self._viewer.render(640, 480, -1)
			img = np.asarray(self._viewer.read_pixels(640, 480, depth=False)[::-1, :, :], dtype=np.uint8)
		return img

	def init_camera(self):
		self._viewer.cam.lookat[:] = [-0.05348492, - 0.46381618, - 0.01835867]
		self._viewer.cam.azimuth = 172.29128760000003
		self._viewer.cam.elevation = -89.0
		self._viewer.cam.distance = 0.7788944976208755

	def reset_camera(self,input):
		self._viewer.cam.lookat[:] = input[0]
		self._viewer.cam.azimuth = input[1]
		self._viewer.cam.elevation = input[2]
		self._viewer.cam.distance = input[3]

	def moving_camera(self,d_position = [],d_rotation = []):
		# Make the free camera look at the scene
		d_position = np.array(d_position)
		d_rotation = np.array(d_rotation)
		if len(d_position) == 3:
			dx = d_position[0]/100
			dy = d_position[1]/100
			dz = d_position[2]/100
			action = const.MOUSE_MOVE_H
			self._viewer.move_camera(action, dx, dy)
			action = const.MOUSE_MOVE_V
			self._viewer.move_camera(action, dx, dy)
			action = const.MOUSE_ZOOM
			self._viewer.move_camera(action, 0, dz)

		if len(d_rotation) == 2:
			drx = d_rotation[0]
			dry = d_rotation[1]
			action = const.MOUSE_ROTATE_H
			self._viewer.move_camera(action, drx, dry)
			action = const.MOUSE_ROTATE_V
			self._viewer.move_camera(action, drx, dry)


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

################
## this part is Image dilation
## the purpose of dilation is to reduce the nose of the background for higher efficiency
## in claster matching part
################
def img_dia(target,itr,ori_mask,kernel,kernel2 = None,save_fl = 0):
    if type(kernel2)== type(kernel):
        kernel2 = kernel
    #img_dir = target + "/img.png"
    img_r_dir = target + "/real{}.jpg".format(itr)
    #plt.imsave(img_dir, ori_mask)
    img_dilation = cv2.dilate(ori_mask, kernel, iterations=3)
    img_real = cv2.imread(img_r_dir)
    img_r_dilation = cv2.dilate(img_real, kernel, iterations=3)
    if save_fl == 1:
        target = "output"
        plt.imsave(target +"/img_dila.png", img_dilation)
        plt.imsave(target+"/real_img_dila.png", img_r_dilation)
    return img_dilation,img_r_dilation,img_real

###################
## This part change the color coordination of the image
## The purpose of this part is to increase the accuracy of the image dilation part
###################
def RGB2YUV(target,mask,real,save_fl = 0):
    mask_out = cv2.cvtColor(mask, cv2.COLOR_BGR2YUV)
    real_out = cv2.cvtColor(real, cv2.COLOR_BGR2YUV)
    if save_fl == 1:
        target = "output"
        plt.imsave(target+"/mask_YUV.png", mask_out)
        plt.imsave(target+"/real_YUV.png", real_out)
    return mask_out, real_out

def pixel_cal(mask):
    n_x, n_y = mask.shape
    out = 0
    for i in range(n_x):
        for j in range(n_y):
            if mask[j][i] != 0:
                out +=1
    return out

########################
## this part generates the K-mean cluster for the image
## the defult number of K is 4, these 4 cluster is use to matching the
## real image and the simulation result to find out the position of the cam


def Kmeanclus(target,mask,real,parK = 5,save_fl = 0):
	Z_mask = np.float32(mask.reshape((-1, 3)))
	Z_real = np.float32(real.reshape((-1, 3)))

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = parK # modifies]d
	ret_real, label_real, center_real = cv2.kmeans(Z_real, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
	ret_mask, label_mask, center_mask = cv2.kmeans(Z_mask, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
	center_real = np.uint8(center_real)
    #1023
	label_real_mod = label_real.reshape(480,640)
	label_real_out = label_real_mod
	for i in range(480):
		for j in range(118):
			label_real_mod[i][j] = label_real_mod[10][118]

	res_real_mod = center_real[label_real_mod]
	center_mask = np.uint8(center_mask)
	res_mask = center_mask[label_mask.flatten()]
	res2_mask = res_mask.reshape((mask.shape))
	if save_fl == 1:
		target = "output"
		plt.imsave(target+"/real_K"+str(K)+"_mod.png", res_real_mod)
		plt.imsave(target+"/mask_K"+str(K)+".png", res2_mask)

	label_mask_out = label_mask.reshape(480, 640)

	return label_mask_out,label_real_out

#117 584

def cluster_matching_mask_g(target, mask_l,real_l,K,itr,save_fl = 0):
	pixel_cal_r = np.ones(K,)
	pixel_cal_m = np.ones(K,)
	for i in range(480):
		for j in range(117, 585):
			pixel_cal_m[mask_l[i][j]] += 1
			pixel_cal_r[real_l[i][j]] += 1

	or_m = np.argsort(pixel_cal_m)
	or_r = np.argsort(pixel_cal_r)
	sim_arm_pix = 0
	for i in range(len(or_m)-1):
		sim_arm_pix += pixel_cal_m[or_m[i]]

	real_arm_pix = 0
	diff_l = []
	for i in range(len(or_r)):
		real_arm_pix += pixel_cal_r[or_r[i]]
		diff_l.append(abs(real_arm_pix - sim_arm_pix))
	threshold = diff_l.index(min(diff_l))+1
	or_list = or_r[0:threshold]
	out_mask = np.zeros([mask_l.shape[0],mask_l.shape[1],3],np.uint8)
	out_label = np.zeros(real_l.shape,np.uint8)

	for th in range(threshold):
		for i in range(480):
			for j in range(117, 585):
				if (real_l[i][j] in or_list) and out_label[i][j] == 0:
					k = random()
					if k < 1:
						out_mask[i][j] = [255,255,255]
						out_label[i][j] = 255
					else:
						out_mask[i][j] = [0,0,0]
						out_label[i][j] = 0
				elif out_label[i][j] != 0:
					continue
				else:
					out_mask[i][j] = [0, 0, 0]
					out_label[i][j] = 0
	if save_fl == 1:
		target = "output"
		plt.imsave(target + "/mask_out_K" + str(itr) + ".png", out_mask)

	return out_label


def grab_cuting(target,K,sim_img,real,nwmask=[],save_fl = 0):
	img = real
	mask = np.zeros(img.shape[:2], np.uint8)
	test_mask = np.zeros(img.shape[:2], np.uint8)
	bgdModel = np.zeros((1, 65), np.float64)
	fgdModel = np.zeros((1, 65), np.float64)
	gray_mjimage = cv2.cvtColor(sim_img, cv2.COLOR_BGR2GRAY)
	_, mj_seg = cv2.threshold(gray_mjimage, 0, 255, cv2.THRESH_BINARY_INV)
	contours, hierarchy = cv2.findContours(nwmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	area = []
	for k in range(len(contours)):
		area.append(cv2.contourArea(contours[k]))
	max_idx = np.argmax(np.array(area))
	h, w = nwmask.shape[:2]
	fillmask = np.zeros((h, w, 3), np.uint8)
	cv2.drawContours(fillmask, contours, max_idx, (255, 255, 255), -1)
	outmask = cv2.cvtColor(fillmask, cv2.COLOR_BGR2GRAY)
	# wherever it is marked white (sure foreground), change mask=1
	# wherever it is marked black (sure background), change mask=0a
	mask[outmask == 0] = 0
	mask[outmask == 255] = 1
	kernel = np.ones((3, 3), np.uint8)
	mask_erod = cv2.dilate(mj_seg, kernel, iterations=4) #using dia as erod since the color
	h, w = mask.shape
	combine_mask = np.zeros((h, w), np.uint8)
	for ii in range(h):
		for j in range(w):
			if mask[ii][j] == 1:
				combine_mask[ii][j] = 1
			else:
				if mask_erod[ii][j] == 0:
					combine_mask[ii][j] = 3
				else:
					combine_mask[ii][j] = 0

	mask, bgdModel, fgdModel = cv2.grabCut(img, combine_mask, None, bgdModel, fgdModel, 7, cv2.GC_INIT_WITH_MASK)
	mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
	img = img * mask[:, :, np.newaxis]
	if save_fl ==1:
		target = "output"
		plt.imsave(target + "/Grab" + str(K) + ".jpg", img)
		plt.imsave(target + "/Grab_mask" + str(K) + ".jpg", mask)
		plt.imsave(target + "/Grab_mjmask" + str(K) + ".jpg", mj_seg)
	return mask, mj_seg

def cal_score(inmask,insimu):
	h,w = inmask.shape
	score = 0
	sim_score = 0
	mask_score = 0
	for x in range(h):
		for y in range(w):

			if insimu[x][y] == 0:
				#print(inmask[x][y])
				sim_score +=1
			if inmask[x][y] == 1:
				mask_score +=1
			if insimu[x][y] == 0 and inmask[x][y] == 1:
				score += 1
	return score

def overlap_image(realim,mj,realgray,i,target):
	h, w = realim.shape
	combine_mask = np.zeros((h, w, 3), np.uint8)

	for ii in range(h):
		for j in range(w):
			if realim[ii][j] == 1:
				combine_mask[ii][j][0] = 255
			if mj[ii][j] == 0:
				combine_mask[ii][j][1] = 255
			combine_mask[ii][j][2] = realgray[ii][j]

	cv2.imshow('My Image2', combine_mask)
	plt.imsave(target + "/contour{}.jpg".format(i), combine_mask)

def seg_process(d_pos,d_rot, filename, sim,target, imag_save_fl = 0,rotation_fl =0,indata=[]):
	#print(indata)
	if len(indata) != 0:
		sim.reset_camera(indata)
	bb = read_ang_data_v(filename)
	FING1 = 70
	FING2 = 72
	FING3 = 73
	d_p = [] # x - right, y + down, z + zoom in
	d_rp = []
	if rotation_fl == 1:
		d_rp = d_rot
		sim.moving_camera(d_p, d_rp)
	elif rotation_fl == 0:
		d_p = d_pos
		sim.moving_camera(d_p, d_rp)

	#d_p = [-0.01, 0.01, 0.01]
	score = []

	for i in range(len(bb)):
		ang = bb[i].copy()
		ang.append(FING1)
		ang.append(FING2)
		ang.append(FING3)
		ang = np.array(ang) / (180 / math.pi)
		ang[0] = -1 * ang[0] + math.pi / 2
		ang[1] = math.pi - ang[1]  # -1+pi
		ang[2] = ang[2]
		ang[3] = ang[3]
		ang[4] = ang[4]
		ang[5] = ang[5]
		ang[6] = ang[6]
		ori_mask = sim.run_mujoco(ang, fl=1)
		kernel = np.ones((3, 3), np.uint8)
		kernel2 = np.ones((3, 3), np.uint8)
		mask_dia, real_dia, real_data = img_dia(target, i, ori_mask, kernel, kernel2)
		mask_yuv, real_yuv = RGB2YUV(target, mask_dia, real_dia)
		mask_l, real_l = Kmeanclus(target, mask_yuv, real_yuv, 4)
		grab_mask = cluster_matching_mask_g(target, mask_l, real_l, 4, i)
		realimg = cv2.imread(target + '/real{}.jpg'.format(i))
		realim, mj = grab_cuting(target, i, ori_mask,realimg,nwmask=grab_mask,save_fl = 1)
		realgray = cv2.cvtColor(realimg, cv2.COLOR_BGR2GRAY)
		if imag_save_fl == 1:
			overlap_image(realim, mj, realgray, i,"output")
		score.append(cal_score(realim, mj))
	score_m = sum(score) / len(score)
	print(score,score_m,d_pos,d_rot)

	print(sim._viewer.cam.lookat[:].copy())
	print(sim._viewer.cam.azimuth)
	print(sim._viewer.cam.elevation)
	print(sim._viewer.cam.distance)

	return score_m*-1


def optmize_passing_func(input, filename, sim,target, rotationfl,i_fl):
	if rotationfl == True:
		return seg_process([], input, filename, sim, target, imag_save_fl=0, rotation_fl=1,indata=i_fl)
	else:
		return seg_process(input, [], filename, sim, target, imag_save_fl=0, rotation_fl=0,indata=i_fl)


def opt_main():
	sim = Kinova_MJ()
	sim.init_camera()
	sim.run_mujoco(fl=1)
	#d_pos_rot = [0, 0, 0.03, 0, 0]
	target = "testdata3"
	#filename = target+"/angle_data.txt"
	filename = target + "/angle_sequence.csv"
	print("working on")
	print(target)
	d_pos = [0, 0, 0]
	d_rot = [0, 0]
	m_list=[]
	for i in range(1):
		resetin = []
		resetin.append(sim._viewer.cam.lookat[:].copy())
		resetin.append(sim._viewer.cam.azimuth)
		resetin.append(sim._viewer.cam.elevation)
		resetin.append(sim._viewer.cam.distance)
		r_fl = 0
		print(resetin)
		xopt = optimize.fmin_powell(optmize_passing_func,d_pos,args=(filename, sim,target, r_fl,resetin),maxiter = 2,full_output = 1,retall =1)
		print('end translation')
		print(xopt)
		d_pos = xopt[0]
		score = seg_process(d_pos, [], filename, sim,target, rotation_fl=0, imag_save_fl=1, indata=resetin)
		r_fl = 1
		ropt = optimize.fmin_powell(optmize_passing_func,d_rot,args=(filename, sim,target, r_fl,resetin),maxiter = 2,full_output = 1,retall =1)
		print('end rotation')
		print(ropt)
		d_rot = ropt[0]
		score2 = seg_process([],d_rot, filename, sim,target,rotation_fl =1,imag_save_fl = 1,indata = resetin)
		m_list.append(d_pos)
		m_list.append(d_rot)
	file1 = open("m_list.txt", "w")
	file1.write(str(m_list))
	file1.close()


def manu_main(angle_data):
	sim = Kinova_MJ()
	sim.init_camera()
	sim.run_mujoco(fl=1)
	# d_pos_rot = [0, 0, 0.03, 0, 0]
	real_in = "real_image"
	filename = angle_data
	print("working...")
	#print(target)
	# d_pos_rot = [-0, 0, 0,0,0] # x - up, y + left, z + zoom in (hand)
	resetin = []
	resetin.append(sim._viewer.cam.lookat[:].copy())
	resetin.append(sim._viewer.cam.azimuth)
	resetin.append(sim._viewer.cam.elevation)
	resetin.append(sim._viewer.cam.distance)

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
	d_pos = [0,-3,0]
	d_rot = [0,0]

	_ = seg_process(d_pos,[], filename, sim,  real_in,rotation_fl =1,imag_save_fl = 1,indata = resetin)
	score = seg_process([],d_rot, filename, sim, real_in,rotation_fl =2,imag_save_fl = 1,indata = resetin)
	print(score)



def main(angle_data):
	sim = Kinova_MJ()
	sim.init_camera()
	sim.run_mujoco(fl=1)
	# d_pos_rot = [0, 0, 0.03, 0, 0]
	real_in = "real_image"
	filename = angle_data

	#target = "testdata5"
	## filename = target+"/angle_data.txt"
	#filename = target + "/angle_sequence.csv"
	print("working...")
	#print(target)
	# d_pos_rot = [-0, 0, 0,0,0] # x - up, y + left, z + zoom in (hand)
	resetin = []
	resetin.append(sim._viewer.cam.lookat[:].copy())
	resetin.append(sim._viewer.cam.azimuth)
	resetin.append(sim._viewer.cam.elevation)
	resetin.append(sim._viewer.cam.distance)

	d_pos = [0, 0, 0]
	d_rot = [0, 0]
	#_ = seg_process(d_pos, [], filename, sim, target, rotation_fl=1, imag_save_fl=1, indata=resetin)
	score = seg_process([], d_rot, filename, sim, real_in, rotation_fl=2, imag_save_fl=1)
	print(score)

# score = seg_process(d_pos,d_rot, filename, sim,imag_save_fl = 1)

# dirsetting = 0.01* np.ones((5, 5))
# dirsetting = 1 * np.ones((5, 5))
###
# xopt = optimize.fmin_powell(seg_process,d_pos_rot,args=(filename, sim),full_output = 1,retall =1,direc = dirsetting)

if __name__ == '__main__':

	a = sys.argv[1]
	file_path = "output"
	try:
		os.makedirs(file_path)
	except FileExistsError:
		pass
	print(a)
	main(a)






