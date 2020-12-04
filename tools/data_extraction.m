clc
clear

bag = rosbag('test1.bag');
arm_angel = select(bag,'Topic','/j2s7s300_driver/out/joint_angles');
msgStructs = readMessages(arm_angel,'DataFormat','struct');

joint1 = cellfun(@(m) double(m.Joint1),msgStructs);
joint2 = cellfun(@(m) double(m.Joint2),msgStructs);
joint3 = cellfun(@(m) double(m.Joint3),msgStructs);
joint4 = cellfun(@(m) double(m.Joint4),msgStructs);
joint5 = cellfun(@(m) double(m.Joint5),msgStructs);
joint6 = cellfun(@(m) double(m.Joint6),msgStructs);
joint7 = cellfun(@(m) double(m.Joint7),msgStructs);


angle_data = [joint1,joint2,joint3,joint4,joint5,joint6,joint7];
writematrix(angle_data,'angle_data.txt') 