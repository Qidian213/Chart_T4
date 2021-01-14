import os
import cv2
import numpy as np

# return shape: (2, height, width)
def generate_constant_paf(shape, joint_from, joint_to, paf_width=4):
    joint_distance = np.linalg.norm(joint_to - joint_from)
    unit_vector = (joint_to - joint_from) / joint_distance 
    rad = np.pi / 2
    
    rot_matrix = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
    vertical_unit_vector = np.dot(rot_matrix, unit_vector) 
    
    grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
    grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose() 
    
    horizontal_inner_product = unit_vector[0] * (grid_x - joint_from[0]) + unit_vector[1] * (grid_y - joint_from[1])
    horizontal_paf_flag = (0 <= horizontal_inner_product) & (horizontal_inner_product <= joint_distance)

    vertical_inner_product = vertical_unit_vector[0] * (grid_x - joint_from[0]) + vertical_unit_vector[1] * (grid_y - joint_from[1])
    vertical_paf_flag = np.abs(vertical_inner_product) <= paf_width 

    paf_flag = horizontal_paf_flag & vertical_paf_flag

    constant_paf = np.stack((paf_flag, paf_flag)) * np.broadcast_to(unit_vector, shape[:-1] + [2,]).transpose(2, 0, 1)

    return constant_paf

# return shape: (2, height, width)
def generate_paf(shape, joint_from, joint_to, paf_width=4):
    joint_distance = np.linalg.norm(joint_to - joint_from)
    unit_vector = (joint_to - joint_from) / joint_distance 
    rad = np.pi / 2
    
    rot_matrix = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
    vertical_unit_vector = np.dot(rot_matrix, unit_vector) 
    
    grid_x = np.tile(np.arange(shape[0]), (shape[1], 1))
    grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose() 
    
    horizontal_inner_product = unit_vector[0] * (grid_x - joint_from[0]) + unit_vector[1] * (grid_y - joint_from[1])
    horizontal_paf_flag = (0 <= horizontal_inner_product) & (horizontal_inner_product <= joint_distance)

    vertical_inner_product = vertical_unit_vector[0] * (grid_x - joint_from[0]) + vertical_unit_vector[1] * (grid_y - joint_from[1])
    vertical_paf_flag = np.abs(vertical_inner_product) <= paf_width 

    paf_flag = horizontal_paf_flag & vertical_paf_flag

    constant_paf = np.stack((paf_flag, paf_flag)) #* np.broadcast_to(unit_vector, shape[:-1] + [2,]).transpose(2, 0, 1)

    return constant_paf
    
tag_offset = np.zeros((2, 800, 800), np.float32)

constant_paf = generate_paf([800,800,3], np.array([200,200]), np.array([500,200]))

tag_offset += constant_paf

cv2.imwrite('pafx.jpg', tag_offset[0]*255)
cv2.imwrite('pafy.jpg', tag_offset[1]*255)