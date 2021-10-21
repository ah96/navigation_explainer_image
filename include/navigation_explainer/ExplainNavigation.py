#!/usr/bin/env python3

# lime image - custom implementation
from navigation_explainer import lime_image

# for managing data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# for calculations
import math
import copy

# for running ROS local_planner C++ node
import shlex
from psutil import Popen
import rospy

# for managing image segmentation
from skimage import *
from skimage.color import gray2rgb
from skimage.segmentation import mark_boundaries, felzenszwalb, slic, quickshift
from skimage.segmentation._slic import (_slic_cython, _enforce_label_connectivity_cython)
from skimage.segmentation.slic_superpixels import _get_grid_centroids, _get_mask_centroids
from skimage.measure import regionprops

# important global variables
perturb_hide_color_value = 50

class ExplainRobotNavigation:

    def __init__(self, explanation_alg, cmd_vel, odom, plan, global_plan, local_plan, current_goal, local_costmap_data,
                 local_costmap_info, amcl_pose, tf_odom_map, tf_map_odom, map_data, map_info, num_of_first_rows_to_delete, footprints, costmap_size):
        print('Constructor starting\n')

        # save variables as class variables
        self.explanation_alg = explanation_alg
        self.cmd_vel_original = cmd_vel
        self.odom = odom
        self.plan = plan
        self.global_plan = global_plan
        self.local_plan = local_plan
        self.current_goal = current_goal
        self.costmap_data = local_costmap_data
        self.costmap_info = local_costmap_info
        self.amcl_pose = amcl_pose
        self.tf_odom_map = tf_odom_map
        self.tf_map_odom = tf_map_odom
        self.map_data = map_data
        self.map_info = map_info
        self.offset = num_of_first_rows_to_delete
        self.footprints = footprints
        self.costmap_size = costmap_size
        self.num_samples = 64

        if self.explanation_alg == 'lime':
            self.explainer = lime_image.LimeImageExplainer(verbose=True)

        print('Constructor ending\n')

    def explain_instance(self, expID):
        print('explain_instance function starting\n')

        self.expID = expID
            
        self.index = self.expID

        self.printImportantInformation()

        # Get local costmap
        # Original costmap will be saved to self.local_costmap_original
        self.local_costmap_original = self.costmap_data.iloc[(self.index) * self.costmap_size:(self.index + 1) * self.costmap_size, :]

        # Make image a np.array deepcopy of local_costmap_original
        self.image = np.array(copy.deepcopy(self.local_costmap_original))

        # Turn inflated area to free space and 100s to 99s
        self.inflatedToFree()

        # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
        self.image = self.image * 1.0

        # Saving data to .csv files for C++ node - local navigation planner
        self.limeImageSaveData()

        # Use new variable in the algorithm - possible time saving
        img = copy.deepcopy(self.image)

        # my custom segmentation func
        segm_fn = 'custom_segmentation'

        if self.explanation_alg == 'lime':

            self.explanation = self.explainer.explain_instance(img, self.classifier_fn_image, hide_color=perturb_hide_color_value, num_samples=self.num_samples,
                                                                batch_size=1024, segmentation_fn=segm_fn, top_labels=10)
            
            self.temp_img, self.mask, self.exp = self.explanation.get_image_and_mask(label=0, positive_only=False,
                                                                            negative_only=False, num_features=100,
                                                                            hide_rest=False,
                                                                            min_weight=0.1)  # min_weight=0.1 - default

            #self.plotExplanationMinimal()
            self.plotExplanationMinimalFlipped()
            #self.plotExplanation()
            #self.plotExplanationFlipped()

        print('explain_instance function ending')

    def inflatedToFree(self):
        #'''
        # Turn inflated area to free space and 100s to 99s
        for i in range(0, self.image.shape[0]):
            for j in range(0, self.image.shape[1]):
                if 99 > self.image[i, j] > 0:
                    self.image[i, j] = 0
                elif self.image[i, j] == 100:
                    self.image[i, j] = 99
        #'''

    def matrixFlip(self, m, d):
        myl = np.array(m)
        if d == 'v':
            return np.flip(myl, axis=0)
        elif d == 'h':
            return np.flip(myl, axis=1)

    def quaternion_to_euler(self, x, y, z, w):
        # roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        # pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        # yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return [yaw, pitch, roll]

    def euler_to_quaternion(self, yaw, pitch, roll):
        #qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        #qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        #qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        #qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        
        return [qx, qy, qz, qw]

    def printImportantInformation(self):
        # print important information

        #'''
        print('self.explanation_alg: ', self.explanation_alg)
        print('self.expID: ', self.expID)
        print('self.offset: ', self.offset)
        print('\n')
        #'''

    def limeImageSaveData(self):
        path_core = '~/amar_ws'
        # Saving data to .csv files for C++ node - local navigation planner
        # Save footprint instance to a file
        self.footprint_tmp = self.footprints.loc[self.footprints['ID'] == self.index + self.offset]
        self.footprint_tmp = self.footprint_tmp.iloc[:, 1:]
        self.footprint_tmp.to_csv(path_core + '/src/navigation_explainer/src/tlp/Data/footprint.csv', index=False, header=False)

        # Save local plan instance to a file
        self.local_plan_tmp = self.local_plan.loc[self.local_plan['ID'] == self.index + self.offset]
        self.local_plan_tmp = self.local_plan_tmp.iloc[:, 1:]
        self.local_plan_tmp.to_csv(path_core + '/src/navigation_explainer/src/tlp/Data/local_plan.csv', index=False, header=False)

        # Save plan (from global planner) instance to a file
        self.plan_tmp = self.plan.loc[self.plan['ID'] == self.index + self.offset]
        self.plan_tmp = self.plan_tmp.iloc[:, 1:]
        self.plan_tmp.to_csv(path_core + '/src/navigation_explainer/src/tlp/Data/plan.csv', index=False, header=False)

        # Save global plan instance to a file
        self.global_plan_tmp = self.global_plan.loc[self.global_plan['ID'] == self.index + self.offset]
        self.global_plan_tmp = self.global_plan_tmp.iloc[:, 1:]
        self.global_plan_tmp.to_csv(path_core + '/src/navigation_explainer/src/tlp/Data/global_plan.csv', index=False,
                                    header=False)

        # Save costmap_info instance to file
        self.costmap_info_tmp = self.costmap_info.iloc[self.index, :]
        self.costmap_info_tmp = pd.DataFrame(self.costmap_info_tmp).transpose()
        self.costmap_info_tmp = self.costmap_info_tmp.iloc[:, 1:]
        self.costmap_info_tmp.to_csv(path_core + '/src/navigation_explainer/src/tlp/Data/costmap_info.csv', index=False,
                                     header=False)

        # Save amcl_pose instance to file
        self.amcl_pose_tmp = self.amcl_pose.iloc[self.index, :]
        self.amcl_pose_tmp = pd.DataFrame(self.amcl_pose_tmp).transpose()
        self.amcl_pose_tmp = self.amcl_pose_tmp.iloc[:, 1:]
        self.amcl_pose_tmp.to_csv(path_core + '/src/navigation_explainer/src/tlp/Data/amcl_pose.csv', index=False, header=False)

        # Save tf_odom_map instance to file
        self.tf_odom_map_tmp = self.tf_odom_map.iloc[self.index, :]
        self.tf_odom_map_tmp = pd.DataFrame(self.tf_odom_map_tmp).transpose()
        self.tf_odom_map_tmp.to_csv(path_core + '/src/navigation_explainer/src/tlp/Data/tf_odom_map.csv', index=False,
                                    header=False)

        # Save tf_map_odom instance to file
        self.tf_map_odom_tmp = self.tf_map_odom.iloc[self.index, :]
        self.tf_map_odom_tmp = pd.DataFrame(self.tf_map_odom_tmp).transpose()
        self.tf_map_odom_tmp.to_csv(path_core + '/src/navigation_explainer/src/tlp/Data/tf_map_odom.csv', index=False,
                                    header=False)

        # Save odometry instance to file
        self.odom_tmp = self.odom.iloc[self.index, :]
        self.odom_tmp = pd.DataFrame(self.odom_tmp).transpose()
        self.odom_tmp = self.odom_tmp.iloc[:, 2:]
        self.odom_tmp.to_csv(path_core + '/src/navigation_explainer/src/tlp/Data/odom.csv', index=False, header=False)

        # Take original command speed
        self.cmd_vel_original_tmp = self.cmd_vel_original.iloc[self.index, :]
        #self.cmd_vel_original_tmp = pd.DataFrame(self.cmd_vel_original_tmp).transpose()
        #self.cmd_vel_original_tmp = self.cmd_vel_original_tmp.iloc[:, 2:]

        # save costmap info to class variables
        self.localCostmapOriginX = self.costmap_info_tmp.iloc[0, 3]
        #print('self.localCostmapOriginX: ', self.localCostmapOriginX)
        self.localCostmapOriginY = self.costmap_info_tmp.iloc[0, 4]
        #print('self.localCostmapOriginY: ', self.localCostmapOriginY)
        self.localCostmapResolution = self.costmap_info_tmp.iloc[0, 0]
        #print('self.localCostmapResolution: ', self.localCostmapResolution)
        self.localCostmapHeight = self.costmap_info_tmp.iloc[0, 2]
        #print('self.localCostmapHeight: ', self.localCostmapHeight)
        self.localCostmapWidth = self.costmap_info_tmp.iloc[0, 1]
        #print('self.localCostmapWidth: ', self.localCostmapWidth)

        # save robot odometry location to class variables
        self.odom_x = self.odom_tmp.iloc[0, 0]
        # print('self.odom_x: ', self.odom_x)
        self.odom_y = self.odom_tmp.iloc[0, 1]
        # print('self.odom_y: ', self.odom_y)

        # save indices of robot's odometry location in local costmap to class variables
        self.localCostmapIndex_x_odom = int((self.odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
        # print('self.localCostmapIndex_x_odom: ', self.localCostmapIndex_x_odom)
        self.localCostmapIndex_y_odom = int((self.odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
        # print('self.localCostmapIndex_y_odom: ', self.localCostmapIndex_y_odom)

        # save indices of robot's odometry location in local costmap to lists which are class variables - suitable for plotting
        self.x_odom_index = [self.localCostmapIndex_x_odom]
        # print('self.x_odom_index: ', self.x_odom_index)
        self.y_odom_index = [self.localCostmapIndex_y_odom]
        # print('self.y_odom_index: ', self.y_odom_index)

        # save robot odometry orientation to class variables
        self.odom_z = self.odom_tmp.iloc[0, 2]
        self.odom_w = self.odom_tmp.iloc[0, 3]
        # calculate Euler angles based on orientation quaternion
        [self.yaw_odom, pitch_odom, roll_odom] = self.quaternion_to_euler(0.0, 0.0, self.odom_z, self.odom_w)
        #print('roll_odom: ', roll_odom)
        #print('pitch_odom: ', pitch_odom)
        #print('self.yaw_odom: ', self.yaw_odom)
        #[qx, qy, qz, qw] = self.euler_to_quaternion(self.yaw_odom, pitch_odom, roll_odom)
        
        # find yaw angles projections on x and y axes and save them to class variables
        self.yaw_odom_x = math.cos(self.yaw_odom)
        self.yaw_odom_y = math.sin(self.yaw_odom)

        # save indices of footprint's poses in local costmap to class variables
        self.footprint_x_list = []
        self.footprint_y_list = []
        for j in range(0, self.footprint_tmp.shape[0]):
            # print(str(self.footprint_tmp.iloc[j, 0]) + '  ' + str(self.footprint_tmp.iloc[j, 1]))
            self.footprint_x_list.append(
                int((self.footprint_tmp.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution))
            self.footprint_y_list.append(
                int((self.footprint_tmp.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution))

        # map info
        self.mapOriginX = self.map_info.iloc[0, 4]
        # print('self.mapOriginX: ', self.mapOriginX)
        self.mapOriginY = self.map_info.iloc[0, 5]
        # print('self.mapOriginY: ', self.mapOriginY)
        self.mapResolution = self.map_info.iloc[0, 1]
        # print('self.mapResolution: ', self.mapResolution)
        self.mapHeight = self.map_info.iloc[0, 3]
        # print('self.mapHeight: ', self.mapHeight)
        self.mapWidth = self.map_info.iloc[0, 2]
        # print('self.mapWidth: ', self.mapWidth)

        # robot amcl location
        self.amcl_x = self.amcl_pose_tmp.iloc[0, 0]
        # print('self.amcl_x: ', self.amcl_x)
        self.amcl_y = self.amcl_pose_tmp.iloc[0, 1]
        # print('self.amcl_y: ', self.amcl_y)

        # robot amcl orientation
        self.amcl_z = self.amcl_pose_tmp.iloc[0, 2]
        self.amcl_w = self.amcl_pose_tmp.iloc[0, 3]
        # calculate Euler angles based on orientation quaternion
        [self.yaw_amcl, pitch_amcl, roll_amcl] = self.quaternion_to_euler(0.0, 0.0, self.amcl_z, self.amcl_w)
        #print('roll_amcl: ', roll_amcl)
        #print('pitch_amcl: ', pitch_amcl)
        #print('yaw_amcl: ', self.yaw_amcl)

    def classifier_fn_image_plot(self):
        '''
        # Visualise last 10 perturbations and last 100 perturbations separately
        self.perturbations_visualization = self.sampled_instance[0][:, :, 0]
        for i in range(1, 120):
            if i == 10:
                self.perturbations_visualization_final = self.perturbations_visualization
                self.perturbations_visualization = self.sampled_instance[i][:, :, 0]
            elif i % 10 == 0 & i != 10:
                self.perturbations_visualization_final = np.concatenate((self.perturbations_visualization_final, self.perturbations_visualization), axis=0)
                self.perturbations_visualization = self.sampled_instance[i][:, :, 0]
            else:
                self.perturbations_visualization = np.concatenate((self.perturbations_visualization, self.sampled_instance[i][:, :, 0]), axis=1)
        self.perturbations_visualization_final = np.concatenate((self.perturbations_visualization_final, self.perturbations_visualization), axis=0)
        '''

        '''
        # Save perturbations as .csv file
        for i in range(0, self.sampled_instance.shape[0]):
            pd.DataFrame(self.sampled_instance[i][:, :, 0]).to_csv('~/amar_ws/perturbation_' + str(i) + '.csv', index=False, header=False)
        '''


        #'''
        # indices of transformed plan's poses in local costmap
        self.transformed_plan_x_list = []
        self.transformed_plan_y_list = []
        for j in range(0, self.transformed_plan.shape[0]):
            self.transformed_plan_x_list.append(int((self.transformed_plan.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution))
            self.transformed_plan_y_list.append(int((self.transformed_plan.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution))
        # print('i: ', i)
        # print('self.transformed_plan_x_list.size(): ', len(self.transformed_plan_x_list))
        # print('self.transformed_plan_y_list.size(): ', len(self.transformed_plan_y_list))

        # plot every perturbation
        for i in range(0, self.sampled_instance.shape[0]):

            # save current perturbation as .csv file
            #pd.DataFrame(self.sampled_instance[i][:, :, 0]).to_csv('perturbation_' + str(i) + '.csv', index=False, header=False)

            # plot perturbed local costmap
            plt.imshow(self.sampled_instance[i][:, :, 0])

            # indices of local plan's poses in local costmap
            self.local_plan_x_list = []
            self.local_plan_y_list = []
            for j in range(0, self.local_plans.shape[0]):
                if self.local_plans.iloc[j, -1] == i:
                    index_x = int((self.local_plans.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                    index_y = int((self.local_plans.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)
                    self.local_plan_x_list.append(index_x)
                    self.local_plan_y_list.append(index_y)
                    '''
                    [yaw, pitch, roll] = self.quaternion_to_euler(0.0, 0.0, self.local_plans.iloc[j, 2], self.local_plans.iloc[j, 3])
                    yaw_x = math.cos(yaw)
                    yaw_y = math.sin(yaw)
                    plt.quiver(index_x, index_y, yaw_x, yaw_y, color='white')
                    '''
            # print('i: ', i)
            # print('self.local_plan_x_list.size(): ', len(self.local_plan_x_list))
            # print('self.local_plan_y_list.size(): ', len(self.local_plan_y_list))

            # plot transformed plan
            plt.scatter(self.transformed_plan_x_list, self.transformed_plan_y_list, c='blue', marker='x')

            # plot footprint
            plt.scatter(self.footprint_x_list, self.footprint_y_list, c='green', marker='x')

            '''
            # plot footprints for first five points of local plan
            # indices of local plan's poses in local costmap
            self.footprint_local_plan_x_list = []
            self.footprint_local_plan_y_list = []
            self.footprint_local_plan_x_list_angle = []
            self.footprint_local_plan_y_list_angle = []
            for j in range(0, self.local_plans.shape[0]):
                if self.local_plans.iloc[j, -1] == i:
                    for k in range(6, 7):

                        [yaw, pitch, roll] = self.quaternion_to_euler(0.0, 0.0, self.local_plans.iloc[j + k, 2], self.local_plans.iloc[j + k, 3])
                        sin_th = math.sin(yaw)
                        cos_th = math.cos(yaw)

                        for l in range(0, self.footprint_tmp.shape[0]):
                            x_new = self.footprint_tmp.iloc[l, 0] + (self.local_plans.iloc[j + k, 0] - self.odom_x)
                            y_new = self.footprint_tmp.iloc[l, 1] + (self.local_plans.iloc[j + k, 1] - self.odom_y)
                            self.footprint_local_plan_x_list.append(int((x_new - self.localCostmapOriginX) / self.localCostmapResolution))
                            self.footprint_local_plan_y_list.append(int((y_new - self.localCostmapOriginY) / self.localCostmapResolution))

                            x_new = self.local_plans.iloc[j + k, 0] + (self.footprint_tmp.iloc[l, 0] - self.odom_x) * sin_th + (self.footprint_tmp.iloc[l, 1] - self.odom_y) * cos_th
                            y_new = self.local_plans.iloc[j + k, 1] - (self.footprint_tmp.iloc[l, 0] - self.odom_x) * cos_th + (self.footprint_tmp.iloc[l, 1] - self.odom_y) * sin_th
                            self.footprint_local_plan_x_list_angle.append(int((x_new - self.localCostmapOriginX) / self.localCostmapResolution))
                            self.footprint_local_plan_y_list_angle.append(int((y_new - self.localCostmapOriginY) / self.localCostmapResolution))
                    break
            #print('self.footprint_local_plan_x_list: ', self.footprint_local_plan_x_list)
            #print('self.footprint_local_plan_y_list: ', self.footprint_local_plan_y_list)
            # plot footprints
            plt.scatter(self.footprint_local_plan_x_list, self.footprint_local_plan_y_list, c='green', marker='x')
            plt.scatter(self.footprint_local_plan_x_list_angle, self.footprint_local_plan_y_list_angle, c='white', marker='x')
            '''

            # plot local plan
            plt.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='x')

            # plot local plan last point
            #if len(self.local_plan_x_list) != 0:
            #    plt.scatter([self.local_plan_x_list[-1]], [self.local_plan_y_list[-1]], c='black', marker='x')

            # plot robot's location and orientation
            plt.scatter(self.x_odom_index, self.y_odom_index, c='white', marker='o')
            plt.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='white')

            # plot command velocities as text
            plt.text(0.0, -5.0, 'lin_x=' + str(round(self.cmd_vel_perturb.iloc[i, 0], 2)) + ', ' + 'ang_z=' + str(round(self.cmd_vel_perturb.iloc[i, 2], 2)))

            # save figure
            plt.savefig('perturbation_' + str(i) + '.png')
            plt.clf()
        #'''

    def classifier_fn_image(self, sampled_instance):

        print('classifier_fn_image started')

        # sampled_instance info
        #print('sampled_instance: ', sampled_instance)
        #print('sampled_instance.shape: ', sampled_instance.shape)
        
        #'''
        # I will use channel 0 from sampled_instance as actual perturbed data
        # Perturbed pixel intensity is perturb_hide_color_value
        # Convert perturbed free space to obstacle (99), and perturbed obstacles to free space (0) in all perturbations
        for i in range(0, sampled_instance.shape[0]):
            for j in range(0, sampled_instance[i].shape[0]):
                for k in range(0, sampled_instance[i].shape[1]):
                    if sampled_instance[i][j, k, 0] == perturb_hide_color_value:
                        if self.image[j, k] == 0:
                            sampled_instance[i][j, k, 0] = 99
                            #print('free space')
                        elif self.image[j, k] == 99:
                            sampled_instance[i][j, k, 0] = 0
                            #print('obstacle')
        #'''

        #'''
        # Save perturbed costmap_data to file for C++ node
        #sampled_instance = sampled_instance.astype(int)
        self.costmap_tmp = pd.DataFrame(sampled_instance[0][:, :, 0])
        for i in range(1, sampled_instance.shape[0]):
            self.costmap_tmp = pd.concat([self.costmap_tmp, pd.DataFrame(sampled_instance[i][:, :, 0])], join='outer', axis=0, sort=False)
        self.costmap_tmp.to_csv('~/amar_ws/src/navigation_explainer/src/tlp/Data/costmap_data.csv', index=False, header=False)
        # print('self.costmap_tmp.shape: ', self.costmap_tmp.shape)
        # self.costmap_tmp.to_csv('~/amar_ws/costmap_data.csv', index=False, header=False)
        #'''

        print('starting C++ node')

        # start perturbed_node_image ROS C++ node
        #Popen(shlex.split('rosrun teb_local_planner perturb_node_image'))
        Popen(shlex.split('rosrun navigation_explainer pni'))

        # Wait until perturb_node_image is finished
        rospy.wait_for_service("/pni/finished")
        #print('perturb_node_image finishedn from python')

        # kill ROS node
        Popen(shlex.split('rosnode kill /pni'))

        #rospy.sleep(1)

        print('C++ node ended')

        # load command velocities
        self.cmd_vel_perturb = pd.read_csv('~/amar_ws/src/navigation_explainer/src/tlp/Data/cmd_vel.csv')
        '''
        print('self.cmd_vel: ', self.cmd_vel_perturb)
        print('self.cmd_vel.shape: ', self.cmd_vel_perturb.shape)
        '''

        # load local plans
        self.local_plans = pd.read_csv('~/amar_ws/src/navigation_explainer/src/tlp/Data/local_plans.csv')
        '''
        print('self.local_plans: ', self.local_plans)
        print('self.local_plans.shape: ', self.local_plans.shape)
        '''

        # load transformed plan
        self.transformed_plan = pd.read_csv('~/amar_ws/src/navigation_explainer/src/tlp/Data/transformed_plan.csv')
        '''
        print('self.transformed_plan: ', self.transformed_plan)
        print('self.transformed_plan.shape: ', self.transformed_plan.shape)
        '''

        # only needed for classifier_fn_image_plot() function
        self.sampled_instance = sampled_instance

        # plot perturbation of local costmap
        #self.classifier_fn_image_plot()

  
        import math

        # fill the list of local plan coordinates
        #print('self.local_plan_tmp: ', self.local_plan_tmp)
        print('\nself.local_plan_tmp.shape[0]: ', self.local_plan_tmp.shape[0])
        local_plan_xs_orig = []
        local_plan_ys_orig = []
        for i in range(0, self.local_plan_tmp.shape[0]):
            x_temp = int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

            if 0 <= x_temp <= 159 and 0 <= y_temp <= 159:
                local_plan_xs_orig.append(x_temp)
                local_plan_ys_orig.append(y_temp)

        print('\nlen(local_plan_xs): ', len(local_plan_xs_orig))
        print('len(local_plan_ys): ', len(local_plan_ys_orig))

        # fill the list of transformed plan coordinates
        print('\nself.transformed_plan.shape[0]: ', self.transformed_plan.shape[0])
        transformed_plan_xs = []
        transformed_plan_ys = []
        closest_to_robot_index = -100
        min_diff = 100
        for i in range(0, self.transformed_plan.shape[0]):
            x_temp = int((self.transformed_plan.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            y_temp = int((self.transformed_plan.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

            if 0 <= x_temp <= 159 and 0 <= y_temp <= 159:
                transformed_plan_xs.append(x_temp)
                transformed_plan_ys.append(y_temp)

                diff = math.sqrt( (transformed_plan_xs[-1]-local_plan_xs_orig[0])**2 + (transformed_plan_ys[-1]-local_plan_ys_orig[0])**2 )
                if diff < min_diff:
                    min_diff = diff
                    closest_to_robot_index = len(transformed_plan_xs) - 1

        print('\nlen(transformed_plan_xs): ', len(transformed_plan_xs))
        print('len(transformed_plan_ys): ', len(transformed_plan_ys))

        print('\nclosest_to_robot_index_original: ', closest_to_robot_index)                    

        transformed_plan_xs = np.array(transformed_plan_xs)
        transformed_plan_ys = np.array(transformed_plan_ys)


        # a new way of deviation logic
        local_plan_gap_threshold = 11
        deviation_threshold = 12

        local_plan_original_gap = False
        local_plan_gaps = []
        diff = 0
        for j in range(0, len(local_plan_xs_orig) - 1):
            diff = math.sqrt( (local_plan_xs_orig[j]-local_plan_xs_orig[j+1])**2 + (local_plan_ys_orig[j]-local_plan_ys_orig[j+1])**2 )
            local_plan_gaps.append(diff)
        if max(local_plan_gaps) > local_plan_gap_threshold:
            local_plan_original_gap = True

        print('\nmax(local_plan_original_gaps): ', max(local_plan_gaps))      
        
        if local_plan_original_gap == True:
            deviation_type = 'deviation'
        else:
            diff_x = 0
            diff_y = 0
            
            real_deviation = False
            for j in range( 0, len(local_plan_xs_orig)):
                #diffs = []
                deviation_local = True  
                for k in range(0, len(transformed_plan_xs)):
                    diff_x = (local_plan_xs_orig[j] - transformed_plan_xs[k]) ** 2
                    diff_y = (local_plan_ys_orig[j] - transformed_plan_ys[k]) ** 2
                    diff = math.sqrt(diff_x + diff_y)
                    #diffs.append(diff)
                    if diff <= deviation_threshold:
                        deviation_local = False
                        break
                #print('j = ', j)
                #print('min(diffs): ', min(diffs))    
                if deviation_local == True:
                    real_deviation = True
                    break
            
            if real_deviation == False:
                deviation_type = 'no_deviation'
            else:
                deviation_type = 'deviation'    
                    

        print('\nself.expID: ', self.expID)

        #'''
        print('\ndeviation_type: ', deviation_type)
        print('local_plan_gap: ', local_plan_original_gap)
        print('command velocities original - lin_x: ' + str(self.cmd_vel_original_tmp.iloc[0]) + ', ang_z: ' + str(self.cmd_vel_original_tmp.iloc[1]) + '\n')
        #'''
        

        # deviation of local plan from global plan
        self.local_plan_deviation = pd.DataFrame(-1.0, index=np.arange(sampled_instance.shape[0]), columns=['deviate'])
        #print('self.local_plan_deviation: ', self.local_plan_deviation)


        # fill in deviation dataframe
        for i in range(0, sampled_instance.shape[0]):
            # test if there is local plan
            local_plan_xs = []
            local_plan_ys = []
            local_plan_found = False
            for j in range(0, self.local_plans.shape[0]):
                if self.local_plans.iloc[j, -1] == i:
                    x_temp = int((self.local_plans.iloc[j, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
                    y_temp = int((self.local_plans.iloc[j, 1] - self.localCostmapOriginY) / self.localCostmapResolution)

                    if 0 <= x_temp <= 159 and 0 <= y_temp <= 159:
                        local_plan_xs.append(x_temp)
                        local_plan_ys.append(y_temp)
                        local_plan_found = True
                
            if local_plan_found == True:
                # test if there is local plan gap
                diff = 0
                local_plan_gap = False
                local_plan_gaps = []
                for j in range(0, len(local_plan_xs) - 1):
                    diff = math.sqrt( (local_plan_xs[j]-local_plan_xs[j+1])**2 + (local_plan_ys[j]-local_plan_ys[j+1])**2 )
                    local_plan_gaps.append(diff)
                
                if max(local_plan_gaps) > local_plan_gap_threshold:
                    local_plan_gap = True
                
                if local_plan_gap == True:
                    if deviation_type == 'no_deviation':
                        self.local_plan_deviation.iloc[i, 0] = 0.0
                    elif deviation_type == 'deviation':
                        self.local_plan_deviation.iloc[i, 0] = 1.0
                else:
                    diff_x = 0
                    diff_y = 0
                    real_deviation = False
                    for j in range( 0, len(local_plan_xs)): #min(len(local_plan_xs), len(local_plan_xs_orig)) ):
                        #diffs = []
                        deviation_local = True  
                        for k in range(0, len(transformed_plan_xs)):
                            diff_x = (local_plan_xs[j] - transformed_plan_xs[k]) ** 2
                            diff_y = (local_plan_ys[j] - transformed_plan_ys[k]) ** 2
                            diff = math.sqrt(diff_x + diff_y)
                            #diffs.append(diff)
                            if diff <= deviation_threshold:
                                deviation_local = False
                                break
                        #print('j = ', j)
                        #print('min(diffs): ', min(diffs))
                        if deviation_local == True:
                            real_deviation = True
                            break
                    
                    if deviation_type == 'no_deviation':
                        if real_deviation == False:
                            self.local_plan_deviation.iloc[i, 0] = 1.0
                        else:    
                            self.local_plan_deviation.iloc[i, 0] = 0.0
                    elif deviation_type == 'deviation':
                        if real_deviation == False:
                            self.local_plan_deviation.iloc[i, 0] = 0.0
                        else:    
                            self.local_plan_deviation.iloc[i, 0] = 1.0                
            else:
                if deviation_type == 'no_deviation':         
                    self.local_plan_deviation.iloc[i, 0] = 0.0
                elif deviation_type == 'deviation':
                    self.local_plan_deviation.iloc[i, 0] = 1.0
            
            '''
            print('\ni: ', i)
            print('local plan found: ', local_plan_found)
            if local_plan_found == True:
                print('local plan length: ', len(local_plan_xs))
                print('local_plan_gap: ', local_plan_gap)
                print('max(local_plan_gaps): ', max(local_plan_gaps))
                if local_plan_gap == False:
                    print('deviation: ', real_deviation)
                    #print('minimal diff: ', min(diffs))
            print('command velocities perturbed - lin_x: ' + str(self.cmd_vel_perturb.iloc[i, 0]) + ', ang_z: ' + str(self.cmd_vel_perturb.iloc[i, 2]))
            '''
        
        #print('self.local_plan_deviation: ', self.local_plan_deviation)
        #'''
        
        self.cmd_vel_perturb['deviate'] = self.local_plan_deviation
        
        # if more outputs wanted
        more_outputs = False
        if more_outputs == True:
            # classification
            stop_list = []
            linear_positive_list = []
            rotate_left_list = []
            rotate_right_list = []
            ahead_straight_list = []
            ahead_left_list = []
            ahead_right_list = []
            for i in range(0, self.cmd_vel_perturb.shape[0]):
                if abs(self.cmd_vel_perturb.iloc[i, 0]) < 0.01:
                    stop_list.append(1.0)
                else:
                    stop_list.append(0.0)

                if self.cmd_vel_perturb.iloc[i, 0] > 0.01:
                    linear_positive_list.append(1.0)
                else:
                    linear_positive_list.append(0.0)

                if self.cmd_vel_perturb.iloc[i, 2] > 0.0:
                    rotate_left_list.append(1.0)
                else:
                    rotate_left_list.append(0.0)

                if self.cmd_vel_perturb.iloc[i, 2] < 0.0:
                    rotate_right_list.append(1.0)
                else:
                    rotate_right_list.append(0.0)

                if self.cmd_vel_perturb.iloc[i, 0] > 0.01 and abs(self.cmd_vel_perturb.iloc[i, 2]) < 0.01:
                    ahead_straight_list.append(1.0)
                else:
                    ahead_straight_list.append(0.0)

                if self.cmd_vel_perturb.iloc[i, 0] > 0.01 and self.cmd_vel_perturb.iloc[i, 2] > 0.0:
                    ahead_left_list.append(1.0)
                else:
                    ahead_left_list.append(0.0)

                if self.cmd_vel_perturb.iloc[i, 0] > 0.01 and self.cmd_vel_perturb.iloc[i, 2] < 0.0:
                    ahead_right_list.append(1.0)
                else:
                    ahead_right_list.append(0.0)

            self.cmd_vel_perturb['stop'] = pd.DataFrame(np.array(stop_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['stop'])
            self.cmd_vel_perturb['linear_positive'] = pd.DataFrame(np.array(linear_positive_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['linear_positive'])
            self.cmd_vel_perturb['rotate_left'] = pd.DataFrame(np.array(rotate_left_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_left'])
            self.cmd_vel_perturb['rotate_right'] = pd.DataFrame(np.array(rotate_right_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['rotate_right'])
            self.cmd_vel_perturb['ahead_straight'] = pd.DataFrame(np.array(ahead_straight_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['ahead_straight'])
            self.cmd_vel_perturb['ahead_left'] = pd.DataFrame(np.array(ahead_left_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['ahead_left'])
            self.cmd_vel_perturb['ahead_right'] = pd.DataFrame(np.array(ahead_right_list), index=np.arange(self.cmd_vel_perturb.shape[0]), columns=['ahead_right'])

            '''
            print('self.cmd_vel_perturb: ', self.cmd_vel_perturb)
            print('self.local_plan_deviation: ', self.local_plan_deviation)
            print('stop_list: ', stop_list)
            print('linear_positive_list: ', linear_positive_list)
            print('rotate_left_list: ', rotate_left_list)
            print('rotate_right_list: ', rotate_right_list)
            print('ahead_straight_list: ', ahead_straight_list)
            print('ahead_left_list: ', ahead_left_list)
            print('ahead_right_list: ', ahead_right_list)
            '''

        print('classifier_fn_image ended')

        return np.array(self.cmd_vel_perturb.iloc[:, 3:])

    def plotExplanationMinimal(self):
        # make a deepcopy of an image
        img_ = copy.deepcopy(self.image)

        # Turn gray image to rgb image
        rgb = gray2rgb(img_)

        # import needed libraries
        from skimage.segmentation import slic
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt

        # show original image
        img = rgb[:, :, 0]

        # segments_1 - good obstacles
        # Find segments_1
        segments_1 = slic(rgb, n_segments=6, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)

        # Find segments_2
        segments_2 = slic(rgb, n_segments=10, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.3, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)

        segments_unique_2 = np.unique(segments_2)
        # Creating segments using segments_1 and segments_2
        #'''
        # Add/Sum segments_1 and segments_2
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                if img[i, j] == 0.0:  # and segments_1[i, j] != segments_unique_1[max_index_1]:
                    segments_1[i, j] = segments_2[i, j] + segments_unique_2.shape[0]
                else:
                    segments_1[i, j] = 2 * segments_1[i, j] + 2 * segments_unique_2.shape[0]
        #'''
        segments_unique = np.unique(segments_1)
        # Get nice segments' numbering
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                for k in range(0, segments_unique.shape[0]):
                    if segments_1[i, j] == segments_unique[k]:
                        segments_1[i, j] = k + 1

        # plot segments with centroids and labels/weights
        #print('segments_1.shape: ', segments_1.shape)
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments_1, aspect='auto')
        
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            ax.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            # plt.text(centers[i][0], centers[i][1], str(v))
            # '''
            # printing/plotting explanation weights
            for j in range(0, len(self.exp)):
                if self.exp[j][0] == i:
                    # print('i: ', i)
                    # print('j: ', j)
                    # print('self.exp[j][0]: ', self.exp[j][0])
                    # print('self.exp[j][1]: ', self.exp[j][1])
                    # print('v: ', v)
                    # print('\n')
                    ax.text(centers[i][0], centers[i][1], str(round(self.exp[j][1], 4)))  # str(round(self.exp[j][1],4)) #str(v))
                    break
            # '''
            i = i + 1

        # Save segments with nice numbering as a picture
        fig.savefig('testSegmentation_segments.png')
        fig.clf()


        # plot explanation
        fig = plt.figure(frameon=True)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 0.95])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # indices of local plan's poses in local costmap
        self.local_plan_x_list = []
        self.local_plan_y_list = []
        for i in range(1, self.local_plan_tmp.shape[0]):
            self.local_plan_x_list.append(int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution))
            self.local_plan_y_list.append(int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution))
        
        # save indices of robot's odometry location in local costmap to class variables
        self.localCostmapIndex_x_odom = int((self.odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
        # print('self.localCostmapIndex_x_odom: ', self.localCostmapIndex_x_odom)
        self.localCostmapIndex_y_odom = int((self.odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
        # print('self.localCostmapIndex_y_odom: ', self.localCostmapIndex_y_odom)
        
        # transform global plan from /map to /odom frame
        # rotation matrix
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(
            [self.tf_map_odom_tmp.iloc[0, 3], self.tf_map_odom_tmp.iloc[0, 4], self.tf_map_odom_tmp.iloc[0, 5],
             self.tf_map_odom_tmp.iloc[0, 6]])
        # print('r: ', r.as_matrix())
        r_array = np.asarray(r.as_matrix())
        # print('r_array: ', r_array)
        # print('r_array.shape: ', r_array.shape)
        
        # translation vector
        t = np.array(
            [self.tf_map_odom_tmp.iloc[0, 0], self.tf_map_odom_tmp.iloc[0, 1], self.tf_map_odom_tmp.iloc[0, 2]])
        # print('t: ', t)
        plan_tmp_tmp = copy.deepcopy(self.global_plan_tmp)
        for i in range(0, self.global_plan_tmp.shape[0]):
            p = np.array(
                [self.global_plan_tmp.iloc[i, 0], self.global_plan_tmp.iloc[i, 1], self.global_plan_tmp.iloc[i, 2]])
            # print('p: ', p)
            pnew = p.dot(r_array) + t
            # print('pnew: ', pnew)
            plan_tmp_tmp.iloc[i, 0] = pnew[0]
            plan_tmp_tmp.iloc[i, 1] = pnew[1]
            plan_tmp_tmp.iloc[i, 2] = pnew[2]
    
        # Get coordinates of the global plan in the local costmap
        self.plan_x_list = []
        self.plan_y_list = []
        for i in range(0, plan_tmp_tmp.shape[0], 3):
            x_temp = int((plan_tmp_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            if 0 <= x_temp <= 159:
                self.plan_x_list.append(x_temp)
                self.plan_y_list.append(
                    int((plan_tmp_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution))
                    
        plt.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')

        # plot robots' local plan
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='o')
                
        # save indices of robot's odometry location in local costmap to lists which are class variables - suitable for plotting
        self.x_odom_index = [self.localCostmapIndex_x_odom]
        # print('self.x_odom_index: ', self.x_odom_index)
        self.y_odom_index = [self.localCostmapIndex_y_odom]
        # print('self.y_odom_index: ', self.y_odom_index)

        # save robot odometry orientation to class variables
        self.odom_z = self.odom_tmp.iloc[0, 2] # minus je upitan
        self.odom_w = self.odom_tmp.iloc[0, 3]
        # calculate Euler angles based on orientation quaternion
        [self.yaw_odom, pitch_odom, roll_odom] = self.quaternion_to_euler(0.0, 0.0, self.odom_z, self.odom_w)
        # print('roll_odom: ', roll_odom)
        # print('pitch_odom: ', pitch_odom)
        # print('self.yaw_odom: ', self.yaw_odom)
        # find yaw angles projections on x and y axes and save them to class variables
        self.yaw_odom_x = math.cos(self.yaw_odom)
        self.yaw_odom_y = math.sin(self.yaw_odom)
        
        # plot robots' location and orientation
        #print('self.x_odom_index: ', self.x_odom_index)
        #print('self.y_odom_index: ', self.y_odom_index)
        ax.scatter(self.x_odom_index, self.y_odom_index, c='black', marker='o')
        ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='black')

        
        # '''
        # plot explanation
        # print('self.cmd_vel_original_tmp.shape: ', self.cmd_vel_original_tmp.shape)
        ax.text(0.0, -4.0, 'lin_x=' + str(round(self.cmd_vel_original_tmp.iloc[0], 2)) + ', ' + 'ang_z=' + str(
            round(self.cmd_vel_original_tmp.iloc[1], 2)))
        
        marked_boundaries = mark_boundaries(self.temp_img / 2 + 0.5, self.mask, color=(1, 1, 0), outline_color=(0, 0, 0), mode='outer', background_label=0)
        
        # marked_boundaries_flipped = marked_boundaries_flipped[0:125, 0:100]
        ax.imshow(marked_boundaries, aspect='auto')  # , aspect='auto')
        fig.savefig('explanation.png', transparent=False)
        fig.clf()
        #fig.close()

    def plotExplanationMinimalFlipped(self):
        # make a deepcopy of an image
        img_ = copy.deepcopy(self.image)

        # Turn gray image to rgb image
        rgb = gray2rgb(img_)

        # import needed libraries
        from skimage.segmentation import slic
        from skimage.measure import regionprops
        import matplotlib.pyplot as plt

        # show original image
        img = rgb[:, :, 0]

        # segments_1 - good obstacles
        # Find segments_1
        segments_1 = slic(rgb, n_segments=6, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)

        # Find segments_2
        segments_2 = slic(rgb, n_segments=10, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.3, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)

        segments_unique_2 = np.unique(segments_2)
        # Creating segments using segments_1 and segments_2
        #'''
        # Add/Sum segments_1 and segments_2
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                if img[i, j] == 0.0:  # and segments_1[i, j] != segments_unique_1[max_index_1]:
                    segments_1[i, j] = segments_2[i, j] + segments_unique_2.shape[0]
                else:
                    segments_1[i, j] = 2 * segments_1[i, j] + 2 * segments_unique_2.shape[0]
        #'''
        segments_unique = np.unique(segments_1)
        # Get nice segments' numbering
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                for k in range(0, segments_unique.shape[0]):
                    if segments_1[i, j] == segments_unique[k]:
                        segments_1[i, j] = k + 1

        # plot segments with centroids and labels/weights
        #print('segments_1.shape: ', segments_1.shape)
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.matrixFlip(segments_1, 'h').astype('uint8'), aspect='auto')
        
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            ax.scatter(160 - centers[i][0], centers[i][1], c='white', marker='o')
            # plt.text(centers[i][0], centers[i][1], str(v))
            # '''
            # printing/plotting explanation weights
            for j in range(0, len(self.exp)):
                if self.exp[j][0] == i:
                    # print('i: ', i)
                    # print('j: ', j)
                    # print('self.exp[j][0]: ', self.exp[j][0])
                    # print('self.exp[j][1]: ', self.exp[j][1])
                    # print('v: ', v)
                    # print('\n')
                    ax.text(160 - centers[i][0], centers[i][1], str(round(self.exp[j][1], 4)))  # str(round(self.exp[j][1],4)) #str(v))
                    break
            # '''
            i = i + 1

        # Save segments with nice numbering as a picture
        fig.savefig('flipped_testSegmentation_segments.png')
        fig.clf()


        # plot explanation
        fig = plt.figure(frameon=True)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # indices of local plan's poses in local costmap
        self.local_plan_x_list = []
        self.local_plan_y_list = []
        for i in range(1, self.local_plan_tmp.shape[0]):
            self.local_plan_x_list.append(160 - int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution))
            self.local_plan_y_list.append(int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution))

        # save indices of robot's odometry location in local costmap to class variables
        self.localCostmapIndex_x_odom = 160 - int((self.odom_x - self.localCostmapOriginX) / self.localCostmapResolution)
        # print('self.localCostmapIndex_x_odom: ', self.localCostmapIndex_x_odom)
        self.localCostmapIndex_y_odom = int((self.odom_y - self.localCostmapOriginY) / self.localCostmapResolution)
        # print('self.localCostmapIndex_y_odom: ', self.localCostmapIndex_y_odom)
        
        # transform global plan from /map to /odom frame
        # rotation matrix
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(
            [self.tf_map_odom_tmp.iloc[0, 3], self.tf_map_odom_tmp.iloc[0, 4], self.tf_map_odom_tmp.iloc[0, 5],
             self.tf_map_odom_tmp.iloc[0, 6]])
        # print('r: ', r.as_matrix())
        r_array = np.asarray(r.as_matrix())
        # print('r_array: ', r_array)
        # print('r_array.shape: ', r_array.shape)
        
        # translation vector
        t = np.array(
            [self.tf_map_odom_tmp.iloc[0, 0], self.tf_map_odom_tmp.iloc[0, 1], self.tf_map_odom_tmp.iloc[0, 2]])
        # print('t: ', t)
        plan_tmp_tmp = copy.deepcopy(self.global_plan_tmp)
        for i in range(0, self.global_plan_tmp.shape[0]):
            p = np.array(
                [self.global_plan_tmp.iloc[i, 0], self.global_plan_tmp.iloc[i, 1], self.global_plan_tmp.iloc[i, 2]])
            # print('p: ', p)
            pnew = p.dot(r_array) + t
            # print('pnew: ', pnew)
            plan_tmp_tmp.iloc[i, 0] = pnew[0]
            plan_tmp_tmp.iloc[i, 1] = pnew[1]
            plan_tmp_tmp.iloc[i, 2] = pnew[2]
        
        # Get coordinates of the global plan in the local costmap
        # '''
        self.plan_x_list = []
        self.plan_y_list = []
        for i in range(0, plan_tmp_tmp.shape[0], 3):
            x_temp = 160 - int((plan_tmp_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            if 0 <= x_temp <= 159:
                self.plan_x_list.append(x_temp)
                self.plan_y_list.append(
                    int((plan_tmp_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution))
        plt.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')

        # plot robots' local plan
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='o')
        
        # save indices of robot's odometry location in local costmap to lists which are class variables - suitable for plotting
        self.x_odom_index = [self.localCostmapIndex_x_odom]
        # print('self.x_odom_index: ', self.x_odom_index)
        self.y_odom_index = [self.localCostmapIndex_y_odom]
        # print('self.y_odom_index: ', self.y_odom_index)

        # save robot odometry orientation to class variables
        self.odom_z = self.odom_tmp.iloc[0, 2] # minus je upitan
        self.odom_w = self.odom_tmp.iloc[0, 3]
        # calculate Euler angles based on orientation quaternion
        [self.yaw_odom, pitch_odom, roll_odom] = self.quaternion_to_euler(0.0, 0.0, self.odom_z, self.odom_w)
        yaw_sign = math.copysign(1, self.yaw_odom)
        self.yaw_odom = yaw_sign * (math.pi - abs(self.yaw_odom)) 
        # print('roll_odom: ', roll_odom)
        # print('pitch_odom: ', pitch_odom)
        # print('self.yaw_odom: ', self.yaw_odom)
        # find yaw angles projections on x and y axes and save them to class variables
        self.yaw_odom_x = math.cos(self.yaw_odom)
        self.yaw_odom_y = math.sin(self.yaw_odom)

        # plot robots' location and orientation
        #print('self.x_odom_index: ', self.x_odom_index)
        #print('self.y_odom_index: ', self.y_odom_index)
        ax.scatter(self.x_odom_index, self.y_odom_index, c='black', marker='o')
        ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='black')

        
        # '''
        # plot explanation
        # print('self.cmd_vel_original_tmp.shape: ', self.cmd_vel_original_tmp.shape)
        #ax.text(0.0, -4.0, 'lin_x=' + str(round(self.cmd_vel_original_tmp.iloc[0], 2)) + ', ' + 'ang_z=' + str(
        #    round(self.cmd_vel_original_tmp.iloc[1], 2)))
        
        marked_boundaries = mark_boundaries(self.temp_img / 2 + 0.5, self.mask, color=(1, 1, 0), outline_color=(0, 0, 0), mode='outer', background_label=0)
        marked_boundaries_flipped = self.matrixFlip(marked_boundaries, 'h')
        
        # marked_boundaries_flipped = marked_boundaries_flipped[0:125, 0:100]
        ax.imshow(marked_boundaries_flipped.astype('float64'), aspect='auto')  # , aspect='auto')
        fig.savefig('flipped_explanation.png', transparent=False)
        fig.clf()
        #fig.close()

    def plotExplanation(self):
        print('plotExplanation starts')

        # plot local costmap
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # plot robot odometry location
        ax.scatter(self.x_odom_index, self.y_odom_index, c='blue', marker='o')
        # robot's odometry orientation
        plt.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='white')


        # indices of local plan's poses in local costmap
        self.local_plan_x_list = []
        self.local_plan_y_list = []
        # print('self.local_plan_tmp.shape: ', self.local_plan_tmp.shape)
        for i in range(0, self.local_plan_tmp.shape[0]):
            if int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution) >= 0 and int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution) <= 160: 
                self.local_plan_x_list.append(
                    int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution))
                self.local_plan_y_list.append(
                    int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution))
        plt.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='x')
        
        # plot local costmap
        ax.imshow(self.image.astype(np.uint8), aspect='auto')
        fig.savefig('straight_local_costmap.png')
        fig.clf()


        # plot global map
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # indices of robot's odometry location in map
        self.mapIndex_x_amcl = int((self.amcl_x - self.mapOriginX) / self.mapResolution)
        # print('self.mapIndex_x_amcl: ', self.mapIndex_x_amcl)
        self.mapIndex_y_amcl = int((self.amcl_y - self.mapOriginY) / self.mapResolution)
        # print('self.mapIndex_y_amcl: ', self.mapIndex_y_amcl)

        # indices of robot's amcl location in map in a list - suitable for plotting
        self.x_amcl_index = [self.mapIndex_x_amcl]
        self.y_amcl_index = [self.mapIndex_y_amcl]
        ax.scatter(self.x_amcl_index, self.y_amcl_index, c='yellow', marker='o')

        self.yaw_amcl_x = math.cos(self.yaw_amcl)
        self.yaw_amcl_y = math.sin(self.yaw_amcl)
        ax.quiver(self.x_amcl_index, self.y_amcl_index, self.yaw_amcl_x, self.yaw_amcl_y, color='white')

        # plan from global planner
        self.plan_x_list = []
        self.plan_y_list = []
        for i in range(19, self.plan_tmp.shape[0], 20):
            self.plan_x_list.append(int((self.plan_tmp.iloc[i, 0] - self.mapOriginX) / self.mapResolution))
            self.plan_y_list.append(int((self.plan_tmp.iloc[i, 1] - self.mapOriginY) / self.mapResolution))
        ax.scatter(self.plan_x_list, self.plan_y_list, c='red', marker='<')

        # global plan from teb algorithm
        self.global_plan_x_list = []
        self.global_plan_y_list = []
        for i in range(19, self.global_plan_tmp.shape[0], 20):
            self.global_plan_x_list.append(
                int((self.global_plan_tmp.iloc[i, 0] - self.mapOriginX) / self.mapResolution))
            self.global_plan_y_list.append(
                int((self.global_plan_tmp.iloc[i, 1] - self.mapOriginY) / self.mapResolution))
        ax.scatter(self.global_plan_x_list, self.global_plan_y_list, c='yellow', marker='>')

        # plot robot's location in the map
        x_map = int((self.amcl_x - self.mapOriginX) / self.mapResolution)
        y_map = int((self.amcl_y - self.mapOriginY) / self.mapResolution)
        ax.scatter(x_map, y_map, c='red', marker='o')

        # plot map, fill -1 with 100
        map_tmp = self.map_data
        for i in range(0, map_tmp.shape[0]):
            for j in range(0, map_tmp.shape[1]):
                if map_tmp.iloc[i, j] == -1:
                    map_tmp.iloc[i, j] = 100
        ax.imshow(map_tmp.astype(np.uint8), aspect='auto')
        fig.savefig('straight_map.png')
        fig.clf()

        # plot image_temp
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.temp_img, aspect='auto')
        fig.savefig('straight_temp_img.png')
        fig.clf()

        # plot mask
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.mask.astype(np.uint8), aspect='auto')
        fig.savefig('straight_mask.png')
        fig.clf()

        # plot explanation - srediti nekad granice
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(mark_boundaries(self.temp_img / 2 + 0.5, self.mask), aspect='auto')
        ax.scatter(self.x_odom_index, self.y_odom_index, c='blue', marker='o')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='x')
        ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='white')
        fig.savefig('straight_explanation.png')
        fig.clf()

        print('plotExplanation ends')

    def plotExplanationFlipped(self):
        print('plotExplanationFlipped starts')

        # make a deepcopy of local costmap and flip it
        local_costmap = copy.deepcopy(self.image)
        local_costmap_flipped = self.matrixFlip(local_costmap, 'h')

        # plot flipped local costmap
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # plot robot odometry location
        self.x_odom_index = [160 - self.localCostmapIndex_x_odom]
        ax.scatter(self.x_odom_index, self.y_odom_index, c='blue', marker='o')

        # indices of local plan's poses in local costmap
        self.local_plan_x_list = []
        self.local_plan_y_list = []
        for i in range(1, self.local_plan_tmp.shape[0]):
            self.local_plan_x_list.append(
                160 - int((self.local_plan_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution))
            self.local_plan_y_list.append(
                int((self.local_plan_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution))
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='x')

        # find yaw angles projections on x and y axes and save them to class variables
        yaw_sign = math.copysign(1, self.yaw_odom)
        self.yaw_odom = yaw_sign * (math.pi - abs(self.yaw_odom))
        self.yaw_odom_x = math.cos(self.yaw_odom)
        self.yaw_odom_y = math.sin(self.yaw_odom)
        # robot's odometry orientation
        ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='white')

        ax.imshow(local_costmap_flipped, aspect='auto')
        fig.savefig('flipped_local_costmap.png')
        fig.clf()


        # plot flipped map
        fig = plt.figure(frameon=False)
        w = 4.0
        h = 6.0
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # indices of robot's odometry location in map
        self.mapIndex_x_amcl = 160 - int((self.amcl_x - self.mapOriginX) / self.mapResolution)
        # print('self.mapIndex_x_amcl: ', self.mapIndex_x_amcl)
        self.mapIndex_y_amcl = int((self.amcl_y - self.mapOriginY) / self.mapResolution)
        # print('self.mapIndex_y_amcl: ', self.mapIndex_y_amcl)

        # indices of robot's amcl location in map in a list - suitable for plotting
        self.x_amcl_index = [self.mapIndex_x_amcl]
        self.y_amcl_index = [self.mapIndex_y_amcl]
        ax.scatter(self.x_amcl_index, self.y_amcl_index, c='yellow', marker='o')

        # find yaw angles projections on x and y axes and save them to class variables
        yaw_sign = math.copysign(1, self.yaw_amcl)
        self.yaw_amcl = yaw_sign * (math.pi - abs(self.yaw_amcl))
        self.yaw_amcl_x = math.cos(self.yaw_amcl)
        self.yaw_amcl_y = math.sin(self.yaw_amcl)
        ax.quiver(self.x_amcl_index, self.y_amcl_index, self.yaw_amcl_x, self.yaw_amcl_y, color='white')

        # plan from global planner
        self.plan_x_list = []
        self.plan_y_list = []
        for i in range(19, self.plan_tmp.shape[0], 20):
            self.plan_x_list.append(160 - int((self.plan_tmp.iloc[i, 0] - self.mapOriginX) / self.mapResolution))
            self.plan_y_list.append(int((self.plan_tmp.iloc[i, 1] - self.mapOriginY) / self.mapResolution))
        ax.scatter(self.plan_x_list, self.plan_y_list, c='red', marker='<')

        # global plan from teb algorithm
        self.global_plan_x_list = []
        self.global_plan_y_list = []
        for i in range(19, self.global_plan_tmp.shape[0], 20):
            self.global_plan_x_list.append(
                160 - int((self.global_plan_tmp.iloc[i, 0] - self.mapOriginX) / self.mapResolution))
            self.global_plan_y_list.append(
                int((self.global_plan_tmp.iloc[i, 1] - self.mapOriginY) / self.mapResolution))
        ax.scatter(self.global_plan_x_list, self.global_plan_y_list, c='yellow', marker='>')

        # plot robot's location in the map
        x_map = 160 - int((self.amcl_x - self.mapOriginX) / self.mapResolution)
        y_map = int((self.amcl_y - self.mapOriginY) / self.mapResolution)
        ax.scatter(x_map, y_map, c='red', marker='o')

        # plot map, fill -1 with 100
        map_tmp = self.map_data
        for i in range(0, map_tmp.shape[0]):
            for j in range(0, map_tmp.shape[1]):
                if map_tmp.iloc[i, j] == -1:
                    map_tmp.iloc[i, j] = 100
        map_tmp_flipped = self.matrixFlip(map_tmp, 'h')
        ax.imshow(map_tmp_flipped, aspect='auto')
        fig.savefig('flipped_map.png')
        fig.clf()

        # plot image_temp
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        temp_img_flipped = self.matrixFlip(self.temp_img, 'h')
        ax.imshow(temp_img_flipped, aspect='auto')
        fig.savefig('flipped_temp_img.png')
        fig.clf()
        #pd.DataFrame(self.temp_img_flipped[:,:,0]).to_csv('~/amar_ws/temp_img_flipped.csv', index=False, header=False)

        # plot mask
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        mask_flipped = self.matrixFlip(self.mask, 'h')
        plt.imshow(mask_flipped, aspect='auto')
        plt.savefig('flipped_mask.png')
        plt.clf()
        #pd.DataFrame(self.mask_flipped[:,:,0]).to_csv('~/amar_ws/mask_flipped.csv', index=False, header=False)

        # plot explanation
        fig = plt.figure(frameon=False)
        w = 1.6
        h = 1.6
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # plot robots' location, orientation and local plan
        ax.scatter(self.x_odom_index, self.y_odom_index, c='black', marker='o')
        ax.quiver(self.x_odom_index, self.y_odom_index, self.yaw_odom_x, self.yaw_odom_y, color='black')
        ax.scatter(self.local_plan_x_list, self.local_plan_y_list, c='red', marker='o')
        
        # transform global plan from /map to /odom frame
        # rotation matrix
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(
            [self.tf_map_odom_tmp.iloc[0, 3], self.tf_map_odom_tmp.iloc[0, 4], self.tf_map_odom_tmp.iloc[0, 5],
             self.tf_map_odom_tmp.iloc[0, 6]])
        #print('r: ', r.as_matrix())
        r_array = np.asarray(r.as_matrix())
        #print('r_array: ', r_array)
        #print('r_array.shape: ', r_array.shape)
        
        # translation vector
        t = np.array(
            [self.tf_map_odom_tmp.iloc[0, 0], self.tf_map_odom_tmp.iloc[0, 1], self.tf_map_odom_tmp.iloc[0, 2]])
        #print('t: ', t)
        plan_tmp_tmp = copy.deepcopy(self.global_plan_tmp)
        for i in range(0, self.global_plan_tmp.shape[0]):
            p = np.array([self.global_plan_tmp.iloc[i, 0], self.global_plan_tmp.iloc[i, 1], self.global_plan_tmp.iloc[i, 2]])
            # print('p: ', p)
            pnew = p.dot(r_array) + t
            # print('pnew: ', pnew)
            plan_tmp_tmp.iloc[i, 0] = pnew[0]
            plan_tmp_tmp.iloc[i, 1] = pnew[1]
            plan_tmp_tmp.iloc[i, 2] = pnew[2]
        
        # Get coordinates of the global plan in the local costmap
        #'''
        self.plan_x_list = []
        self.plan_y_list = []
        for i in range(0, plan_tmp_tmp.shape[0], 3):
            x_temp = 160 - int((plan_tmp_tmp.iloc[i, 0] - self.localCostmapOriginX) / self.localCostmapResolution)
            if 0 <= x_temp <= 159:
                self.plan_x_list.append(x_temp)
                self.plan_y_list.append(int((plan_tmp_tmp.iloc[i, 1] - self.localCostmapOriginY) / self.localCostmapResolution))
        plt.scatter(self.plan_x_list, self.plan_y_list, c='blue', marker='o')
        
        #'''
        # plot explanation
        #print('self.cmd_vel_original_tmp.shape: ', self.cmd_vel_original_tmp.shape)
        ax.text(0.0, -5.0, 'lin_x=' + str(round(self.cmd_vel_original_tmp.iloc[0], 2)) + ', ' + 'ang_z=' + str(round(self.cmd_vel_original_tmp.iloc[1], 2)))
        marked_boundaries = mark_boundaries(self.temp_img / 2 + 0.5, self.mask)
        marked_boundaries_flipped = self.matrixFlip(marked_boundaries, 'h')
        
        # marked_boundaries_flipped = marked_boundaries_flipped[0:125, 0:100]
        ax.imshow(marked_boundaries_flipped) #, aspect='auto')
        fig.savefig('flipped_explanation.png', transparent=False)
        fig.clf()

        print('plotExplanationFlipped ends')

    
    
    def testSegmentation(self, expID):

        print('Test segmentation function beginning')

        self.expID = expID
            
        self.index = self.expID

        # Get local costmap
        # Original costmap will be saved to self.local_costmap_original
        self.local_costmap_original = self.costmap_data.iloc[(self.index) * self.costmap_size:(self.index + 1) * self.costmap_size, :]

        # Make image a np.array deepcopy of local_costmap_original
        self.image = np.array(copy.deepcopy(self.local_costmap_original))

        # Turn inflated area to free space and 100s to 99s
        self.inflatedToFree()

        # Turn every local costmap entry from int to float, so the segmentation algorithm works okay
        self.image = self.image * 1.0

        # Make image a np.array deepcopy of local_costmap_original
        img_ = copy.deepcopy(self.image)

        '''
        # Save local costmap as gray image
        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img_, aspect='auto')
        fig.savefig('testSegmentation_image_gray.png')
        fig.clf()
        '''

        # Turn gray image to rgb image
        rgb = gray2rgb(img_)

        # Save local costmap as rgb image
        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(rgb, aspect='auto')
        fig.savefig('testSegmentation_image_rgb.png')
        fig.clf()

        # Superpixel segmentation with skimage functions

        # felzenszwalb
        #segments = felzenszwalb(rgb, scale=100, sigma=5, min_size=30, multichannel=True)
        #segments = felzenszwalb(rgb, scale=1, sigma=0.8, min_size=20, multichannel=True)  # default

        # quickshift
        #segments = quickshift(rgb, ratio=0.0001, kernel_size=8, max_dist=10, return_tree=False, sigma=0.0, convert2lab=True, random_seed=42)
        #segments = quickshift(rgb, ratio=1.0, kernel_size=5, max_dist=10, return_tree=False, sigma=0, convert2lab=True, random_seed=42) # default

        # slic
        #segments = slic(rgb, n_segments=6, compactness=10.0, max_iter=1000, sigma=0, spacing=None, multichannel=True, convert2lab=None, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=5, slic_zero=False, start_label=None, mask=None)
        #segments = slic(rgb, n_segments=100, compactness=10.0, max_iter=1000, sigma=0, spacing=None, multichannel=True, convert2lab=None, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3, slic_zero=False, start_label=None, mask=None) # default

        # Turn segments gray image to rgb image
        #segments_rgb = gray2rgb(segments)

        # Generate segments - superpixels with my slic function
        segments = self.mySlicTest(rgb)

        # Save segments to .csv file
        path_core = '~/amar_ws'
        #pd.DataFrame(segments).to_csv(path_core + '/segments_segmentation_test.csv', index=False, header=False)

        print('Test segmentation function ending')

    def mySlicTest(self, img_rgb):

        print('mySlic for testSegmentation starts')

        # import needed libraries
        #from skimage.segmentation import slic
        #from skimage.measure import regionprops
        import matplotlib.pyplot as plt

        
        # show original image
        img = img_rgb[:, :, 0]
        
        '''
        # Save picture for segmenting
        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img, aspect='auto')
        fig.savefig('testSegmentation_img.png')
        fig.clf()
        '''

        # segments_1 - good obstacles
        # Find segments_1
        segments_1 = slic(img_rgb, n_segments=6, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.01, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)
        '''
        # plot segments_1 with centroids and labels
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            plt.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            plt.text(centers[i][0], centers[i][1], str(v))
            i = i + 1
        
        # Save segments_1 as a picture
        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments_1, aspect='auto')
        fig.savefig('testSegmentation_segments_1.png')
        fig.clf()
        '''

        # find segments_unique_1
        segments_unique_1 = np.unique(segments_1)
        print('segments_unique_1: ', segments_unique_1)
        print('segments_unique_1.shape: ', segments_unique_1.shape)

        # Find segments_2
        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        segments_2 = slic(img_rgb, n_segments=10, compactness=100.0, max_iter=1000, sigma=0, spacing=None,
                          multichannel=True, convert2lab=True,
                          enforce_connectivity=True, min_size_factor=0.3, max_size_factor=5, slic_zero=False,
                          start_label=1, mask=None)
        '''
        # find segments_unique_2
        segments_unique_2 = np.unique(segments_2)
        print('segments_unique_2: ', segments_unique_2)
        print('segments_unique_2.shape: ', segments_unique_2.shape)
        # make obstacles on segments_2 nice - not needed
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                if img[i, j] == 99:
                   segments_2[i, j] = segments_1[i, j] + segments_unique_2.shape[0]
        '''

        '''
        # plot segments_2 with centroids and labels
        regions = regionprops(segments_2)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            ax.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            ax.text(centers[i][0], centers[i][1], str(v))
            i = i + 1
        # Save segments_2 as a picture
        ax.imshow(segments_2, aspect='auto')
        fig.savefig('testSegmentation_segments_2.png')
        fig.clf()
        '''

        # find segments_unique_2
        segments_unique_2 = np.unique(segments_2)
        print('segments_unique_2: ', segments_unique_2)
        print('segments_unique_2.shape: ', segments_unique_2.shape)

        # Creating segments using segments_1 and segments_2
        #'''
        # Add/Sum segments_1 and segments_2
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                if img[i, j] == 0.0:  # and segments_1[i, j] != segments_unique_1[max_index_1]:
                    segments_1[i, j] = segments_2[i, j] + segments_unique_2.shape[0]
                else:
                    segments_1[i, j] = 2 * segments_1[i, j] + 2 * segments_unique_2.shape[0]
        #'''
        
        '''
        # plot segments with centroids and labels/weights
        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments_1, aspect='auto')
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            ax.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            ax.text(centers[i][0], centers[i][1], str(v))
            i = i + 1
        # Save segments as a picture before nice segment numbering
        fig.savefig('testSegmentation_segments_beforeNiceNumbering.png')
        fig.clf()
        '''
        
        # find segments_unique before nice segment numbering
        segments_unique = np.unique(segments_1)
        print('segments_unique: ', segments_unique)
        print('segments_unique.shape: ', segments_unique.shape)

        # Get nice segments' numbering
        for i in range(0, segments_1.shape[0]):
            for j in range(0, segments_1.shape[1]):
                for k in range(0, segments_unique.shape[0]):
                    if segments_1[i, j] == segments_unique[k]:
                        segments_1[i, j] = k + 1
        
        # find segments_unique after nice segment numbering
        segments_unique = np.unique(segments_1)
        print('segments_unique (with nice numbering): ', segments_unique)
        print('segments_unique.shape (with nice numbering): ', segments_unique.shape)
        
        # plot segments with centroids and labels/weights
        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(segments_1, aspect='auto')
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            ax.scatter(centers[i][0], centers[i][1], c='white', marker='o')
            #plt.text(centers[i][0], centers[i][1], str(v))
            #'''
            # printing/plotting explanation weights
            for j in range(0, len(self.exp)):
                if self.exp[j][0] == i:
                    #print('i: ', i)
                    #print('j: ', j)
                    #print('self.exp[j][0]: ', self.exp[j][0])
                    #print('self.exp[j][1]: ', self.exp[j][1])
                    #print('v: ', v)
                    #print('\n')
                    ax.text(centers[i][0], centers[i][1], str(round(self.exp[j][1],4)))  #str(round(self.exp[j][1],4)) #str(v))
                    break
            #'''
            i = i + 1
        
        # Save segments with nice numbering as a picture
        fig.savefig('testSegmentation_segments.png')
        fig.clf()

        # plot segments with centroids and labels/weights
        fig = plt.figure(frameon=False)
        w = 4.8
        h = 4.8
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.matrixFlip(segments_1, 'h'), aspect='auto')
        regions = regionprops(segments_1)
        centers = []
        i = 0
        for props in regions:
            v = props.label  # value of label
            cx, cy = props.centroid  # centroid coordinates
            centers.append([cy, cx])
            ax.scatter(160 - centers[i][0], centers[i][1], c='white', marker='o')
            # plt.text(centers[i][0], centers[i][1], str(v))
            # '''
            # printing/plotting explanation weights
            for j in range(0, len(self.exp)):
                if self.exp[j][0] == i:
                    # print('i: ', i)
                    # print('j: ', j)
                    # print('self.exp[j][0]: ', self.exp[j][0])
                    # print('self.exp[j][1]: ', self.exp[j][1])
                    # print('v: ', v)
                    # print('\n')
                    ax.text(160 - centers[i][0], centers[i][1], str(round(self.exp[j][1], 4)))  # str(round(self.exp[j][1],4)) #str(v))
                    break
            # '''
            i = i + 1
        # Save segments with nice numbering as a picture
        fig.savefig('flipped_testSegmentation_segments.png')
        fig.clf()

        # print explanation
        #print('self.exp: ', self.exp)
        #print('len(self.exp): ', len(self.exp))

        print('mySlic for testSegmentation ends')

        return segments_1

