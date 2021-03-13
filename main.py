#!/usr/bin/env python
import time
import datetime
import os
import random
import threading
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2
import torch
from torch.autograd import Variable
from robot import Robot
from trainer import Trainer
from logger import Logger
import utils
from tensorboardX import SummaryWriter
from skimage.morphology.convex_hull import convex_hull_image
from scipy.ndimage.morphology import binary_dilation

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(args):
    # --------------- General setup options ---------------
    obj_mesh_dir = os.path.abspath(args.obj_mesh_dir) # Directory containing 3D mesh files (.obj) of objects to be added to simulation
    num_obj = args.num_obj # Number of objects to add to simulation
    heightmap_resolution = args.heightmap_resolution # Meters per pixel of heightmap
    random_seed = args.random_seed
    force_cpu = args.force_cpu

    # --------------- Workspace setting -----------------
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    
    # ------------- Training stage options -------------
    stage = args.stage
    max_push_episode_length = args.max_push_episode_length
    grasp_reward_threshold = args.grasp_reward_threshold
    alternating_training = args.alternating_training
    cooperative_training = args.cooperative_training
    
    # -------------- Q-learning tricks ----------------
    future_reward_discount = args.future_reward_discount
    experience_replay = args.experience_replay # Use prioritized experience replay?
    heuristic_bootstrap = args.heuristic_bootstrap # Use handcrafted grasping algorithm when grasping fails too many times in a row?
    explore_rate_decay = args.explore_rate_decay

    # -------------- Testing options --------------
    is_testing = args.is_testing
    max_test_trials = args.max_test_trials # Maximum number of test runs per case/scenario
    test_preset_cases = args.test_preset_cases
    test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None
    random_scene_testing = args.random_scene_testing  

    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot # Load pre-trained snapshot of model?
    snapshot_file = os.path.abspath(args.snapshot_file) if load_snapshot else None
    continue_logging = args.continue_logging # Continue logging from previous session
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
    save_visualizations = args.save_visualizations # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True
    
    # ------- Goal-conditioned grasp net explore options ---------
    load_explore_snapshot = args.load_explore_snapshot # Load pre-trained snapshot of model?
    explore_snapshot_file = os.path.abspath(args.explore_snapshot_file) if load_explore_snapshot else None

    # Set random seed
    np.random.seed(random_seed)

    # ------- Goal-conditioned option ------------
    goal_conditioned = args.goal_conditioned
    grasp_goal_conditioned = args.grasp_goal_conditioned
    
    # Initialize pick-and-place system (camera and robot)
    robot = Robot(stage, obj_mesh_dir, num_obj, workspace_limits,
                  is_testing, test_preset_cases, test_preset_file, 
                  goal_conditioned, grasp_goal_conditioned)

    # Initialize trainer
    trainer = Trainer(stage, future_reward_discount,
                      is_testing, load_snapshot, snapshot_file, 
                      load_explore_snapshot, explore_snapshot_file,
                      alternating_training, cooperative_training,
                      force_cpu, grasp_goal_conditioned)

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory)
    logger.save_heightmap_info(workspace_limits, heightmap_resolution) # Save heightmap parameters
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    # Find last executed iteration of pre-loaded log, and load execution info and RL variables
    if continue_logging:
        trainer.preload(logger.transitions_directory)

    # Initialize episode loss
    episode_loss = 0

    # Initialize variables for heuristic bootstrapping and exploration probability
    no_change_count = [2, 2] if not is_testing else [0, 0]
    explore_prob = 0.5 if not is_testing else 0.0
    grasp_explore_prob = 0.8 if not is_testing else 0.0
    grasp_explore = args.grasp_explore

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'executing_action' : False,
                          'primitive_action' : None,
                          'best_pix_ind' : None,
                          'push_success' : False,
                          'grasp_success' : False,
                          'grasp_reward' : 0,
                          'improved_grasp_reward' : 0,
                          'push_step' : 0, # plus one after pushing
                          'goal_obj_idx' : 0,
                          'goal_catched' : 0,
                          'border_occupy_ratio' : 1,
                          'decreased_occupy_ratio' : 0,
                          'restart_scene' : 0,
                          'episode' : 0, # episode number
                          'new_episode_flag' : 0, # flag to begin a new episode
                          'episode_grasp_reward' : 0, # grasp reward at the end of a episode
                          'episode_ratio_of_grasp_to_push' : 0, # ratio of grasp to push at the end of a episode
                          'episode_improved_grasp_reward' : 0,
                          'push_predictions': np.zeros((16, 224, 224), dtype=float),
                          'grasp_predictions' : np.zeros((16,224,224),dtype=float)} # average of improved grasp reward of a episode

    # --------- Initialize nonlocal variables -----------
    nonlocal_variables['goal_obj_idx'] = args.goal_obj_idx

    if continue_logging:
        if not is_testing:
            nonlocal_variables['episode'] = trainer.episode_log[len(trainer.episode_log) - 1][0]
        if stage == 'push_only':
            # Initialize nonlocal memory
            nonlocal_variables['push_step'] = trainer.push_step_log[trainer.iteration - 1][0]
            nonlocal_variables['episode_improved_grasp_reward'] = trainer.episode_improved_grasp_reward_log[len(trainer.episode_improved_grasp_reward_log) - 1][0]

    # ------ Tensorboard setting --------
    timestamp = time.time()
    timestamp_value = datetime.datetime.fromtimestamp(timestamp)
    tensor_logging_directory = args.tensor_logging_directory
    if continue_logging:
        writer = SummaryWriter(os.path.join(tensor_logging_directory, logging_directory.split('/')[-1]))
    else:
        writer = SummaryWriter(os.path.join(tensor_logging_directory, timestamp_value.strftime('%Y-%m-%d.%H:%M:%S')))

    # Parallel thread to process network output and execute actions
    def process_actions():
        while True:
            if nonlocal_variables['executing_action']:
                push_predictions = nonlocal_variables['push_predictions']
                grasp_predictions = nonlocal_variables['grasp_predictions']

                # For goal-conditioned case, cut grasp predictions with goal mask
                if goal_conditioned:
                    if is_testing and not random_scene_testing:
                        obj_contour = robot.get_test_obj_mask(nonlocal_variables['goal_obj_idx'])
                    else:
                        obj_contour = robot.get_obj_mask(nonlocal_variables['goal_obj_idx'])
                    mask = np.zeros(color_heightmap.shape[:2], np.uint8)
                    mask = utils.get_goal_mask(obj_contour, mask, workspace_limits, heightmap_resolution)
                    obj_grasp_prediction = np.multiply(grasp_predictions, mask)
                    grasp_predictions = obj_grasp_prediction / 255

                    # if goal object is pushed completely out of scene, restart scene
                    if np.max(obj_contour[:, 0]) < 0 or np.max(obj_contour[:, 1]) < 0 or np.min(obj_contour[:, 0]) > 224 or np.min(obj_contour[:, 1]) > 224:
                        nonlocal_variables['new_episode_flag'] = 1
                        robot.restart_sim()
                        robot.add_objects()
                        if is_testing: # If at end of test run, re-load original weights (before test run)
                            trainer.model.load_state_dict(torch.load(snapshot_file))
                        trainer.clearance_log.append([trainer.iteration])
                        logger.write_to_log('clearance', trainer.clearance_log)
                        if is_testing and len(trainer.clearance_log) >= max_test_trials:
                            exit_called = True # Exit after training thread (backprop and saving labels)
                        continue
                    
                best_push_conf = np.max(push_predictions)
                best_grasp_conf = np.max(grasp_predictions)
                nonlocal_variables['grasp_reward'] = best_grasp_conf

                print('Primitive confidence scores: %f (push), %f (grasp)' % (best_push_conf, best_grasp_conf))
                
                # ------- Action selection --------
                explore_actions = False

                if stage == 'grasp_only':                   
                    nonlocal_variables['primitive_action'] = 'grasp'
                    trainer.episode_log.append([nonlocal_variables['episode']])
                    logger.write_to_log('episode', trainer.episode_log)

                elif stage == 'push_only':
                    nonlocal_variables['primitive_action'] = 'push'
                    trainer.episode_log.append([nonlocal_variables['episode']])
                    logger.write_to_log('episode', trainer.episode_log)
                    # executing grasp if grasp reward exceeds reward threshold or push length exceeds max_push_episode_length
                    if best_grasp_conf > grasp_reward_threshold or nonlocal_variables['push_step'] == max_push_episode_length:
                        nonlocal_variables['primitive_action'] = 'grasp'
                        nonlocal_variables['episode_grasp_reward'] = best_grasp_conf
                        nonlocal_variables['episode_ratio_of_grasp_to_push'] = best_grasp_conf / best_push_conf

                elif stage == 'push_grasp':
                    # testing is more conservative for pushing, somewhat reduce ratio of grasping and pushing
                    nonlocal_variables['primitive_action'] = 'grasp'
                    if is_testing:
                        if not goal_conditioned and not grasp_goal_conditioned:
                            if best_push_conf > 1.5 * best_grasp_conf:
                                nonlocal_variables['primitive_action'] = 'push'
                        else:
                            print('border_occupy_ratio', nonlocal_variables['border_occupy_ratio'])
                            if random_scene_testing:
                                if best_grasp_conf < 1.5:
                                    nonlocal_variables['primitive_action'] = 'push'
                            else:
                                if nonlocal_variables['border_occupy_ratio'] > 0.3 or best_grasp_conf < 1.5:
                                    nonlocal_variables['primitive_action'] = 'push'
                    else:
                        if best_push_conf > best_grasp_conf:
                            nonlocal_variables['primitive_action'] = 'push'

                    explore_actions = np.random.uniform() < explore_prob

                    if explore_actions: # Exploitation (do best action) vs exploration (do other action)
                        print('Strategy: explore (exploration probability: %f)' % (explore_prob))
                        nonlocal_variables['primitive_action'] = 'push' if np.random.randint(0,2) == 0 else 'grasp'
                    else:
                        print('Strategy: exploit (exploration probability: %f)' % (explore_prob))

                trainer.is_exploit_log.append([0 if explore_actions else 1])
                logger.write_to_log('is-exploit', trainer.is_exploit_log)

                # --------- Generate position of selected action ----------
                # If heuristic bootstrapping is enabled: if change has not been detected more than 2 times, execute heuristic algorithm to detect grasps/pushes
                # NOTE: typically not necessary and can reduce final performance.
                if heuristic_bootstrap and nonlocal_variables['primitive_action'] == 'push' and no_change_count[0] >= 2:
                    print('Change not detected for more than two pushes. Running heuristic pushing.')
                    nonlocal_variables['best_pix_ind'] = trainer.push_heuristic(valid_depth_heightmap)
                    no_change_count[0] = 0
                    predicted_value = push_predictions[nonlocal_variables['best_pix_ind']]
                    use_heuristic = True

                elif heuristic_bootstrap and nonlocal_variables['primitive_action'] == 'grasp' and no_change_count[1] >= 2:
                    print('Change not detected for more than two grasps. Running heuristic grasping.')
                    nonlocal_variables['best_pix_ind'] = trainer.grasp_heuristic(valid_depth_heightmap)
                    no_change_count[1] = 0
                    predicted_value = grasp_predictions[nonlocal_variables['best_pix_ind']]
                    use_heuristic = True
                else:
                    use_heuristic = False

                    # Get pixel location and rotation with highest affordance prediction from heuristic algorithms (rotation, y, x)
                    if nonlocal_variables['primitive_action'] == 'push':
                        if is_testing and not random_scene_testing:
                            obj_contours = robot.get_test_obj_masks()
                            obj_number = len(robot.test_obj_mesh_files)
                        else:
                            obj_contours = robot.get_obj_masks()
                            obj_number = len(robot.obj_mesh_ind)
                            mask = 255 * np.ones(color_heightmap.shape[:2], np.uint8)
                            mask = utils.get_all_mask(obj_contours, mask, obj_number, workspace_limits, heightmap_resolution)
                            push_predictions = np.multiply(push_predictions, mask) / 255

                        nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
                        predicted_value = np.max(push_predictions)

                    elif nonlocal_variables['primitive_action'] == 'grasp':
                        if goal_conditioned:
                            if is_testing and not random_scene_testing:
                                obj_contour = robot.get_test_obj_mask(nonlocal_variables['goal_obj_idx'])
                            else:
                                obj_contour = robot.get_obj_mask(nonlocal_variables['goal_obj_idx'])
                            obj_contour[:, 0] = (obj_contour[:, 0] - workspace_limits[0][0]) / heightmap_resolution  # drop_x to pixel_dimension2
                            obj_contour[:, 1] = (obj_contour[:, 1] - workspace_limits[1][0]) / heightmap_resolution  # drop_y to pixel_dimension1
                            obj_contour = np.array(obj_contour).astype(int)
                            # if goal object is pushed completely out of scene, restart episode
                            if np.max(obj_contour[:, 0]) < 0 or np.max(obj_contour[:, 1]) < 0 or np.min(obj_contour[:, 0]) > 224 or np.min(obj_contour[:, 1]) > 224:
                                nonlocal_variables['new_episode_flag'] = 1
                                nonlocal_variables['restart_scene'] = robot.num_obj / 2

                        nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
                        predicted_value = np.max(grasp_predictions)

                trainer.use_heuristic_log.append([1 if use_heuristic else 0])
                logger.write_to_log('use-heuristic', trainer.use_heuristic_log)

                # Save predicted confidence value
                trainer.predicted_value_log.append([predicted_value])
                logger.write_to_log('predicted-value', trainer.predicted_value_log)

                # Compute 3D position of pixel
                print('Action: %s at (%d, %d, %d)' % (nonlocal_variables['primitive_action'], nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]))
                best_rotation_angle = np.deg2rad(nonlocal_variables['best_pix_ind'][0]*(360.0/trainer.model.num_rotations))
                best_pix_x = nonlocal_variables['best_pix_ind'][2]
                best_pix_y = nonlocal_variables['best_pix_ind'][1]
                # 3D position
                primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0], best_pix_y * heightmap_resolution + workspace_limits[1][0], valid_depth_heightmap[best_pix_y][best_pix_x] + workspace_limits[2][0]]
                # If pushing, adjust start position, and make sure z value is safe and not too low
                if nonlocal_variables['primitive_action'] == 'push': # or nonlocal_variables['primitive_action'] == 'place':
                    # simulation parameter
                    finger_width = 0.02
                    safe_kernel_width = int(np.round((finger_width/2)/heightmap_resolution))
                    local_region = valid_depth_heightmap[max(best_pix_y - safe_kernel_width, 0):min(best_pix_y + safe_kernel_width + 1, valid_depth_heightmap.shape[0]), max(best_pix_x - safe_kernel_width, 0):min(best_pix_x + safe_kernel_width + 1, valid_depth_heightmap.shape[1])]
                    if local_region.size == 0:
                        safe_z_position = workspace_limits[2][0] - 0.01
                    else:
                        safe_z_position = np.max_z_position = np.max(local_region) + workspace_limits[2][0] - 0.01
                    primitive_position[2] = safe_z_position
                    print('3D z position:', primitive_position[2])

                    # Before pushing
                    if stage == 'push_only':
                        if best_grasp_conf <= grasp_reward_threshold and nonlocal_variables['push_step'] < max_push_episode_length:
                            # Get grasp reward before pushing
                            prev_img = color_heightmap
                            obj_contours = robot.get_obj_masks()
                            obj_number = len(robot.obj_mesh_ind)
                            mask_all = np.zeros(prev_img.shape[:2], np.uint8)
                            obj_grasp_predictions, mask_all = utils.get_obj_grasp_predictions(grasp_predictions, obj_contours, mask_all, prev_img, obj_number, workspace_limits, heightmap_resolution)

                            prev_single_predictions = [np.max(obj_grasp_predictions[i]) for i in range(len(obj_grasp_predictions))]
                            print('reward of grasping before pushing: ', prev_single_predictions) 

                            # Get occupy ratio before pushing
                            if goal_conditioned:
                                prev_occupy_ratio = utils.get_occupy_ratio(goal_mask_heightmap, depth_heightmap)      

                # Save executed primitive
                if nonlocal_variables['primitive_action'] == 'push':
                    trainer.executed_action_log.append([0, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]]) # 0 - push
                elif nonlocal_variables['primitive_action'] == 'grasp':
                    trainer.executed_action_log.append([1, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]]) # 1 - grasp
                logger.write_to_log('executed-action', trainer.executed_action_log)

                # Visualize executed primitive, and affordances
                if save_visualizations:
                    push_pred_vis = trainer.get_prediction_vis(push_predictions, color_heightmap, nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, push_pred_vis, 'push')
                    cv2.imwrite('visualization.push.png', push_pred_vis)
                    grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions, color_heightmap, nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, grasp_pred_vis, 'grasp')
                    cv2.imwrite('visualization.grasp.png', grasp_pred_vis)
                    push_direction_vis = trainer.get_best_push_direction_vis(nonlocal_variables['best_pix_ind'], color_heightmap)
                    logger.save_visualizations(trainer.iteration, push_direction_vis, 'best_push_direction')
                    cv2.imwrite('visualization.best_push_direction.png', push_direction_vis)

                # ------- Executing Actions ---------
                nonlocal_variables['push_success'] = False
                nonlocal_variables['grasp_success'] = False
                change_detected = False

                # Execute primitive
                if nonlocal_variables['primitive_action'] == 'push':
                    nonlocal_variables['push_success'] = robot.push(primitive_position, best_rotation_angle, workspace_limits)
                    print('Push successful: %r' % (nonlocal_variables['push_success']))
                    trainer.grasp_obj_log.append([-1])
                    logger.write_to_log('grasp-obj', trainer.grasp_obj_log) 
                    if stage == 'push_only':
                        if best_grasp_conf <= grasp_reward_threshold and nonlocal_variables['push_step'] < max_push_episode_length:
                            # Get latest RGB-D image
                            latest_color_img, latest_depth_img = robot.get_camera_data()
                            latest_depth_img = latest_depth_img * robot.cam_depth_scale # Apply depth scale from calibration

                            # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
                            latest_color_heightmap, latest_depth_heightmap = utils.get_heightmap(latest_color_img, latest_depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
                            latest_valid_depth_heightmap = latest_depth_heightmap.copy()
                            latest_valid_depth_heightmap[np.isnan(latest_valid_depth_heightmap)] = 0
                            
                            # Get goal mask heightmap 
                            if grasp_goal_conditioned or goal_conditioned:
                                if is_testing and not random_scene_testing:
                                    obj_contour = robot.get_test_obj_mask(nonlocal_variables['goal_obj_idx'])
                                else:
                                    obj_contour = robot.get_obj_mask(nonlocal_variables['goal_obj_idx'])
                                latest_goal_mask_heightmap = np.zeros(latest_color_heightmap.shape[:2], np.uint8)
                                latest_goal_mask_heightmap = utils.get_goal_mask(obj_contour, latest_goal_mask_heightmap, workspace_limits, heightmap_resolution)

                            if not grasp_goal_conditioned:
                                latest_push_predictions, latest_grasp_predictions, latest_state_feat = trainer.forward(latest_color_heightmap, latest_valid_depth_heightmap, is_volatile=True)
                            else:
                                latest_push_predictions, latest_grasp_predictions, latest_state_feat = trainer.goal_forward(latest_color_heightmap, latest_valid_depth_heightmap, latest_goal_mask_heightmap, is_volatile=True)
                            
                            # Get grasp reward after pushing
                            if goal_conditioned:
                                if is_testing and not random_scene_testing:
                                    obj_contour = robot.get_test_obj_mask(nonlocal_variables['goal_obj_idx'])
                                else:
                                    obj_contour = robot.get_obj_mask(nonlocal_variables['goal_obj_idx'])
                                mask = np.zeros(latest_color_heightmap.shape[:2], np.uint8)
                                mask = utils.get_goal_mask(obj_contour, mask, workspace_limits, heightmap_resolution)
                                latest_obj_grasp_prediction = np.multiply(latest_grasp_predictions, mask)
                                latest_grasp_predictions = latest_obj_grasp_prediction / 255
                             
                            img = latest_color_heightmap
                            obj_contours = robot.get_obj_masks() # get latest contours of objects
                            obj_number = len(robot.obj_mesh_ind) # get mask image and predictions of each object
                            mask_all = np.zeros(img.shape[:2], np.uint8)
                            obj_grasp_predictions, mask_all = utils.get_obj_grasp_predictions(latest_grasp_predictions, obj_contours, mask_all, img, obj_number, workspace_limits, heightmap_resolution)

                            single_predictions = [np.max(obj_grasp_predictions[i]) for i in range(len(obj_grasp_predictions))]
                            print('reward of grasping after pushing: ', single_predictions)

                            # Get improved grasp reward by pushing 
                            improved_grasp_reward = [single_predictions[i] - prev_single_predictions[i] for i in range(len(single_predictions))]
                            print('expected reward of pushing(improved grasp reward)', improved_grasp_reward)
                            if not grasp_goal_conditioned:
                                nonlocal_variables['improved_grasp_reward'] = np.max(improved_grasp_reward)
                            else:
                                nonlocal_variables['improved_grasp_reward'] = improved_grasp_reward[nonlocal_variables['goal_obj_idx']]
                            print('improved grasp reward in thread:', nonlocal_variables['improved_grasp_reward'])
                            writer.add_scalar('improved grasp reward', nonlocal_variables['improved_grasp_reward'], trainer.iteration)

                            nonlocal_variables['episode_improved_grasp_reward'] += nonlocal_variables['improved_grasp_reward']
                            trainer.episode_improved_grasp_reward_log.append([nonlocal_variables['episode_improved_grasp_reward']])
                            logger.write_to_log('episode-improved-grasp-reward', trainer.episode_improved_grasp_reward_log)

                            # Get occupy ratio after pushing
                            if goal_conditioned:
                                occupy_ratio =  utils.get_occupy_ratio(latest_goal_mask_heightmap, latest_depth_heightmap)
                                nonlocal_variables['decreased_occupy_ratio'] = prev_occupy_ratio - occupy_ratio
                                print('decreased_occupy_ratio', nonlocal_variables['decreased_occupy_ratio'])
                                writer.add_scalar('decreased_occupy_ratio', nonlocal_variables['decreased_occupy_ratio'], trainer.iteration)

                        # update push step
                        print('step %d in episode (at most five pushes correspond one episode)' % nonlocal_variables['push_step'])
                        nonlocal_variables['push_step'] += 1

                elif nonlocal_variables['primitive_action'] == 'grasp':
                    nonlocal_variables['grasp_success'], color_image, depth_image, color_height_map, depth_height_map, grasped_object_ind = robot.grasp(primitive_position, best_rotation_angle, workspace_limits)                  
                    print('Grasp successful: %r' % (nonlocal_variables['grasp_success']))
                    writer.add_scalar('grasp success', nonlocal_variables['grasp_success'], nonlocal_variables['episode'])   
                    
                    if nonlocal_variables['grasp_success']:
                        print('Grasp object: %d' % grasped_object_ind)
                        trainer.grasp_obj_log.append([grasped_object_ind])
                        logger.write_to_log('grasp-obj', trainer.grasp_obj_log) 
                    else:
                        trainer.grasp_obj_log.append([-1])
                        logger.write_to_log('grasp-obj', trainer.grasp_obj_log) 

                    # update episode
                    nonlocal_variables['episode'] += 1
                    if stage == 'push_only':
                        writer.add_scalar('episode improved grasp reward', nonlocal_variables['episode_improved_grasp_reward'], nonlocal_variables['episode'])
                        nonlocal_variables['episode_improved_grasp_reward'] = 0                        
                        # update push step
                        print('step %d in episode (five pushes correspond one episode)' % nonlocal_variables['push_step'])
                        writer.add_scalar('episode push step', nonlocal_variables['push_step'], nonlocal_variables['episode'])
                        nonlocal_variables['push_step'] += 1
                        nonlocal_variables['new_episode_flag'] = 1

                    if goal_conditioned:
                        if nonlocal_variables['grasp_success']:
                            if grasped_object_ind == nonlocal_variables['goal_obj_idx']:
                                nonlocal_variables['goal_catched'] = 1
                                print('Goal object catched!')
                                nonlocal_variables['new_episode_flag'] = 1
                                if is_testing:
                                    nonlocal_variables['restart_scene'] = robot.num_obj / 2
                                
                            else:
                                nonlocal_variables['goal_catched'] = 0.5
                                print('A different goal catched! Change the goal object index to', grasped_object_ind)
                                if not is_testing:
                                    nonlocal_variables['goal_obj_idx'] = grasped_object_ind
                                    nonlocal_variables['new_episode_flag'] = 1
                                                   
                        else:
                            nonlocal_variables['goal_catched'] = 0
                        writer.add_scalar('episode goal catched', nonlocal_variables['goal_catched'], nonlocal_variables['episode'])

                nonlocal_variables['executing_action'] = False

            time.sleep(0.01)

    action_thread = threading.Thread(target=process_actions)
    action_thread.daemon = True # thread will stop when main process exits
    action_thread.start()
    exit_called = False
    # -------------------------------------------------------------

    # Start main training/testing loop
    while True:
        print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', trainer.iteration))
        iteration_time_0 = time.time()

        # Make sure simulation is still stable (if not, reset simulation)
        robot.check_sim()

        # Get latest RGB-D image
        color_img, depth_img = robot.get_camera_data()
        depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration

        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
        if goal_conditioned:
            if is_testing and not random_scene_testing:
                obj_contour = robot.get_test_obj_mask(nonlocal_variables['goal_obj_idx'])
            else:
                obj_contour = robot.get_obj_mask(nonlocal_variables['goal_obj_idx'])
            goal_mask_heightmap = np.zeros(color_heightmap.shape[:2], np.uint8)
            goal_mask_heightmap = utils.get_goal_mask(obj_contour, goal_mask_heightmap, workspace_limits, heightmap_resolution)
            kernel = np.ones((3,3))
            nonlocal_variables['border_occupy_ratio'] = utils.get_occupy_ratio(goal_mask_heightmap, depth_heightmap)
            writer.add_scalar('border_occupy_ratio', nonlocal_variables['border_occupy_ratio'], trainer.iteration)

        # Save RGB-D images and RGB-D heightmaps
        logger.save_images(trainer.iteration, color_img, depth_img, '0')
        logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, '0')
                
        writer.add_image('goal_mask_heightmap', cv2.cvtColor(goal_mask_heightmap, cv2.COLOR_BGR2RGB), global_step=trainer.iteration, walltime=None, dataformats='HWC')
        logger.save_visualizations(trainer.iteration, goal_mask_heightmap, 'mask')
        cv2.imwrite('visualization.mask.png', goal_mask_heightmap)
        # Reset simulation or pause real-world training if table is empty
        stuff_count = np.zeros(valid_depth_heightmap.shape)
        stuff_count[valid_depth_heightmap > 0.02] = 1
        empty_threshold = 300
        if is_testing:
            empty_threshold = 10
        if np.sum(stuff_count) < empty_threshold or (no_change_count[0] + no_change_count[1] > 10):
            no_change_count = [0, 0]
            if np.sum(stuff_count) < empty_threshold:
                print('Not enough objects in view (value: %d)! Repositioning objects.' % (np.sum(stuff_count)))
            elif no_change_count[0] + no_change_count[1] > 10:
                print('Too many no change counts (value: %d)! Repositioning objects.' % (no_change_count[0] + no_change_count[1]))

            robot.restart_sim()
            robot.add_objects()
            if is_testing: # If at end of test run, re-load original weights (before test run)
                trainer.model.load_state_dict(torch.load(snapshot_file))

            trainer.clearance_log.append([trainer.iteration])
            logger.write_to_log('clearance', trainer.clearance_log)
            if is_testing and len(trainer.clearance_log) >= max_test_trials:
                exit_called = True # Exit after training thread (backprop and saving labels)
            continue

        # Restart for push_only stage and goal-conditioned case
        if nonlocal_variables['push_step'] == max_push_episode_length + 1 or nonlocal_variables['new_episode_flag'] == 1 or nonlocal_variables['restart_scene'] == robot.num_obj / 2:
            nonlocal_variables['push_step'] = 0  # reset push step
            nonlocal_variables['new_episode_flag'] = 0
            # save episode_improved_grasp_reward
            print('episode %d begins' % nonlocal_variables['episode'])
            if nonlocal_variables['restart_scene'] == robot.num_obj / 2: # If at end of test run, re-load original weights (before test run)
                nonlocal_variables['restart_scene'] = 0
                robot.restart_sim()
                robot.add_objects()
                if is_testing: # If at end of test run, re-load original weights (before test run)
                    trainer.model.load_state_dict(torch.load(snapshot_file))

            trainer.clearance_log.append([trainer.iteration])
            logger.write_to_log('clearance', trainer.clearance_log)
            if is_testing and len(trainer.clearance_log) >= max_test_trials:
                exit_called = True # Exit after training thread (backprop and saving labels)
            continue

        trainer.push_step_log.append([nonlocal_variables['push_step']])
        logger.write_to_log('push-step', trainer.push_step_log)              

        if not exit_called:
            if stage == 'grasp_only' and grasp_explore:
                grasp_explore_actions = np.random.uniform() < grasp_explore_prob
                print('Strategy: explore (exploration probability: %f)' % (grasp_explore_prob))
                if grasp_explore_actions:
                    # Run forward pass with network to get affordances
                    push_predictions, grasp_predictions, state_feat = trainer.forward(color_heightmap, valid_depth_heightmap, is_volatile=True, grasp_explore_actions=True)
                    obj_contour = robot.get_obj_mask(nonlocal_variables['goal_obj_idx'])
                    mask = np.zeros(color_heightmap.shape[:2], np.uint8)
                    mask = utils.get_goal_mask(obj_contour, mask, workspace_limits, heightmap_resolution)
                    obj_grasp_prediction = np.multiply(grasp_predictions, mask)
                    grasp_predictions = obj_grasp_prediction / 255
                else:
                    push_predictions, grasp_predictions, state_feat = trainer.goal_forward(color_heightmap, valid_depth_heightmap, goal_mask_heightmap, is_volatile=True)

            else:
                if not grasp_goal_conditioned:
                    push_predictions, grasp_predictions, state_feat = trainer.forward(color_heightmap, valid_depth_heightmap, is_volatile=True)
                else:
                    push_predictions, grasp_predictions, state_feat = trainer.goal_forward(color_heightmap, valid_depth_heightmap, goal_mask_heightmap, is_volatile=True)
            
            nonlocal_variables['push_predictions'] = push_predictions
            nonlocal_variables['grasp_predictions'] = grasp_predictions

            # Execute best primitive action on robot in another thread
            nonlocal_variables['executing_action'] = True

        # Run training iteration in current thread (aka training thread)
        if 'prev_color_img' in locals():
            # Detect changes
            if not goal_conditioned:
                depth_diff = abs(depth_heightmap - prev_depth_heightmap)
                change_threshold = 300
                change_value = utils.get_change_value(depth_diff)
                change_detected = change_value > change_threshold or prev_grasp_success
                print('Change detected: %r (value: %d)' % (change_detected, change_value))
            else:
                prev_mask_hull = binary_dilation(convex_hull_image(prev_goal_mask_heightmap), iterations=5)
                depth_diff = prev_mask_hull*(prev_depth_heightmap-depth_heightmap)
                change_threshold = 50
                change_value = utils.get_change_value(depth_diff)
                change_detected = change_value > change_threshold
                print('Goal change detected: %r (value: %d)' % (change_detected, change_value)) 

            if change_detected:
                if prev_primitive_action == 'push':
                    no_change_count[0] = 0
                elif prev_primitive_action == 'grasp':
                    no_change_count[1] = 0
            else:
                if prev_primitive_action == 'push':
                    no_change_count[0] += 1
                elif prev_primitive_action == 'grasp':
                    no_change_count[1] += 1

            # Compute training labels
            if not grasp_goal_conditioned:
                label_value, prev_reward_value = trainer.get_label_value(prev_primitive_action, prev_grasp_success, prev_grasp_reward, prev_improved_grasp_reward, change_detected, color_heightmap, valid_depth_heightmap)
            else:
                label_value, prev_reward_value = trainer.get_label_value(prev_primitive_action, prev_grasp_success, prev_grasp_reward, prev_improved_grasp_reward, change_detected, color_heightmap, valid_depth_heightmap, 
                goal_mask_heightmap, nonlocal_variables['goal_catched'], nonlocal_variables['decreased_occupy_ratio'])

            trainer.label_value_log.append([label_value])
            logger.write_to_log('label-value', trainer.label_value_log)
            trainer.reward_value_log.append([prev_reward_value])
            logger.write_to_log('reward-value', trainer.reward_value_log)

            # Backpropagate
            if not grasp_goal_conditioned:
                loss = trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap, prev_primitive_action, prev_best_pix_ind, label_value)
            else:
                loss = trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap, prev_primitive_action, prev_best_pix_ind, label_value, prev_goal_mask_heightmap)
            writer.add_scalar('loss', loss, trainer.iteration)

            episode_loss += loss
            if nonlocal_variables['push_step'] == max_push_episode_length or nonlocal_variables['new_episode_flag'] == 1:
                writer.add_scalar('episode loss', episode_loss, nonlocal_variables['episode'])
                episode_loss = 0
            
            # Adjust exploration probability
            if not is_testing:
                explore_prob = max(0.5 * np.power(0.9998, trainer.iteration),0.1) if explore_rate_decay else 0.5
                grasp_explore_prob = max(0.8 * np.power(0.998, trainer.iteration),0.1) if explore_rate_decay else 0.8

            # Do sampling for experience replay
            if experience_replay and not is_testing:
                sample_primitive_action = prev_primitive_action
                if grasp_goal_conditioned:
                    sample_goal_obj_idx = nonlocal_variables['goal_obj_idx']
                    print('sample_goal_obj_idx', sample_goal_obj_idx)
                if sample_primitive_action == 'push':
                    sample_primitive_action_id = 0
                    sample_reward_value = 0 if prev_reward_value == 0.5 else 0.5
                elif sample_primitive_action == 'grasp':
                    sample_primitive_action_id = 1
                    sample_reward_value = 0 if prev_reward_value == 1 else 1

                # Get samples of the same primitive but with different results
                if not grasp_goal_conditioned or sample_primitive_action == 'push':
                    sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.reward_value_log)[0:trainer.iteration,0] == sample_reward_value, np.asarray(trainer.executed_action_log)[0:trainer.iteration,0] == sample_primitive_action_id))
                else:
                    sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.reward_value_log)[0:trainer.iteration,0] == sample_reward_value, 
                    np.asarray(trainer.grasp_obj_log)[0:trainer.iteration,0] == sample_goal_obj_idx))
                   
                if sample_ind.size > 0:
                    print('reward_value_log:', np.asarray(trainer.reward_value_log)[sample_ind[:,0], 0])
                    # Find sample with highest surprise value
                    sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:,0]] - np.asarray(trainer.label_value_log)[sample_ind[:,0]])
                    sorted_surprise_ind = np.argsort(sample_surprise_values[:,0])
                    sorted_sample_ind = sample_ind[sorted_surprise_ind,0]
                    pow_law_exp = 2
                    rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1)*(sample_ind.size-1)))
                    sample_iteration = sorted_sample_ind[rand_sample_ind]
                    print('Experience replay: iteration %d (surprise value: %f)' % (sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))

                    # Load sample RGB-D heightmap
                    sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration)))
                    sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
                    sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration)), -1)
                    sample_depth_heightmap = sample_depth_heightmap.astype(np.float32)/100000
                    
                    if grasp_goal_conditioned or goal_conditioned:
                        if is_testing and not random_scene_testing:
                            obj_contour = robot.get_test_obj_mask(nonlocal_variables['goal_obj_idx'])
                        else:
                            obj_contour = robot.get_obj_mask(nonlocal_variables['goal_obj_idx'])
                        sample_goal_mask_heightmap = np.zeros(color_heightmap.shape[:2], np.uint8)
                        sample_goal_mask_heightmap = utils.get_goal_mask(obj_contour, sample_goal_mask_heightmap, workspace_limits, heightmap_resolution)
                        writer.add_image('goal_mask_heightmap', cv2.cvtColor(sample_goal_mask_heightmap, cv2.COLOR_BGR2RGB), global_step=trainer.iteration, walltime=None, dataformats='HWC')

                    # Compute forward pass with sample
                    with torch.no_grad():
                        if not grasp_goal_conditioned:
                            sample_push_predictions, sample_grasp_predictions, sample_state_feat = trainer.forward(sample_color_heightmap, sample_depth_heightmap, is_volatile=True)
                        else:
                            sample_push_predictions, sample_grasp_predictions, sample_state_feat = trainer.goal_forward(sample_color_heightmap, sample_depth_heightmap, sample_goal_mask_heightmap, is_volatile=True)

                    sample_grasp_success = sample_reward_value == 1
                    # Get labels for sample and backpropagate
                    sample_best_pix_ind = (np.asarray(trainer.executed_action_log)[sample_iteration,1:4]).astype(int)
                    if not grasp_goal_conditioned:  
                        trainer.backprop(sample_color_heightmap, sample_depth_heightmap, sample_primitive_action, sample_best_pix_ind, trainer.label_value_log[sample_iteration])
                    else:
                        trainer.backprop(sample_color_heightmap, sample_depth_heightmap, sample_primitive_action, sample_best_pix_ind, trainer.label_value_log[sample_iteration], sample_goal_mask_heightmap)

                    # Recompute prediction value and label for replay buffer
                    if sample_primitive_action == 'push':
                        trainer.predicted_value_log[sample_iteration] = [np.max(sample_push_predictions)]
                    elif sample_primitive_action == 'grasp':
                        trainer.predicted_value_log[sample_iteration] = [np.max(sample_grasp_predictions)]

                else:
                    print('Not enough prior training samples. Skipping experience replay.')

            # Save model snapshot
            if not is_testing:
                logger.save_backup_model(trainer.model, stage)
                if trainer.iteration % 50 == 0:
                    logger.save_model(trainer.iteration, trainer.model, stage)
                    if trainer.use_cuda:
                        trainer.model = trainer.model.cuda()

        # Sync both action thread and training thread
        while nonlocal_variables['executing_action']:
            time.sleep(0.01)

        if exit_called:
            break

        # Save information for next training step
        prev_color_img = color_img.copy()
        prev_depth_img = depth_img.copy()
        prev_color_heightmap = color_heightmap.copy()
        prev_depth_heightmap = depth_heightmap.copy()
        prev_valid_depth_heightmap = valid_depth_heightmap.copy()
        prev_push_success = nonlocal_variables['push_success']
        prev_grasp_success = nonlocal_variables['grasp_success']
        prev_primitive_action = nonlocal_variables['primitive_action']
        prev_push_predictions = push_predictions.copy()
        prev_grasp_predictions = grasp_predictions.copy()
        prev_best_pix_ind = nonlocal_variables['best_pix_ind']
        prev_grasp_reward = nonlocal_variables['grasp_reward']
        if grasp_goal_conditioned or goal_conditioned:
            prev_goal_mask_heightmap = goal_mask_heightmap.copy()
        if stage == 'push_only':
            prev_improved_grasp_reward = nonlocal_variables['improved_grasp_reward']
            prev_grasp_reward = nonlocal_variables['grasp_reward']
        else:
            prev_improved_grasp_reward = 0.0

        trainer.iteration += 1
        iteration_time_1 = time.time()
        print('Time elapsed: %f' % (iteration_time_1-iteration_time_0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/blocks',                  help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=10,                                help='number of objects to add to simulation')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store', default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234,                      help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,                                    help='force code to run in CPU mode')

    # ------------- Algorithm options -------------
    parser.add_argument('--stage', dest='stage', action='store', default='grasp_only',                               help='stage of training: 1.grasp_only, 2.push_only, 3.push_grasp')
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store', default=0.5)
    parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=False,              help='use prioritized experience replay?')
    parser.add_argument('--heuristic_bootstrap', dest='heuristic_bootstrap', action='store_true', default=False,          help='use handcrafted grasping algorithm when grasping fails too many times in a row during training?')
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=False)
    parser.add_argument('--grasp_reward_threshold', dest='grasp_reward_threshold', type=float, action='store', default=1.8)
    parser.add_argument('--max_push_episode_length', dest='max_push_episode_length', type=int, action='store', default=5)
    parser.add_argument('--grasp_explore', dest='grasp_explore', action='store_true', default=False)

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=30,                help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='test-10-obj-01.txt')
    parser.add_argument('--random_scene_testing', dest='random_scene_testing', action='store_true', default=False)
    
    # -------------- Goal-conditioned options --------------
    parser.add_argument('--goal_obj_idx', dest='goal_obj_idx', type=int, action='store', default=0)
    parser.add_argument('--goal_conditioned', dest='goal_conditioned', action='store_true', default=False)
    parser.add_argument('--grasp_goal_conditioned', dest='grasp_goal_conditioned', action='store_true', default=False)

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,                      help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
    parser.add_argument('--load_explore_snapshot', dest='load_explore_snapshot', action='store_true', default=False,                      help='load pre-trained snapshot of model?')
    parser.add_argument('--explore_snapshot_file', dest='explore_snapshot_file', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,                help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,          help='save visualizations of FCN predictions?')
    parser.add_argument('--tensor_logging_directory', dest='tensor_logging_directory', action='store', default='./tensorlog')
    parser.add_argument('--alternating_training', dest='alternating_training', action='store_true', default=False)
    parser.add_argument('--cooperative_training', dest='cooperative_training', action='store_true', default=False)

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)