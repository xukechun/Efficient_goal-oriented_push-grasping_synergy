import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import CrossEntropyLoss2d
from models import push_grasp_net, goal_conditioned_net
from scipy import ndimage
import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self, stage, future_reward_discount,
                 is_testing, load_snapshot, snapshot_file,
                 load_explore_snapshot, explore_snapshot_file, 
                 alternating_training, cooperative_training,
                 force_cpu, grasp_goal_conditioned):

        self.stage = stage
        self.grasp_goal_conditioned = grasp_goal_conditioned
        self.is_testing = is_testing
        self.alternating_training = alternating_training

        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print("CUDA detected. Running with GPU acceleration.")
            self.use_cuda = True
        elif force_cpu:
            print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            self.use_cuda = False
        else:
            print("CUDA is *NOT* detected. Running with only CPU.")
            self.use_cuda = False

        # Fully convolutional Q network for deep reinforcement learning
        if not self.grasp_goal_conditioned:
            self.model = push_grasp_net(self.use_cuda)
        else:
            self.model = goal_conditioned_net(self.use_cuda)
            if load_explore_snapshot:
                self.explore_model = push_grasp_net(self.use_cuda)

        self.future_reward_discount = future_reward_discount

        # Initialize Huber loss
        self.criterion = torch.nn.SmoothL1Loss(reduce=False) # Huber loss
        if self.use_cuda:
            self.criterion = self.criterion.cuda()

        # Load pre-trained model
        if load_explore_snapshot:
            self.explore_model.load_state_dict(torch.load(explore_snapshot_file))
            print('Explore model snapshot loaded from: %s' % (explore_snapshot_file))

        if load_snapshot:
            if self.stage == 'grasp_only':
                self.model.load_state_dict(torch.load(snapshot_file))
                print('Pre-trained model snapshot loaded from: %s' % (snapshot_file))
            elif self.stage == 'push_only':
                # only load grasp then initialize grasp
                self.model.load_state_dict(torch.load(snapshot_file))
                print('Pre-trained model snapshot loaded from: %s' % (snapshot_file))
            elif self.stage == 'push_grasp':
                self.model.load_state_dict(torch.load(snapshot_file))
                print('Pre-trained model snapshot loaded from: %s' % (snapshot_file))

        # For push_only stage, grasp net is fixed at the beginning, for grasp_only stage, push net will not be trained
        if self.stage == 'push_only':
            if not cooperative_training:
                # co-training
                for k,v in self.model.named_parameters():
                    if 'grasp-'in k:
                        v.requires_grad=False # fix parameters
            if self.alternating_training:
                ########################################
                # change me to update different policies
                ########################################
                for k,v in self.model.named_parameters():
                    if 'push-'in k:
                        v.requires_grad=False # fix parameters
                for k,v in self.model.named_parameters():
                    if 'grasp-'in k:
                        v.requires_grad=True # fix parameters
                                                  
            # Print
            for k,v in self.model.named_parameters():
                if 'push-'in k:
                    print(v.requires_grad) # supposed to be false 
                if 'grasp-'in k:
                    print(v.requires_grad) # supposed to be false 

        elif self.stage == 'grasp_only':
            for k,v in self.model.named_parameters():
                if 'push-'in k:
                    v.requires_grad=False # fix parameters
            # Print
            for k,v in self.model.named_parameters():
                if 'push-'in k:
                    print(v.requires_grad) # supposed to be false 
        
        # for real world experiments
        elif self.stage == 'push_grasp':
            for k,v in self.model.named_parameters():
                if 'push-'in k:
                    v.requires_grad=False # fix parameters 
            # Print
            for k,v in self.model.named_parameters():
                if 'push-'in k:
                    print(v.requires_grad) # supposed to be false 

        # Convert model from CPU to GPU
        if self.use_cuda:
            self.model = self.model.cuda()
            # print('change to cuda!')
            if load_explore_snapshot:
                self.explore_model = self.explore_model.cuda()

        # Set model to training mode
        self.model.train()

        # Initialize optimizer Adam
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=2e-5, betas=(0.9,0.99))
        self.iteration = 0

        # Initialize lists to save execution info and RL variables
        self.executed_action_log = []
        self.label_value_log = []
        self.reward_value_log = []
        self.predicted_value_log = []
        self.use_heuristic_log = []
        self.is_exploit_log = []
        self.clearance_log = []
        self.push_step_log = []
        self.grasp_obj_log = [] # grasp object index (if push or grasp fail then index is -1)
        self.episode_log = []
        self.episode_improved_grasp_reward_log = []


    # Pre-load execution info and RL variables
    def preload(self, transitions_directory):
        self.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), delimiter=' ')
        self.iteration = self.executed_action_log.shape[0] - 2
        self.executed_action_log = self.executed_action_log[0:self.iteration,:]
        self.executed_action_log = self.executed_action_log.tolist()
        self.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'label-value.log.txt'), delimiter=' ')
        self.label_value_log = self.label_value_log[0:self.iteration]
        self.label_value_log.shape = (self.iteration,1)
        self.label_value_log = self.label_value_log.tolist()
        self.predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'), delimiter=' ')
        self.predicted_value_log = self.predicted_value_log[0:self.iteration]
        self.predicted_value_log.shape = (self.iteration,1)
        self.predicted_value_log = self.predicted_value_log.tolist()
        self.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
        self.reward_value_log = self.reward_value_log[0:self.iteration]
        self.reward_value_log.shape = (self.iteration,1)
        self.reward_value_log = self.reward_value_log.tolist()
        self.use_heuristic_log = np.loadtxt(os.path.join(transitions_directory, 'use-heuristic.log.txt'), delimiter=' ')
        self.use_heuristic_log = self.use_heuristic_log[0:self.iteration]
        self.use_heuristic_log.shape = (self.iteration,1)
        self.use_heuristic_log = self.use_heuristic_log.tolist()
        self.is_exploit_log = np.loadtxt(os.path.join(transitions_directory, 'is-exploit.log.txt'), delimiter=' ')
        self.is_exploit_log = self.is_exploit_log[0:self.iteration]
        self.is_exploit_log.shape = (self.iteration,1)
        self.is_exploit_log = self.is_exploit_log.tolist()
        self.clearance_log = np.loadtxt(os.path.join(transitions_directory, 'clearance.log.txt'), delimiter=' ')
        self.clearance_log.shape = (self.clearance_log.shape[0],1)
        self.clearance_log = self.clearance_log.tolist()
        self.grasp_obj_log = np.loadtxt(os.path.join(transitions_directory, 'grasp-obj.log.txt'), delimiter=' ')
        self.grasp_obj_log = self.grasp_obj_log[0:self.iteration]
        self.grasp_obj_log.shape = (self.iteration,1)
        self.grasp_obj_log = self.grasp_obj_log.tolist()
        
        if not self. is_testing:
            self.episode_log = np.loadtxt(os.path.join(transitions_directory, 'episode.log.txt'), delimiter=' ')
            self.episode_log = self.episode_log[0:self.iteration]
            self.episode_log.shape = (self.iteration,1)
            self.episode_log = self.episode_log.tolist()
        
        if self.stage == 'push_only':
            self.push_step_log = np.loadtxt(os.path.join(transitions_directory, 'push-step.log.txt'), delimiter=' ')
            self.push_step_log = self.push_step_log[0:len(self.push_step_log)]
            self.push_step_log.shape = (len(self.push_step_log),1)
            self.push_step_log = self.push_step_log.tolist()
            self.episode_improved_grasp_reward_log = np.loadtxt(os.path.join(transitions_directory, 'episode-improved-grasp-reward.log.txt'), delimiter=' ')
            self.episode_improved_grasp_reward_log = self.episode_improved_grasp_reward_log[0:len(self.episode_improved_grasp_reward_log)]
            self.episode_improved_grasp_reward_log.shape = (len(self.episode_improved_grasp_reward_log),1)
            self.episode_improved_grasp_reward_log = self.episode_improved_grasp_reward_log.tolist()


    # Compute forward pass through model to compute affordances/Q
    def forward(self, color_heightmap, depth_heightmap, is_volatile=False, grasp_explore_actions=False, specific_rotation=-1):

        # Apply 2x scale to input heightmaps
        # The third dimension is rgb
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2,2,1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2,2], order=0)
        assert(color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2])

        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length/32)*32
        padding_width = int((diag_length - color_heightmap_2x.shape[0])/2)
        color_heightmap_2x_r =  np.pad(color_heightmap_2x[:,:,0], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
        color_heightmap_2x_g =  np.pad(color_heightmap_2x[:,:,1], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
        color_heightmap_2x_b =  np.pad(color_heightmap_2x[:,:,2], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
        depth_heightmap_2x =  np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)

        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float)/255
        for c in range(3):
            input_color_image[:,:,c] = (input_color_image[:,:,c] - image_mean[c])/image_std[c]

        # Pre-process depth image (normalize)
        # change me
        image_mean = [0.01, 0.01, 0.01]
        image_std = [0.03, 0.03, 0.03]
        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = np.concatenate((depth_heightmap_2x, depth_heightmap_2x, depth_heightmap_2x), axis=2)
        for c in range(3):
            input_depth_image[:,:,c] = (input_depth_image[:,:,c] - image_mean[c])/image_std[c]

        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3,2,0,1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3,2,0,1)

        # Pass input data through model
        if not grasp_explore_actions:
            output_prob, state_feat = self.model.forward(input_color_data, input_depth_data, is_volatile, specific_rotation)
        else:
            output_prob, state_feat = self.explore_model.forward(input_color_data, input_depth_data, is_volatile, specific_rotation)
            print('Explore from graps net!')

        if self.stage == 'grasp_only':

            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    push_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    push_predictions = np.concatenate((push_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        elif self.stage == 'push_only':

            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    push_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    push_predictions = np.concatenate((push_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        elif self.stage == 'push_grasp':

            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    push_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    push_predictions = np.concatenate((push_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        return push_predictions, grasp_predictions, state_feat


    def goal_forward(self, color_heightmap, depth_heightmap, goal_mask_heightmap, is_volatile=False, specific_rotation=-1):

        # Apply 2x scale to input heightmaps
        # The third dimension is rgb
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2,2,1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2,2], order=0)
        goal_mask_heightmap_2x = ndimage.zoom(goal_mask_heightmap, zoom=[2,2], order=0)
        assert(color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2])

        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length/32)*32
        padding_width = int((diag_length - color_heightmap_2x.shape[0])/2)
        color_heightmap_2x_r =  np.pad(color_heightmap_2x[:,:,0], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
        color_heightmap_2x_g =  np.pad(color_heightmap_2x[:,:,1], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
        color_heightmap_2x_b =  np.pad(color_heightmap_2x[:,:,2], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)

        depth_heightmap_2x =  np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)
        goal_mask_heightmap_2x =  np.pad(goal_mask_heightmap_2x, padding_width, 'constant', constant_values=0)

        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float)/255
        for c in range(3):
            input_color_image[:,:,c] = (input_color_image[:,:,c] - image_mean[c])/image_std[c]

        # Pre-process depth image (normalize)
        image_mean = [0.01, 0.01, 0.01]
        image_std = [0.03, 0.03, 0.03]
        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = np.concatenate((depth_heightmap_2x, depth_heightmap_2x, depth_heightmap_2x), axis=2)
        goal_mask_heightmap_2x.shape = (goal_mask_heightmap_2x.shape[0], goal_mask_heightmap_2x.shape[1], 1)
        input_goal_mask = np.concatenate((goal_mask_heightmap_2x, goal_mask_heightmap_2x, goal_mask_heightmap_2x), axis=2)
        input_goal_mask = input_goal_mask.astype(float)/255
        for c in range(3):
            input_depth_image[:,:,c] = (input_depth_image[:,:,c] - image_mean[c])/image_std[c]

        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3,2,0,1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3,2,0,1)

        input_goal_mask.shape = (input_goal_mask.shape[0], input_goal_mask.shape[1], input_goal_mask.shape[2], 1)
        input_goal_mask_data = torch.from_numpy(input_goal_mask.astype(np.float32)).permute(3,2,0,1)


        # Pass input data through model
        output_prob, state_feat= self.model.forward(input_color_data, input_depth_data, input_goal_mask_data, is_volatile, specific_rotation)

        if self.stage == 'grasp_only':

            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    push_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    push_predictions = np.concatenate((push_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        elif self.stage == 'push_only':

            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    push_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    push_predictions = np.concatenate((push_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        elif self.stage == 'push_grasp':

            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    push_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    push_predictions = np.concatenate((push_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        return push_predictions, grasp_predictions, state_feat


    # grasp reward is rate of sucessfully-grasping
    def get_label_value(self, primitive_action,  grasp_success, grasp_reward, improved_grasp_reward, change_detected, next_color_heightmap, next_depth_heightmap, next_goal_mask_heightmap=None, goal_catched=0, decreased_occupy_ratio=0):

        if self.stage == 'grasp_only':
            # Compute current reward
            current_reward = 0
            if grasp_success:
                current_reward = 1.0

            # Compute future reward
            if not grasp_success:
                future_reward = 0
            else:
                if not self.grasp_goal_conditioned:
                    next_push_predictions, next_grasp_predictions, next_state_feat = self.forward(next_color_heightmap, next_depth_heightmap, is_volatile=True)
                else:
                    next_push_predictions, next_grasp_predictions, next_state_feat = self.goal_forward(next_color_heightmap, next_depth_heightmap, next_goal_mask_heightmap, is_volatile=True)
                future_reward = np.max(next_grasp_predictions)

            print('Current reward: %f' % (current_reward))
            print('Future reward: %f' % (future_reward))
            if goal_catched == 1:
                expected_reward = current_reward + self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward))
            else:
                expected_reward = self.future_reward_discount * future_reward
                print('Expected reward: %f x %f = %f' % (self.future_reward_discount, future_reward, expected_reward))
            return expected_reward, current_reward
        
        elif self.stage == 'push_only':

            # Compute current reward
            if primitive_action == 'push':
                if improved_grasp_reward > 0.1 and change_detected and decreased_occupy_ratio > 0.1:
                    # change of reward after pushing
                    current_reward = 0.5
                elif not change_detected or not decreased_occupy_ratio > 0.1:
                    current_reward = -0.5
                else:
                    current_reward = 0
                print('improved grasp reward after pushing in trainer:', improved_grasp_reward)

            elif primitive_action == 'grasp':
                if grasp_success:
                    current_reward = 1.5
                else:
                    current_reward = 0
                    if self.alternating_training:
                        current_reward = -1.0

            # Compute future reward
            if improved_grasp_reward <= 0.1 and grasp_reward < 0.5 and not grasp_success:
                future_reward = 0
            else:
                if not self.grasp_goal_conditioned:
                    next_push_predictions, next_grasp_predictions, next_state_feat = self.forward(next_color_heightmap, next_depth_heightmap, is_volatile=True)
                else:
                    next_push_predictions, next_grasp_predictions, next_state_feat = self.goal_forward(next_color_heightmap, next_depth_heightmap, next_goal_mask_heightmap, is_volatile=True)

                future_reward = np.max(next_push_predictions)

            print('Current reward: %f' % (current_reward))
            print('Future reward: %f' % (future_reward))
            
            if primitive_action == 'push':
                expected_reward = current_reward + self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward))
            else:
                expected_reward = current_reward + self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward))

            return expected_reward, current_reward
            
        elif self.stage == 'push_grasp':
            # Compute current reward
            current_reward = 0
            if primitive_action == 'push':
                if change_detected:
                    current_reward = 0.5
            elif primitive_action == 'grasp':
                if grasp_success:
                    current_reward = 1.0

            # Compute future reward
            if not change_detected and not grasp_success:
                future_reward = 0
            else:
                if not self.grasp_goal_conditioned:
                    next_push_predictions, next_grasp_predictions, next_state_feat = self.forward(next_color_heightmap, next_depth_heightmap, is_volatile=True)
                else:
                    next_push_predictions, next_grasp_predictions, next_state_feat = self.goal_forward(next_color_heightmap, next_depth_heightmap, next_goal_mask_heightmap, is_volatile=True)

                future_reward = max(np.max(next_push_predictions), np.max(next_grasp_predictions))

            print('Current reward: %f' % (current_reward))
            print('Future reward: %f' % (future_reward))
            if primitive_action == 'push':
                expected_reward = self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' % (0.0, self.future_reward_discount, future_reward, expected_reward))
            else:
                expected_reward = current_reward + self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward))
            return expected_reward, current_reward


    # Compute labels and backpropagate
    def backprop(self, color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value, goal_mask_heightmap=None):

        if self.stage == 'grasp_only':
            # Compute labels
            label = np.zeros((1,320,320))
            action_area = np.zeros((224,224))
            action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
            # blur_kernel = np.ones((5,5),np.float32)/25
            # action_area = cv2.filter2D(action_area, -1, blur_kernel)
            tmp_label = np.zeros((224,224))
            tmp_label[action_area > 0] = label_value
            label[0,48:(320-48),48:(320-48)] = tmp_label

            # Compute label mask
            label_weights = np.zeros(label.shape)
            tmp_label_weights = np.zeros((224,224))
            tmp_label_weights[action_area > 0] = 1
            label_weights[0,48:(320-48),48:(320-48)] = tmp_label_weights

            # Compute loss and backward pass
            self.optimizer.zero_grad()
            loss_value = 0

            # Do forward pass with specified rotation (to save gradients)
            if not self.grasp_goal_conditioned:
                push_predictions, grasp_predictions, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])
            else:
                push_predictions, grasp_predictions, state_feat = self.goal_forward(color_heightmap, depth_heightmap, goal_mask_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])

            if self.use_cuda:
                loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
            else:
                loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(), requires_grad=False)
            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

            opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations/2) % self.model.num_rotations

            if not self.grasp_goal_conditioned:
                push_predictions, grasp_predictions, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx)
            else:
                push_predictions, grasp_predictions, state_feat = self.goal_forward(color_heightmap, depth_heightmap, goal_mask_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx)

            if self.use_cuda:
                loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
            else:
                loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)

            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

            loss_value = loss_value/2

            print('Training loss: %f' % (loss_value))
            self.optimizer.step()

        elif self.stage == 'push_only':

            # Compute labels
            label = np.zeros((1, 320, 320))
            action_area = np.zeros((224, 224))
            action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
            tmp_label = np.zeros((224, 224))
            tmp_label[action_area > 0] = label_value
            label[0, 48:(320-48), 48:(320-48)] = tmp_label

            # Compute label mask
            label_weights = np.zeros(label.shape)
            tmp_label_weights = np.zeros((224, 224))
            tmp_label_weights[action_area > 0] = 1
            label_weights[0,48:(320-48),48:(320-48)] = tmp_label_weights

            # Compute loss and backward pass
            self.optimizer.zero_grad()
            loss_value = 0
           # Do forward pass with specified rotation (to save gradients)
            if not self.grasp_goal_conditioned:
                push_predictions, grasp_predictions, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])
            else:
                push_predictions, grasp_predictions, state_feat = self.goal_forward(color_heightmap, depth_heightmap, goal_mask_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])

            if self.use_cuda:
                loss = self.criterion(self.model.output_prob[0][0].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
            else:
                loss = self.criterion(self.model.output_prob[0][0].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)
            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()
            print('Training loss: %f' % (loss_value))
            self.optimizer.step()

        elif self.stage == 'push_grasp':

            # Compute labels
            label = np.zeros((1,320,320))
            action_area = np.zeros((224,224))
            action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
            tmp_label = np.zeros((224,224))
            tmp_label[action_area > 0] = label_value
            label[0,48:(320-48),48:(320-48)] = tmp_label

            # Compute label mask
            label_weights = np.zeros(label.shape)
            tmp_label_weights = np.zeros((224,224))
            tmp_label_weights[action_area > 0] = 1
            label_weights[0,48:(320-48),48:(320-48)] = tmp_label_weights

            # Compute loss and backward pass
            self.optimizer.zero_grad()
            loss_value = 0
            if primitive_action == 'push':

                # Do forward pass with specified rotation (to save gradients)
                if not self.grasp_goal_conditioned:
                    push_predictions, grasp_predictions, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])
                else:
                    push_predictions, grasp_predictions, state_feat = self.goal_forward(color_heightmap, depth_heightmap, goal_mask_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])

                if self.use_cuda:
                    loss = self.criterion(self.model.output_prob[0][0].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
                else:
                    loss = self.criterion(self.model.output_prob[0][0].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)
                loss = loss.sum()
                loss.backward()
                loss_value = loss.cpu().data.numpy()

            elif primitive_action == 'grasp':

                # Do forward pass with specified rotation (to save gradients)
                if not self.grasp_goal_conditioned:
                    push_predictions, grasp_predictions, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])
                else:
                    push_predictions, grasp_predictions, state_feat = self.goal_forward(color_heightmap, depth_heightmap, goal_mask_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])

                if self.use_cuda:
                    loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
                else:
                    loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)
                loss = loss.sum()
                loss.backward()
                loss_value = loss.cpu().data.numpy()

                opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations/2) % self.model.num_rotations

                if not self.grasp_goal_conditioned:
                    push_predictions, grasp_predictions, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx)
                else:
                    push_predictions, grasp_predictions, state_feat = self.goal_forward(color_heightmap, depth_heightmap, goal_mask_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx)


                if self.use_cuda:
                    loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
                else:
                    loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)

                loss = loss.sum()
                loss.backward()
                loss_value = loss.cpu().data.numpy()

                loss_value = loss_value/2

            print('Training loss: %f' % (loss_value))
            self.optimizer.step()
        return loss_value


    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind):

        canvas = None
        num_rotations = predictions.shape[0]
        for canvas_row in range(int(num_rotations/4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row*4+canvas_col
                prediction_vis = predictions[rotate_idx,:,:].copy()
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                if rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0,0,255), 2)
                prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                background_image = ndimage.rotate(color_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

        return canvas


    def get_push_direction_vis(self, push_predictions, color_heightmap):
        push_direction_canvas = color_heightmap
        x = 0
        while x < push_predictions.shape[2]:
            y = 0
            while y < push_predictions.shape[1]:
                angle_idx = np.argmax(push_predictions[:, y, x])
                angel = np.deg2rad(angle_idx*(360.0/self.model.num_rotations))
                start_point = (x, y)
                end_point = (int(x + 10*np.cos(angel)), int(y + 10*np.sin(angel)))
                quality = np.max(push_predictions[:, y, x])
                
                color = (0, 0, (quality*255).astype(np.uint8))
                cv2.arrowedLine(push_direction_canvas,start_point, end_point, (0,0,255), 1, 0, 0, 0.3)
                y+=10
            x+=10

        plt.figure()
        plt.imshow(push_direction_canvas)
        plt.show()
        return push_direction_canvas


    def get_best_push_direction_vis(self, best_pix_ind, color_heightmap):
        push_direction_canvas = color_heightmap
        angle_idx = best_pix_ind[0]
        angel = np.deg2rad(angle_idx*(360.0/self.model.num_rotations))
        start_point = (best_pix_ind[2], best_pix_ind[1])
        end_point = (int(best_pix_ind[2] + 20*np.cos(angel)), int(best_pix_ind[1] + 20*np.sin(angel)))
        cv2.arrowedLine(push_direction_canvas,start_point, end_point, (0,0,255), 1, 0, 0, 0.3)
        cv2.circle(push_direction_canvas, (int(best_pix_ind[2]), int(best_pix_ind[1])), 4, (0,0,255), 1)

        return push_direction_canvas


    def push_heuristic(self, depth_heightmap):
        num_rotations = 16
        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(depth_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            valid_areas[ndimage.interpolation.shift(rotated_heightmap, [0,-25], order=0) - rotated_heightmap > 0.02] = 1
            # valid_areas = np.multiply(valid_areas, rotated_heightmap)
            blur_kernel = np.ones((25,25),np.float32)/9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_push_predictions = ndimage.rotate(valid_areas, -rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            tmp_push_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

            if rotate_idx == 0:
                push_predictions = tmp_push_predictions
            else:
                push_predictions = np.concatenate((push_predictions, tmp_push_predictions), axis=0)

        best_pix_ind = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
        return best_pix_ind


    def grasp_heuristic(self, depth_heightmap):

        num_rotations = 16

        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(depth_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            valid_areas[np.logical_and(rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,-25], order=0) > 0.02, rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,25], order=0) > 0.02)] = 1
            # valid_areas = np.multiply(valid_areas, rotated_heightmap)
            blur_kernel = np.ones((25,25),np.float32)/9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_grasp_predictions = ndimage.rotate(valid_areas, -rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            tmp_grasp_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

            if rotate_idx == 0:
                grasp_predictions = tmp_grasp_predictions
            else:
                grasp_predictions = np.concatenate((grasp_predictions, tmp_grasp_predictions), axis=0)

        best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
        return best_pix_ind

