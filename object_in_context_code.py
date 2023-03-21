"""
Created on Tue Mar  7 10:55:06 2023
@author: Cees van Middelkoop
ANN Objects in Context
"""

# dependencies
import torch
from torch import nn
from torchvision import models, transforms, datasets
from torchvision.models import EfficientNet_V2_L_Weights, VGG16_Weights, ViT_H_14_Weights
from collections import OrderedDict
import timm

import time
import pandas as pd
import re
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import seaborn as sn
from adjustText import adjust_text

import math
import scipy.stats
import random
import numpy as np
from numpy.polynomial.polynomial import polyfit
import os

# from torchvision.models import detection
# from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
#%% 


class ObjectAnalysis:
    
    def __init__(self):
        """
        hardware acceleration check
        """
        
        self.device = torch.device('cpu') # cpu
        
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device('mps') # apple m1
        elif torch.cuda.is_available():
            self.device = torch.device('cuda') # nvidia
            

    def model_init(self, model_name, input_dim, model_type, weights=None, model_url=None, cornet_model=None):
        """
        loading a pretrained model
        """

        self.model_name = model_name
        self.input_dim = input_dim
        self.model_type = model_type
        self.weights = weights
        self.model_url = model_url
        self.cornet_model = cornet_model
        print(self.model_name)
        
        if self.model_type == 'timm':
            
            self.model = timm.create_model(self.model_name, pretrained=True)
            if self.weights != None:
                self.model.load_state_dict(torch.load(self.weights)["state_dict"])
                self.model_name = self.weights
            
            self.model.to(self.device)
            self.model.eval()
    
        if self.model_type == 'pytorch':
            
            self.device = torch.device('cpu') # override
            func_pytorch_model = getattr(models, self.model_name)
            self.model = func_pytorch_model(weights=self.weights)
            self.model.to(self.device)
            self.model.eval()
            
        # if self.model_type == 'pytorch_detection':
            
        #     self.device = torch.device('cpu') # override
            
        #     func_pytorch_model = getattr(detection, self.model_name)
        #     self.model = func_pytorch_model(weights=self.weights, device=self.device)
        #     self.model.to(self.device)
        #     self.model.eval()
    
        if self.model_type == 'cornet_z':
            
            self.model = CORnet_Z()
            self.model = torch.nn.DataParallel(self.model)
            self.model.to(self.device)
            self.modelzoo_weights = torch.utils.model_zoo.load_url(self.model_url, map_location=self.device)
            self.model.load_state_dict(self.modelzoo_weights["state_dict"])
            self.model.eval()

        if self.model_type == 'cornet_rt':
            
            self.device = torch.device('cpu') # override
            
            self.model = CORnet_RT()
            self.model = torch.nn.DataParallel(self.model)
            self.model.to(self.device)
            self.modelzoo_weights = torch.utils.model_zoo.load_url(self.model_url, map_location=self.device)
            self.model.load_state_dict(self.modelzoo_weights["state_dict"])
            self.model.eval()
            
        if self.model_type == 'cornet_s':
            
            self.device = torch.device('cpu') # override
            
            self.model = CORnet_S()
            self.model = torch.nn.DataParallel(self.model)
            self.model.to(self.device)
            self.modelzoo_weights = torch.utils.model_zoo.load_url(self.model_url, map_location=self.device)
            self.model.load_state_dict(self.modelzoo_weights["state_dict"])
            self.model.eval()


    def set_dirs(self, stimuli_dir):
        """
        setting the dataset-stimuli directory
        """
        self.stimuli_dir = stimuli_dir
        self.data_dirs = []
        print(self.stimuli_dir)
    
        for placement in ['far_large', 'far_small', 'near_large', 'near_small']:
            globals()[placement] = os.path.join(self.stimuli_dir, placement)
            self.data_dirs.append(globals()[placement])
            
        # image to tensor and resize
        self.data_transforms = {
            'test': transforms.Compose([
                transforms.Resize(self.input_dim, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()])}
        

    def predict(self):
        """
        apply model to dataset and assign label
        """
        
        placement = str()
        congruent = False
        self.df = pd.DataFrame()
        self.file_paths_total = []
        
        self.zero_start_time = time.time() # duration check
        
        zero_start_time = time.time()
    
        for dir_ in self.data_dirs:
            print(dir_)
    
            if dir_.endswith('far_large'):
                placement = 'far_large'
                congruent = False
            if dir_.endswith('far_small'):
                placement = 'far_small'
                congruent = True
            if dir_.endswith('near_large'):
                placement = 'near_large'
                congruent = True
            if dir_.endswith('near_small'):
                placement = 'near_small'
                congruent = False
            
            
            # dataset 
            self.image_dataset = datasets.ImageFolder(dir_, self.data_transforms['test'])
            self.class_names = self.image_dataset.classes
            
            # dataloader
            data_loader = torch.utils.data.DataLoader(self.image_dataset,
                                                          batch_size=7,
                                                          shuffle=False,
                                                          num_workers=0)
            
            file_paths = data_loader.sampler.data_source.imgs
    
            for file_path_sample in file_paths:
                file_path_sample = re.sub(f"{dir_}/.*/", "", file_path_sample[0])
                file_path_sample = re.sub(".png", "", file_path_sample)
                self.file_paths_total.append(file_path_sample)
    
            for i, data in enumerate(data_loader):
    
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # no backprop  while testing
                with torch.no_grad():
                    outputs = torch.nn.functional.softmax(self.model(inputs), dim=1) # prediction
                    
                # > numpy > pandas
                output_np = outputs.cpu().numpy()
                df_output = pd.DataFrame(output_np)
                
                df_output['label'] = labels.cpu()
                
                df_output['congruent'] = congruent
                df_output['placement'] = placement
                
                self.df = pd.concat([self.df, df_output], ignore_index=True)
                print(f"iteration {i+1}/12 of {placement}")
    
        print(f"total time: {((time.time() - zero_start_time)/60):.2f} minutes")

    
    def wrangle(self, imgnet_labels, imgnet_sysnets, sysnets_car, sysnets_noise):
        """
        wrangle into a dataframe with softmax scores and labels
        """
        
        self.imgnet_labels = imgnet_labels
        self.imgnet_sysnets = imgnet_sysnets
        self.sysnets_car = sysnets_car
        self.sysnets_noise = sysnets_noise
        
        
        with open(self.imgnet_labels) as f:
            classes = [line.strip() for line in f.readlines()]
            
            label_names = []
            for i in classes:
                label_name = re.sub(".*, ", "", i)
                label_names.append(label_name)
                
            label_indices = []
            for i in classes:
                label_idx = re.sub("\, .*", "", i)
                label_indices.append(label_idx)
                
        with open(self.imgnet_sysnets) as f:
            sysnets = [line.strip() for line in f.readlines()]
            
            
        label_names.extend([np.nan, np.nan, np.nan])
        label_indices.extend([np.nan, np.nan, np.nan])
            
        sysnets.append('label')
        sysnets.append('congruent')
        sysnets.append('placement')

        sys_classes = zip(sysnets, label_indices, label_names)
        sys_index = list(sys_classes)

        multi_index = pd.MultiIndex.from_tuples(sys_index, names=["sysnet", "idx", "label"])
            
        self.df.columns = multi_index
        
        class_names = self.image_dataset.classes

        labels = self.df['label', np.nan, np.nan]
        label_names = []
        new_name = str()

        for i in labels:
            for j in range(17):
                if i == j:
                    new_name = class_names[j]
                    label_names.append(new_name)
                    
        self.df['label_name'] = label_names
        self.df['scene_ID'] = self.file_paths_total
        self.df = self.df[self.df['label', np.nan, np.nan] != 10] # delete humans
        
        # car ID's
        df_car = self.df.loc[:, (self.sysnets_car, slice(None), slice(None))]
        car_multiidxs = list(df_car.columns)
        car_idxs = []
        for col_tup in car_multiidxs:
            car_idx = col_tup[1]
            car_idxs.append(car_idx)
            
        car_idxs = list(map(int, car_idxs))


        no_car_idxs = list(range(0,1000,1))
        no_car_idxs = SysnetProcessing.subtraction(no_car_idxs, car_idxs)
        
        # set all noise labels to zero softmax score
        for col in self.sysnets_noise:
            self.df.loc[:, (col, slice(None), slice(None))] = 0
        
        
        for row_index, row in self.df.iterrows():
            # ape
            if row['label', np.nan, np.nan] == 0:
                sum_correct = row.iloc[365:384].sum()
                self.df.loc[row_index, 'correct_sum'] = sum_correct
                
                sum_incorrect = row.iloc[np.r_[0:365, 384:1000]].sum()
                self.df.loc[row_index, 'incorrect_sum'] = sum_incorrect
                
            # bear
            if row['label', np.nan, np.nan] == 1:
                sum_correct = row.iloc[np.r_[294, 295, 296, 297, 850]].sum()
                self.df.loc[row_index, 'correct_sum'] = sum_correct
                
                sum_incorrect = row.iloc[np.r_[0:294, 298:850, 851:1000]].sum()
                self.df.loc[row_index, 'incorrect_sum'] = sum_incorrect
                
            # bench
            if row['label', np.nan, np.nan] == 2:
                sum_correct = row.iloc[703] #.sum()
                self.df.loc[row_index, 'correct_sum'] = sum_correct
                
                sum_incorrect = row.iloc[np.r_[0:703, 704:1000]].sum()
                self.df.loc[row_index, 'incorrect_sum'] = sum_incorrect
                
            # bin
            if row['label', np.nan, np.nan] == 3:
                sum_correct = row.iloc[412] #.sum()
                self.df.loc[row_index, 'correct_sum'] = sum_correct
                
                sum_incorrect = row.iloc[np.r_[0:412, 413:1000]].sum()
                self.df.loc[row_index, 'incorrect_sum'] = sum_incorrect
                
            # boar
            if row['label', np.nan, np.nan] == 4:
                sum_correct = row.iloc[341:344].sum()
                self.df.loc[row_index, 'correct_sum'] = sum_correct
                
                sum_incorrect = row.iloc[np.r_[0:341, 344:1000]].sum()
                self.df.loc[row_index, 'incorrect_sum'] = sum_incorrect

            # car 
            if row['label', np.nan, np.nan] == 5:
                sum_correct = row.iloc[np.r_[car_idxs]].sum()
                self.df.loc[row_index, 'correct_sum'] = sum_correct
                
                sum_incorrect = row.iloc[np.r_[no_car_idxs]].sum()
                self.df.loc[row_index, 'incorrect_sum'] = sum_incorrect
            
            # chair
            
            # cow
            if row['label', np.nan, np.nan] == 7:
                sum_correct = row.iloc[345:348].sum()
                self.df.loc[row_index, 'correct_sum'] = sum_correct
                
                sum_incorrect = row.iloc[np.r_[0:345, 348:1000]].sum()
                self.df.loc[row_index, 'incorrect_sum'] = sum_incorrect
                
            # dog
            if row['label', np.nan, np.nan] == 8:
                sum_correct = row.iloc[151:270].sum()
                self.df.loc[row_index, 'correct_sum'] = sum_correct
                
                sum_incorrect = row.iloc[np.r_[0:151, 270:1000]].sum()
                self.df.loc[row_index, 'incorrect_sum'] = sum_incorrect
                
            # horse 
            if row['label', np.nan, np.nan] == 9:
                sum_correct = row.iloc[np.r_[339, 603]].sum()
                self.df.loc[row_index, 'correct_sum'] = sum_correct
                
                sum_incorrect = row.iloc[np.r_[0:339, 340:603, 604:1000]].sum()
                self.df.loc[row_index, 'incorrect_sum'] = sum_incorrect
            # human
            
            # kangaroo
            if row['label', np.nan, np.nan] == 11:
                sum_correct = row.iloc[104] #.sum()
                self.df.loc[row_index, 'correct_sum'] = sum_correct
                
                sum_incorrect = row.iloc[np.r_[0:104, 105:1000]].sum()
                self.df.loc[row_index, 'incorrect_sum'] = sum_incorrect
                
            # motorcycle 
            if row['label', np.nan, np.nan] == 12:
                sum_correct = row.iloc[np.r_[665, 670]].sum()
                self.df.loc[row_index, 'correct_sum'] = sum_correct
                
                sum_incorrect = row.iloc[np.r_[0:665, 666:670, 671:1000]].sum()
                self.df.loc[row_index, 'incorrect_sum'] = sum_incorrect
            
            # sign 
            # suitcase 
            # table 

            # wheelbarrow
            if row['label', np.nan, np.nan] == 16:
                sum_correct = row.iloc[428] #.sum()
                self.df.loc[row_index, 'correct_sum'] = sum_correct
                
                sum_incorrect = row.iloc[np.r_[0:428, 429:1000]].sum()
                self.df.loc[row_index, 'incorrect_sum'] = sum_incorrect
        

        self.df['correct_and_incorrect'] = self.df['correct_sum'] + self.df['incorrect_sum']
        self.df['correct_sum'] = self.df['correct_sum'] / self.df['correct_and_incorrect']
        self.df['incorrect_sum'] = self.df['incorrect_sum'] / self.df['correct_and_incorrect']
        
        
        # top 10
        self.df['top10'] = str()
        list_of_fc_labels = []

        for i in range(len(self.df)):
            fc_series = self.df.iloc[i, 0:1000]
            fc_series_sorted = fc_series.sort_values(ascending=False)
            fc_series_sorted = fc_series_sorted[0:10]
            fc_series_idx = fc_series_sorted.index.values.tolist()
            
            
            fc_labels = []
            for pred_label in fc_series_idx:
                # pred_label = re.sub(".*, ", "", pred_label)
                fc_labels.append(pred_label[2])
            
            list_of_fc_labels.append(fc_labels)
                
        self.df['top10'] = list_of_fc_labels
        
        
        return self.df
    
    def diff_wrangle(self):
        """
        wrangle into a dataframe with difference scores per scene
        """
        
        df_cutted = self.df[['placement', 'label_name', 'scene_ID', 'correct_sum']]
        df_cutted = df_cutted[df_cutted['correct_sum'].notnull()]
        df_cutted.columns = df_cutted.columns.droplevel(1)
        df_cutted.columns = df_cutted.columns.droplevel(1)

        differences = []
        scene_idss = df_cutted['scene_ID'].tolist()
        scene_idss = set(scene_idss)

        for i in scene_idss:
            df_scene = df_cutted[df_cutted['scene_ID'] == i]
            
            near_l = float(df_scene.loc[df_scene['placement'] == 'near_large']['correct_sum'])
            far_l = float(df_scene.loc[df_scene['placement'] == 'far_large']['correct_sum'])
            
            far_s = float(df_scene.loc[df_scene['placement'] == 'far_small']['correct_sum'])
            near_s = float(df_scene.loc[df_scene['placement'] == 'near_small']['correct_sum'])
            
            diff_large = near_l - far_l
            diff_small = far_s - near_s

            label = df_scene['label_name'].iloc[0]
            
            differences.append([i, label, near_l, far_l, far_s, near_s, diff_large, diff_small])
            
        df_differences = pd.DataFrame(differences, columns=['scene_ID', 'label_name', 'near_l', 'far_l', 'far_s', 'near_s', 'diff_large', 'diff_small'])
        
        
        df_differences = df_differences.sort_values('label_name')
        df_differences['positive_s'] = df_differences['diff_small'] > 0
        df_differences['positive_l'] = df_differences['diff_large'] > 0

        df_differences['label_name'] = df_differences['label_name'].astype(str)
        df_differences['scene_ID'] = df_differences['scene_ID'].astype(str)
        df_differences['name_idx'] = df_differences['label_name'] + '_' + df_differences['scene_ID']
        
        # -- natural log transformation -- #
        df_differences['log_diff_large_pos'] = (-1 * (np.log(df_differences['diff_large']))).fillna(0)
        df_differences['log_diff_large_neg'] = (np.log(-1 * df_differences['diff_large'])).fillna(0)
        df_differences['log_diff_large'] = df_differences['log_diff_large_pos'] + df_differences['log_diff_large_neg']
        
        df_differences['log_diff_small_pos'] = (-1 * (np.log(df_differences['diff_small']))).fillna(0)
        df_differences['log_diff_small_neg'] = (np.log(-1 * df_differences['diff_small'])).fillna(0)
        df_differences['log_diff_small'] = df_differences['log_diff_small_pos'] + df_differences['log_diff_small_neg']
        
        df_log_diff = df_differences[['log_diff_large', 'log_diff_small']]
        df_log_diff.rename(columns={'log_diff_large':'diff_large', 'log_diff_small':'diff_small'}, inplace=True)

        return df_differences, df_log_diff
    
    
class SysnetProcessing:
    """
    processing the wordnet hierarchy to combine labels and remove noise
    """
        
    def make_sysnet_list(self, super_sysnet, found_ids):
        self.super_sysnet = super_sysnet
        self.found_ids = found_ids
        
        if isinstance(self.super_sysnet, dict):
            found_ids.append(self.super_sysnet['id'])
            
            if 'children' in self.super_sysnet:
                new_input = self.super_sysnet['children']
                self.make_sysnet_list(new_input, found_ids) # recursive
        
        if isinstance(self.super_sysnet, list):
            for item in self.super_sysnet:
                if isinstance(item, dict):
                    found_ids.append(item['id'])
                    
                    if 'children' in item:
                        new_input = item['children']
                        self.make_sysnet_list(new_input, found_ids) # recursive            
        return found_ids

    @staticmethod
    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3


    @staticmethod
    def subtraction(lst1, lst2):
        lst3 = [value for value in lst1 if value not in lst2]
        return lst3
            
    
    def sysnet_application(self, imgnet_sysnets, imgnet_tree):
        self.imgnet_sysnets = imgnet_sysnets
        self.imgnet_tree = imgnet_tree
        
        with open(self.imgnet_sysnets) as f:
            sysnets = [line.strip() for line in f.readlines()]

        with open(self.imgnet_tree) as json_data:
            data = json.load(json_data)
        
        wordnet = pd.json_normalize(data['children'])
        
        # -- animacy -- #
        organism  = wordnet.loc[20, 'children']
        found_ids = [] 
        organis_nodes = self.make_sysnet_list(organism, found_ids)
        sysnets_organism = SysnetProcessing.intersection(sysnets, organis_nodes)

        animal = organism[3]
        found_ids = [] 
        animal_nodes = self.make_sysnet_list(animal, found_ids)
        sysnets_animacy = SysnetProcessing.intersection(sysnets, animal_nodes)

        sysnets_organism_clean = SysnetProcessing.subtraction(sysnets_organism, sysnets_animacy)

        # -- inanimacy -- #
        artifact = wordnet.loc[19, 'children']
        structure = artifact[1]
        found_ids = [] 
        structure_nodes = self.make_sysnet_list(structure, found_ids)
        sysnets_structure = SysnetProcessing.intersection(sysnets, structure_nodes)

        found_ids = [] 
        artifact_nodes = self.make_sysnet_list(artifact, found_ids)
        sysnets_artifact = SysnetProcessing.intersection(sysnets, artifact_nodes)

        sysnets_inanimacy = SysnetProcessing.subtraction(sysnets_artifact, sysnets_structure)
        sysnets_inanimacy.extend(['n06794110', 'n06874185']) # street sign, traffic light

        # -- noise -- #
        total_remainder = []
        for i in range(len(wordnet.loc[0:18, 'children'])):
            
            remainder = wordnet.loc[i, 'children']
            found_ids = [] # empty
            remaind = self.make_sysnet_list(remainder, found_ids)
            total_remainder += remaind
        total_remain = SysnetProcessing.intersection(sysnets, total_remainder)
        loose_sysnets = wordnet.loc[0:7, 'id']
        total_remain.extend(loose_sysnets)
        sysnets_remainder = SysnetProcessing.subtraction(total_remain, sysnets_inanimacy)

        sysnets_noise = sysnets_organism_clean + sysnets_remainder + sysnets_structure
        
        # print(f'total sysnets of 1000 should be: {len(sysnets_animacy) + len(sysnets_organism_clean) + len(sysnets_inanimacy) + len(sysnets_remainder) + len(sysnets_structure)}')
        
        sys_instrumentality = artifact[0]['children']
        sys_container = sys_instrumentality[2]['children']
        sys_wheeled_vehicle = sys_container[0]['children']
        sys_self_propelled_vehicle = sys_wheeled_vehicle[4]['children']
        sys_motor_vehicle = sys_self_propelled_vehicle[2]['children']

        found_ids = []
        car_nodes = self.make_sysnet_list(sys_motor_vehicle, found_ids)
        car_nodes = SysnetProcessing.intersection(sysnets, car_nodes)
        
        return sysnets_noise, car_nodes


class ObjectPlotting:
    """
    visualization with matplotlib
    """
    
    def plot_init(self, df, plot_dir, model_name, df_diff, df_log_diff, stim_set):
        
        self.df = df
        self.plot_dir = plot_dir
        self.model_name = model_name
        self.df_diff = df_diff
        self.df_log_diff = df_log_diff
        self.stim_set = stim_set
        
        self.animate_labels = ['Ape', 'Bear', 'Boar', 'Cow', 'Dog', 'Horse', 'Kangaroo'] # 'Human'
        self.inanimate_labels = ['Bench', 'Bin', 'Car', 'Chair', 'Motorcycle', 'Sign', 'Suitcase', 'Table', 'Wheelbarrow']
    
        self.plot_model_dir = os.path.join(self.plot_dir, self.model_name)
        map_isExist = os.path.exists(self.plot_model_dir)
        if not map_isExist:
            os.makedirs(self.plot_model_dir)
        
    
    def outlier_plot(self):
        """
        softmax scores for individual instances formatted per category 
        """

        fig, axes = plt.subplots(ncols=1, figsize=(20,14))
        ax1 = axes

        label_collection = [self.animate_labels, self.inanimate_labels]
        colors = {True:'blue', False:'red'}
        sizes = {'far_small':100, 'far_large':300, 'near_small':100, 'near_large':300}

        i = 0
        stim_labels = []
        x_positions = []
        y_positions = []
        scene_ids = []

        for in_or_an_label in label_collection:
            for stim_class in in_or_an_label:
                df_single_stim = self.df.loc[self.df['label_name'] == stim_class]
                df_single_stim.reset_index(inplace=True)
                
                x = np.random.uniform(-0.1, 0.1, len(df_single_stim)) + i
                y = df_single_stim['correct_sum']
                ax1.scatter(x, y,\
                            marker = 'x',\
                            c = [colors[j] for j in df_single_stim.loc[:, ('congruent', np.nan, np.nan)]],\
                            s = [sizes[k] for k in df_single_stim.loc[:, ('placement', np.nan, np.nan)]])
               
                i += 1
                stim_labels.append(stim_class)
                x_positions.append(x)
                y_positions.append(y)
                scene_ids.append(df_single_stim['scene_ID'])

        # -- labels -- #
        x_positions = [item for sublist in x_positions for item in sublist]
        y_positions = [item for sublist in y_positions for item in sublist]
        scene_ids = [item for sublist in scene_ids for item in sublist]
        texts = []

        for x, y, s in zip(x_positions, y_positions, scene_ids):
            texts.append(ax1.text(x, y, s))
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black',  lw=0.4), force_points=(0.3,5.0),\
                    force_objects=(0.4,1), fontsize=16)

        ax1.set_xticks(np.arange(0,16, 1))
        ax1.set_xticklabels(stim_labels, rotation=45)

        ax1.set_yticks(np.arange(0, 1.1, 0.1))
        ax1.set_yticklabels(np.round(np.arange(0,1.1, 0.1),2))


        ax1.grid(True)
        ax1.set_title(f'{self.model_name} softmax scores normalized',\
                      fontsize=16)
            
        blue_patch = mpatches.Patch(color='blue', label='Congruent')
        red_patch = mpatches.Patch(color='red', label='Incongruent')

        big_cross = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                                  markersize=18, label='Large examples')

        small_cross = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                                  markersize=9, label='Small examples')

        ax1.legend(handles=[blue_patch, red_patch, big_cross, small_cross], fontsize=16)

        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_model_dir, 'cat_softmax_normalized.png'))
        plt.show()
    
    
    def aggr_barplot(self):
        """
        aggregated softmax scores for {far_large, far_small, near_large, near_small}
        """
        
        df_correct_sum_grouped = self.df.groupby([('placement', np.nan, np.nan)])['correct_sum'].mean()
        df_correct_sum_grouped = pd.DataFrame(df_correct_sum_grouped)
        df_correct_sum_grouped['pushed_index'] = [1, 2, 0, 3]    
        df_correct_sum_grouped.sort_values('pushed_index', inplace=True)

        x_pos = np.arange(len(df_correct_sum_grouped))
        fig, ax = plt.subplots(figsize=(3, 12))
        ax.bar(x_pos, df_correct_sum_grouped['correct_sum'], color = ['blue', 'red', 'blue', 'red'])
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_xticks(np.arange(0, 4, 1))
        ax.set_xticklabels(df_correct_sum_grouped.index, rotation=90)
        ax.set_title(f'{self.model_name}_softmax scores \n label vs (all - noise)')

        blue_patch = mpatches.Patch(color='blue', label='Congruent')
        red_patch = mpatches.Patch(color='red', label='Incongruent')
        ax.legend(handles=[blue_patch, red_patch], fontsize=10)


        plt.savefig(os.path.join(self.plot_model_dir, 'aggregated_bar_plot.png'))
        plt.show()
        
        self.acc_large = (df_correct_sum_grouped['correct_sum'].iloc[0] + df_correct_sum_grouped['correct_sum'].iloc[1])/2
        self.acc_small = (df_correct_sum_grouped['correct_sum'].iloc[2] + df_correct_sum_grouped['correct_sum'].iloc[3])/2
        self.acc_all = (self.acc_large + self.acc_small)/2

    def diff_plot(self):
        """
        outlier plot with a difference score visualization
        {far_small - near_small} and {near_large - far_large}
        """
        

        
        y_limss = [[0,1], [0.9,1], [0.99,1]]
        y_tickss = [np.round(np.arange(0, 1.1, 0.1),2), np.round(np.arange(0.9, 1, 0.01),2),\
                    np.round(np.arange(0.99, 1, 0.001),3)]
        
        # show different granularities
        for y_l, y_t in zip(y_limss, y_tickss):
            
            fig, axes = plt.subplots(ncols=1, figsize=(24,14))
            ax1 = axes
            
            x = self.df_diff['name_idx']
            y = self.df_diff['near_l']
            z = self.df_diff['far_l']
                
            s = self.df_diff['far_s']
            t = self.df_diff['near_s']

            
            feight = range(58)
            hsix = np.arange(0,58,0.5)
            
            min_hsix = []
            for i in hsix:
                i += -0.25
                min_hsix.append(i)
            
            min_feight = []
            for i in feight:
                i += -0.25
                min_feight.append(i)
            
            max_feight = []
            for i in feight:
                i += +0.25
                max_feight.append(i)
            
            x_labels = [val for val in x for _ in (0, 1)]
            
            ax1.set_ylim(y_l)
            ax1.set_xticks(min_hsix)
            ax1.set_xticklabels(x_labels, rotation=90)
            ax1.set_yticks(y_t)
            ax1.set_yticklabels(y_t, rotation=90)
            
            ax1.set_ylabel(f'{self.model_name} difference plot congruent - incongruent [0.0 - 1.0]', fontsize=16)
            
            for i, j, k in zip(min_feight, y, z):
                if j > k:
                    ax1.vlines(x=i, ymin=k, ymax=j, color='blue', linewidth=1.4)
                if k > j:
                    ax1.vlines(x=i, ymin=j, ymax=k, color='red', linewidth=1.4)
                    
            for i, j, k in zip(max_feight, s, t):
                if j > k:
                    ax1.vlines(x=i, ymin=k, ymax=j, color='blue', linewidth=1.4, linestyle='dashed')
                if k > j:
                    ax1.vlines(x=i, ymin=j, ymax=k, color='red', linewidth=1.4, linestyle='dashed')

            
            ax1.grid(True)
            ax1.scatter(min_feight,y, color = 'blue', marker='x', s=200)
            ax1.scatter(min_feight,z, color = 'red', marker='x', s=200)
            
            
            ax1.scatter(max_feight,s, color = 'blue', marker='x', s=50)
            ax1.scatter(max_feight,t, color = 'red', marker='x', s=50)

            # only save this scale into png
            if y_l == [0,1]:
                plt.savefig(os.path.join(self.plot_model_dir, f'difference_plot_{y_l}.png'), dpi=200)
            plt.show()
            
            
    @staticmethod
    def mean_confidence_interval(data, confidence=0.9):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        h_min = m - h
        h_max = m + h
        return h_min, h_max
    

    def diff_boxplot(self):
        """
        boxplot for all-, large- and small examples with confidence interval
        + log boxplot
        """
        

        
        for df, y_scale, plot_name in zip([self.df_diff, self.df_log_diff], [[-0.5,0.5],[-5,5]], ['not_transformed', 'natural_log']):
            differences_large = df['diff_large']
            differences_small = df['diff_small']
            differences_both = pd.concat([differences_large, differences_small])
            
            both_hmin, both_hmax = ObjectPlotting.mean_confidence_interval(differences_both)
            small_hmin, small_hmax = ObjectPlotting.mean_confidence_interval(differences_small)
            large_hmin, large_hmax = ObjectPlotting.mean_confidence_interval(differences_large)
            
            fig, axes = plt.subplots(ncols=1, figsize=(24,14))
            ax1 = axes
            ax1.grid(True)
            
            meanpointprops = dict(marker='_', markersize=30)
    
            ax1.boxplot([differences_both, differences_small, differences_large], showmeans=True, meanprops=meanpointprops)
            ax1.set_xticklabels(['small and large', 'small', 'large'], rotation=0, fontsize=16)
            ax1.set_ylim(y_scale)
            ax1.set_title(f'{self.model_name} - {plot_name} - mean of the difference values, 90% confidence interval', fontsize=16)
            ax1.hlines(0, color='blue', linewidth=1, xmin=0.5, xmax=3.5)
            
            
            ax1.vlines(1, color='r', ymin=both_hmin, ymax=both_hmax)
            ax1.scatter(1, both_hmin, c='k', label='lower boundary')
            ax1.text(1.05, both_hmin, f'lower boundary {np.round(both_hmin,3)}')
            ax1.scatter(1, both_hmax, c='k', label='upper boundary')
            ax1.text(1.05, both_hmax, f'upper boundary {np.round(both_hmax,3)}')
            ax1.text(1.05, np.mean(differences_both), f'mean {np.round(np.mean(differences_both),4)}')
              
            ax1.vlines(2, color='r', ymin=small_hmin, ymax=small_hmax)
            ax1.vlines(2, color='r', ymin=small_hmin, ymax=small_hmax)
            ax1.scatter(2, small_hmin, c='k', label='lower boundary')
            ax1.text(2.05, small_hmin, f'lower boundary {np.round(small_hmin,3)}')
            ax1.scatter(2, small_hmax, c='k', label='upper boundary')
            ax1.text(2.05, small_hmax, f'upper boundary {np.round(small_hmax,3)}')
            ax1.text(2.05, np.mean(differences_small), f'mean {np.round(np.mean(differences_small),4)}')
            
            
            ax1.vlines(3, color='r', ymin=large_hmin, ymax=large_hmax)
            ax1.vlines(3, color='r', ymin=large_hmin, ymax=large_hmax)
            ax1.scatter(3, large_hmin, c='k', label='lower boundary')
            ax1.text(3.05, large_hmin, f'lower boundary {np.round(large_hmin,3)}')
            ax1.scatter(3, large_hmax, c='k', label='upper boundary')
            ax1.text(3.05, large_hmax, f'upper boundary {np.round(large_hmax,3)}')
            ax1.text(3.05, np.mean(differences_large), f'mean {np.round(np.mean(differences_large),4)}')
    
            plt.savefig(os.path.join(self.plot_model_dir, f'mean_difference_{plot_name}.png'), dpi=200)
            plt.show()

    
    def statistical_check(self):
        """
        one sided
        if the mean of the underlying distribution in the observed outcome would be 0 in reality
        the possibility of seeing this result or higher would be (pvalue * 100%)
        """
        statistic_outcomes = []
        
        for df_temp, transform in zip([self.df_diff, self.df_log_diff], ['no_transform', 'natural_log']):
            
            diff_both = pd.concat([df_temp['diff_small'], df_temp['diff_large']], axis=0)
            df_perms = pd.DataFrame()
            observed_means = []
            
    
            for idx, example_set, name in zip([0,1,2], [diff_both, df_temp['diff_small'], df_temp['diff_large']],\
                               ['all', 'small', 'large']):
                
                observed_mean = np.mean(example_set)
                observed_means.append(observed_mean)
                permutated_means = []
                
                for j in range(10000):
                    
                    permutations_temp = []
                    for k in example_set:
                        new_k = k * np.random.choice([-1,1])
                        permutations_temp.append(new_k)
                    
                    temp_perm_mean = np.mean(permutations_temp)
                    permutated_means.append(temp_perm_mean)
    
                df_perms[name] = permutated_means
    
                z_score = (observed_means[idx] - np.mean(df_perms[name])) / np.std(df_perms[name])
                p_value = scipy.stats.norm.sf(z_score)
                
                if name == 'all':
                    acc_score = self.acc_all
                if name == 'small':
                    acc_score = self.acc_small
                if name == 'large':
                    acc_score = self.acc_large
                    
                
                statistic_outcome = [self.model_name, transform, name, acc_score, observed_mean, z_score, p_value, self.stim_set]
                statistic_outcomes.append(statistic_outcome)
                
            df_pvalue = pd.DataFrame(statistic_outcomes, columns = ['model_name', 'transform', 'stim_size', 'accuracy', 'difference_score', 'z_score', 'p_value', 'stimuli_dir'])
        
        df_pvalue.to_csv(os.path.join(self.plot_model_dir, 'pvalues.csv'))
        
        
        df_model_comparison = df_pvalue
        df_no_transform = df_model_comparison[df_model_comparison['transform'] == 'no_transform']
        df_natural_log = df_model_comparison[df_model_comparison['transform'] == 'natural_log']
        df_natural_log = df_natural_log[['accuracy', 'difference_score', 'p_value']]
        df_natural_log = df_natural_log.rename(columns={'accuracy':'log_acc', 'difference_score':'log_diff', 'p_value':'log_p'})
        df_natural_log.reset_index(inplace=True, drop=True)
        df_no_transform.reset_index(inplace=True, drop=True)
        df_model_comp = pd.concat([df_no_transform, df_natural_log], axis=1)
        
        return df_model_comp
    
# ---- CORNET ---- #
class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_Z(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2)
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        x = self.conv(inp)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.output(x)  # for an easy access to this block's output
        return x


def CORnet_Z():
    model = nn.Sequential(OrderedDict([
        ('V1', CORblock_Z(3, 64, kernel_size=7, stride=2)),
        ('V2', CORblock_Z(64, 128)),
        ('V4', CORblock_Z(128, 256)),
        ('IT', CORblock_Z(256, 512)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ])))
    ]))

    # weight initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model


class CORblock_RT(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, out_shape=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_shape = out_shape

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=kernel_size // 2)
        self.norm_input = nn.GroupNorm(32, out_channels)
        self.nonlin_input = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp=None, state=None, batch_size=None):
        if inp is None:  # at t=0, there is no input yet except to V1
            inp = torch.zeros([batch_size, self.out_channels, self.out_shape, self.out_shape])
            device = torch.device('mps')
            inp.to(device)
        else:
            inp = self.conv_input(inp)
            inp = self.norm_input(inp)
            inp = self.nonlin_input(inp)

        if state is None:  # at t=0, state is initialized to 0
            state = 0
        skip = inp + state

        x = self.conv1(skip)
        x = self.norm1(x)
        x = self.nonlin1(x)

        state = self.output(x)
        output = state
        return output, state


class CORnet_RT(nn.Module):

    def __init__(self, times=5):
        super().__init__()
        self.times = times

        self.V1 = CORblock_RT(3, 64, kernel_size=7, stride=4, out_shape=56)
        self.V2 = CORblock_RT(64, 128, stride=2, out_shape=28)
        self.V4 = CORblock_RT(128, 256, stride=2, out_shape=14)
        self.IT = CORblock_RT(256, 512, stride=2, out_shape=7)
        self.decoder = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000))
        ]))

    def forward(self, inp):
        outputs = {'inp': inp}
        states = {}
        blocks = ['inp', 'V1', 'V2', 'V4', 'IT']

        for block in blocks[1:]:
            if block == 'V1':  # at t=0 input to V1 is the image
                this_inp = outputs['inp']
            else:  # at t=0 there is no input yet to V2 and up
                this_inp = None
                
            # device = torch.device('mps')
            # this_inp.to(device)
            
            new_output, new_state = getattr(self, block)(this_inp, batch_size=len(outputs['inp']))
            outputs[block] = new_output
            states[block] = new_state

        for t in range(1, self.times):
            new_outputs = {'inp': inp}
            for block in blocks[1:]:
                prev_block = blocks[blocks.index(block) - 1]
                prev_output = outputs[prev_block]
                prev_state = states[block]
                new_output, new_state = getattr(self, block)(prev_output, prev_state)
                new_outputs[block] = new_output
                states[block] = new_state
            outputs = new_outputs

        out = self.decoder(outputs['IT'])
        return out
    

class CORblock_S(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output


def CORnet_S():
    model = nn.Sequential(OrderedDict([
        ('V1', nn.Sequential(OrderedDict([  # this one is custom to save GPU memory
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('nonlin1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('nonlin2', nn.ReLU(inplace=True)),
            ('output', Identity())
        ]))),
        ('V2', CORblock_S(64, 128, times=2)),
        ('V4', CORblock_S(128, 256, times=4)),
        ('IT', CORblock_S(256, 512, times=2)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ])))
    ]))

    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        # nn.Linear is missing here because I originally forgot 
        # to add it during the training of this network
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model

#%% preprocessing
imgnet_labels =     '/Users/cees/Desktop/oic_march/ext_data/imagenet_classes.txt'
imgnet_sysnets =    '/Users/cees/Desktop/oic_march/ext_data/wordnet_ids.txt'
imgnet_tree =       '/Users/cees/Desktop/oic_march/ext_data/imagenet_tree.json'

# using the wordnet hierarchy to combine and divide imagenet labels
sysnets_noise_ = SysnetProcessing()
sysnets_noise, sysnets_car = sysnets_noise_.sysnet_application(imgnet_sysnets=imgnet_sysnets,\
                                                               imgnet_tree=imgnet_tree)
#%% timm library exploration
avail_pretrained_models = timm.list_models('*[4-9][0-9][0-9]*', pretrained=True)
print(avail_pretrained_models)

#%%
model_name = 'crossvit_18_dagger_408'
model_weights = None
input_dim = 408
model_type = 'timm'
model_url = None
crossvit_18_org = [model_name, model_weights, input_dim, model_type, model_url]

model_name = 'crossvit_18_dagger_408'
state_dict = '/Users/cees/Desktop/oic_march/packages/crossvit_brainscore_rot.pt'
model_weights = state_dict
input_dim = 408
model_type = 'timm'
model_url = None
crossvit_18_bs_rot = [model_name, model_weights, input_dim, model_type, model_url]

model_name = 'crossvit_18_dagger_408'
state_dict = '/Users/cees/Desktop/oic_march/packages/crossvit_brainscore_adv.pt'
model_weights = state_dict
input_dim = 408
model_type = 'timm'
model_url = None
crossvit_18_bs_adv = [model_name, model_weights, input_dim, model_type, model_url]

model_name = 'crossvit_18_dagger_408'
state_dict = '/Users/cees/Desktop/oic_march/packages/crossvit_brainscore_both.pt'
model_weights = state_dict
input_dim = 408
model_type = 'timm'
model_url = None
crossvit_18_bs_both = [model_name, model_weights, input_dim, model_type, model_url]

model_name = 'resnetrs420'
model_weights = None
input_dim = 420
model_type = 'timm'
model_url = None
resnet = [model_name, model_weights, input_dim, model_type, model_url]

model_name = 'beit_large_patch16_512'
model_weights = None
input_dim = 512
model_type = 'timm'
model_url = None
beit = [model_name, model_weights, input_dim, model_type, model_url]

model_name = 'volo_d5_512'
model_weights = None
input_dim = 512
model_type = 'timm'
model_url = None
volo = [model_name, model_weights, input_dim, model_type, model_url]

model_name = 'efficientnet_v2_l'
model_weights = EfficientNet_V2_L_Weights.DEFAULT
input_dim = 480
model_type = 'pytorch'
model_url = None
efficientnetv2 = [model_name, model_weights, input_dim, model_type, model_url]

model_name = 'vgg16'
model_weights = VGG16_Weights.DEFAULT
input_dim = 224
model_type = 'pytorch'
model_url = None
vgg16 = [model_name, model_weights, input_dim, model_type, model_url]

# cornets #
model_name = 'cornet_z'
model_weights = None
input_dim = 224
model_type = 'cornet_z'
model_url = 'https://s3.amazonaws.com/cornet-models/cornet_z-5c427c9c.pth'
cornet_z = [model_name, model_weights, input_dim, model_type, model_url]

model_name = 'cornet_rt'
model_weights = None
input_dim = 224
model_type = 'cornet_rt'
model_url = 'https://s3.amazonaws.com/cornet-models/cornet_rt-933c001c.pth'
cornet_rt = [model_name, model_weights, input_dim, model_type, model_url]

model_name = 'cornet_s'
model_weights = None
input_dim = 224
model_type = 'cornet_s'
model_url = 'https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth'
cornet_s = [model_name, model_weights, input_dim, model_type, model_url]

model_name = 'hf_hub:timm/maxvit_base_tf_512.in21k_ft_in1k'
model_weights = None
input_dim = 512
model_type = 'timm'
model_url = None
maxvit = [model_name, model_weights, input_dim, model_type, model_url]

model_name = 'vit_h_14'
model_weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
input_dim = 518
model_type = 'pytorch'
model_url = None
vit = [model_name, model_weights, input_dim, model_type, model_url]


#%% application
def one_model_one_dataset(model, stimuli_dir, result_dir):
    # model input variables
    model_name = model[0]
    model_weights = model[1]
    input_dim = model[2]
    model_type = model[3]
    model_url = model[4]
    
    # prediction & wrangling
    analysis = ObjectAnalysis()
    analysis.model_init(model_name=model_name, input_dim=input_dim, model_type=model_type,\
                        weights=model_weights, model_url=model_url) 
    analysis.set_dirs(stimuli_dir=stimuli_dir)
    analysis.predict()
    df = analysis.wrangle(imgnet_labels=imgnet_labels, imgnet_sysnets=imgnet_sysnets,\
                          sysnets_car=sysnets_car, sysnets_noise=sysnets_noise)
    df_diff, df_log_diff = analysis.diff_wrangle()
    
    # visualising & statistics
    plots = ObjectPlotting()
    plots.plot_init(df=df, plot_dir=result_dir, model_name=model_name, df_diff=df_diff,\
                    df_log_diff=df_log_diff, stim_set=stimuli_dir)
        
    # plots.outlier_plot()
    plots.aggr_barplot()
    plots.diff_plot()
    plots.diff_boxplot()
    p_values = plots.statistical_check()
    return p_values


def mult_models_one_dataset(pretrained_models, stimuli_dir, result_dir):
    df_model_comparison = pd.DataFrame()
    
    for model in pretrained_models:
        # prediction pipeline
        model_name = model[0]
        model_weights = model[1]
        input_dim = model[2]
        model_type = model[3]
        model_url = model[4]
        
        analysis = ObjectAnalysis()
        analysis.model_init(model_name=model_name, weights=model_weights, input_dim=input_dim,\
                            model_type=model_type, model_url=model_url)
        analysis.set_dirs(stimuli_dir=stimuli_dir)
        analysis.predict()
        df = analysis.wrangle(imgnet_labels=imgnet_labels, imgnet_sysnets=imgnet_sysnets,\
                              sysnets_car=sysnets_car, sysnets_noise=sysnets_noise)
        df_diff, df_log_diff = analysis.diff_wrangle()
        
        # output results pipeline
        plots = ObjectPlotting()
        plots.plot_init(df=df, plot_dir=result_dir, model_name=model_name, df_diff=df_diff,\
                        df_log_diff=df_log_diff, stim_set=stimuli_dir)
        plots.aggr_barplot()
        plots.diff_plot()
        plots.diff_boxplot()
        df_p_values = plots.statistical_check()
        
        # fetch all model results
        df_model_comparison = pd.concat([df_model_comparison, df_p_values], axis=0)
        df_model_comparison.to_csv(os.path.join(result_dir, 'df_model_comparison.csv'))
        
    return df_model_comparison
    

def mult_models_mult_datasets(pretrained_models, result_dir, batched_stim_parent,\
                         noise_bot_sce, noise_top_sce, noise_step_sce,\
                         noise_bot_obj, noise_top_obj, noise_step_obj):
    
    list_of_result_dfs = []
    
    for model in pretrained_models:
        
        model_name = model[0]
        model_weights = model[1]
        input_dim = model[2]
        model_type = model[3]
        model_url = model[4]
        df_batched_results = pd.DataFrame()
        
        steps_scene = np.arange(noise_bot_sce, noise_top_sce, noise_step_sce).tolist()
        steps_object = np.arange(noise_bot_obj, noise_top_obj, noise_step_obj).tolist()
        
        noise_steps_scene = [round(x,2) for x in steps_scene] 
        noise_steps_object = [round(x,2) for x in steps_object]
        noise_steps_object = [0] # hard coded
        
        analysis = ObjectAnalysis()
        analysis.model_init(model_name=model_name, weights=model_weights, input_dim=input_dim,\
                            model_type=model_type, model_url=model_url)
        
        for noise_step_object in noise_steps_object:
            for noise_step_scene in noise_steps_scene:
                
                noise_map = f'scene_{noise_step_scene}_object_{noise_step_object}' 
                batched_stim_dir = os.path.join(batched_stim_parent, noise_map)
        
                # predict
                analysis.set_dirs(stimuli_dir=batched_stim_dir)
                analysis.predict()
                df = analysis.wrangle(imgnet_labels=imgnet_labels, imgnet_sysnets=imgnet_sysnets,\
                                      sysnets_car=sysnets_car, sysnets_noise=sysnets_noise)
                df_diff, df_log_diff = analysis.diff_wrangle()
                
                # plot
                plots = ObjectPlotting()
                plots.plot_init(df=df, plot_dir=result_dir, model_name=model_name, df_diff=df_diff,\
                                df_log_diff=df_log_diff, stim_set=batched_stim_dir)
                plots.aggr_barplot()
                plots.diff_boxplot()
                df_p_values = plots.statistical_check()
                
                df_batched_results = pd.concat([df_batched_results, df_p_values], axis=0)
                df_batched_results.to_csv(os.path.join(result_dir, f'df_batched_{model_name}.csv'))
        
        list_of_result_dfs.append(df_batched_results)
        
    return list_of_result_dfs
        
#%%
def plotting_scores(list_of_dfs, result_dir,\
                    noise_bot_sce, noise_top_sce, noise_step_sce,\
                    noise_bot_obj, noise_top_obj, noise_step_obj):
    
    steps_scene = np.arange(noise_bot_sce, noise_top_sce, noise_step_sce).tolist()
    steps_object = np.arange(noise_bot_obj, noise_top_obj, noise_step_obj).tolist()
    
    n_steps = len(steps_object)
    
    for df, model_name in zip(list_of_dfs, ['s_min_rt', 's_min_z', 'rt_min_z']):
        
        for metric in ['accuracy', 'difference_score', 'p_value', 'log_diff', 'log_p']:
            for stim_size in ['all', 'small', 'large']:
                
                # model_name = df['model_name'].iloc[0]
                temp_df = df[df['stim_size'] == stim_size]
                temp_df = temp_df[metric].tolist()
                
                noise_steps_scene = np.round(steps_scene,2)
                noise_steps_object = np.round(steps_object,2)
                reversed_noise_steps_bject = np.round(list(reversed(noise_steps_object)),3)
                
                scores = np.array(temp_df)
                scores = np.reshape(scores, (-1, n_steps))
                scores = pd.DataFrame(scores)
                scores = scores.reindex(index=scores.index[::-1]) # invert y axis for object0 and scene0 be at the bottom left
                
                scene_means = pd.DataFrame(scores)
                scene_means = scene_means.mean(axis=0)
                scene_means_labels = ['%.4f' % elem for elem in scene_means]
                
                object_means = pd.DataFrame(scores)
                object_means = object_means.mean(axis=1)
                object_means_labels = ['%.4f' % elem for elem in object_means]
                object_means_labels = list(reversed(object_means_labels))
            
                # vcenter = 0.00000001
                # vmin, vmax = min(temp_df), max(temp_df)
                
                # if vmin > vcenter:
                #     vmin, vcenter = vcenter, vmin
            
            
                if metric == 'accuracy':
                    cmap = sn.color_palette("flare_r", as_cmap=True)
                    vmin = 0
                    vcenter = 0.5
                    vmax = 1
                    normalize = mcolors.TwoSlopeNorm(vcenter=np.float(vcenter), vmin=np.float(vmin), vmax=np.float(vmax))
                        
                if metric == 'difference_score':
                    cmap = sn.diverging_palette(31, 321, as_cmap=True)
                    vmin = -0.1
                    vcenter = 0 
                    vmax = 0.1
                    normalize = mcolors.TwoSlopeNorm(vcenter=np.float(vcenter), vmin=np.float(vmin), vmax=np.float(vmax))
                
                if metric == 'p_value':
                    cmap = sn.color_palette("light:#5A9", as_cmap=True)
                    vmin = 0
                    vcenter = 0.5
                    vmax = 1
                    normalize = mcolors.TwoSlopeNorm(vcenter=np.float(vcenter), vmin=np.float(vmin), vmax=np.float(vmax))
                    
                if metric == 'log_diff':
                    cmap = sn.diverging_palette(7, 81, as_cmap=True)
                    vmin = -3
                    vcenter = 0
                    vmax = 3
                    normalize = mcolors.TwoSlopeNorm(vcenter=np.float(vcenter), vmin=np.float(vmin), vmax=np.float(vmax))
     
                if metric == 'log_p':
                    cmap = sn.color_palette("Purples", as_cmap=True)
                    vmin = 0
                    vcenter = 1.5
                    vmax = 3
                    normalize = mcolors.TwoSlopeNorm(vcenter=np.float(vcenter), vmin=np.float(vmin), vmax=np.float(vmax))
                
                
                fig = plt.figure(figsize=(20,24))
                ax1 = plt.subplot2grid((14,10), (0,0), colspan=10, rowspan=10) # heatmap
                ax2 = plt.subplot2grid((14,10), (10,0), colspan=8, rowspan=2) # regression bottom: scene noise
                ax3 = plt.subplot2grid((14,10), (12,0), colspan=8, rowspan=2) # regression left: object noise
            
                # ax1 heatmap #
                # cmap = sn.diverging_palette(cmap_min, cmap_max, as_cmap=True)
                sn.heatmap(scores, ax=ax1, annot=True, cmap=cmap,  fmt='.5f',\
                          cbar=True, annot_kws={'color':'black'}, square=True,\
                          cbar_kws={'ticks':ticker.LinearLocator(numticks=10)},\
                          norm=normalize)
            
            
                ax1.set_xticklabels(noise_steps_scene)
                ax1.set_yticklabels(reversed_noise_steps_bject, rotation='horizontal')
                ax1.tick_params(axis='x', which='major', pad=17)
            
            
                # ax2 scene regression
                b_scene, m_scene = polyfit(noise_steps_scene, scene_means, 1)
                ax2.plot(noise_steps_scene, b_scene + m_scene * noise_steps_scene, linestyle='dashed', linewidth=1, color='black')
            
                ax2.scatter(x=noise_steps_scene,y=scene_means, color='black') #, width=0.2)
                ax2.axis(xmin=-0.065, xmax=0.475)
                ax2.axhline(0, linewidth=0.5, linestyle='dashed', color='k')
                ax2.set_xticks(noise_steps_scene)
                ax2.set_xticklabels([])
                ax2.xaxis.tick_top() 
                for i in range (0,n_steps):
                    xy=(noise_steps_scene[i] + 0.005,scene_means[i])
                    ax2.annotate(scene_means_labels[i],xy)
            
                
                ax2.bar(noise_steps_scene, scene_means, color='k', edgecolor='k', width=0.0001)
                ax2.set_ylim(vmin, vmax)
                
                # ax3 object regression
                reversed_object_means = np.round(list(reversed(object_means)),5)
                
                
                b_object, m_object = polyfit(reversed_noise_steps_bject, reversed_object_means, 1)
                ax3.plot(reversed_noise_steps_bject, b_object + m_object * reversed_noise_steps_bject, linestyle='dashed', linewidth=1, color='black')
            
                ax3.scatter(x=reversed_noise_steps_bject,y=reversed_object_means, color='black') #, width=0.2)
                ax3.axis(xmin=-0.065, xmax=0.475)
                ax3.axhline(0, linewidth=0.5, linestyle='dashed', color='k')
                ax3.set_xticks(noise_steps_object)
                ax3.set_xticklabels([])
                ax3.xaxis.tick_top() 
                for i in range (0,n_steps):
                    xy=(noise_steps_object[i] + 0.005,object_means[i])
                    ax3.annotate(object_means_labels[i],xy)
            
                ax3.set_ylim(vmin, vmax)
                ax3.bar(reversed_noise_steps_bject, reversed_object_means, color='k', edgecolor='k', width=0.0001)
            
                # formatting #
                ax1.set_ylabel("Object", labelpad=10, fontsize=16)
                ax1.set_xlabel("Scene", labelpad=10, fontsize=16)
            
                ax1.axhline(y = 0, color = 'k', linewidth = 2)
                ax1.axhline(y = n_steps, color = 'k', linewidth = 2)
                ax1.axvline(x = 0, color = 'k', linewidth = 2)
                ax1.axvline(x = n_steps, color = 'k', linewidth = 2)
            
                ax1.set_title(f'\n\n model = {model_name}, object size = {stim_size}, metric = {metric}\n\n', fontsize=24)
                                
                # significance check for regression slope value for object and scene noise
                shuffled_scene_slopes = []
                shuffled_object_slopes = []
                
                for i in range(10000):
                    shuffled_data = temp_df.copy()
                    random.shuffle(shuffled_data)
                    sh_scores = np.array(shuffled_data)
                    sh_scores = np.reshape(sh_scores, (-1, n_steps))
                    sh_scores = pd.DataFrame(sh_scores)
                    sh_scores = sh_scores.reindex(index=sh_scores.index[::-1]) # invert y axis for object0 and scene0 be at the bottom left
                    
                    sh_scene_means = pd.DataFrame(sh_scores)
                    sh_scene_means = sh_scene_means.mean(axis=0)
                    
                    sh_object_means = pd.DataFrame(sh_scores)
                    sh_object_means = sh_object_means.mean(axis=1)
                    sh_reversed_object_means = np.round(list(reversed(sh_object_means)),5)
                    
                    sh_b_scene, sh_m_scene = polyfit(noise_steps_scene, sh_scene_means, 1)
                    sh_b_object, sh_m_object = polyfit(reversed_noise_steps_bject, sh_reversed_object_means, 1)
                    
                    shuffled_scene_slopes.append(sh_m_scene)
                    shuffled_object_slopes.append(sh_m_object)
            
                z_score_scene = (m_scene - np.mean(shuffled_scene_slopes) / np.std(shuffled_scene_slopes))
                p_value_scene = scipy.stats.norm.sf(abs(z_score_scene))
                
                z_score_object = (m_object - np.mean(shuffled_object_slopes) / np.std(shuffled_object_slopes))
                p_value_object = scipy.stats.norm.sf(abs(z_score_object))
                
                ax2.set_xlabel(f'{metric} = (intercept = {format(b_scene,".4f")}) + SCENE_noise_percentage * (slope = {format(m_scene,".4f")})\n\n scene slope P value = {format(p_value_scene, ".6f")}', labelpad=15, fontsize=24)
                ax3.set_xlabel(f'{metric} = (intercept = {format(b_object,".4f")}) + OBJECT_noise_percentage * (slope = {format(m_object,".4f")})\n\n object slope P value = {format(p_value_object,".6f")}', labelpad=15, fontsize=24)
                
                
                output_dir = os.path.join(result_dir, model_name)
                map_isExist = os.path.exists(output_dir)
                if not map_isExist:
                    os.makedirs(output_dir)
                
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, f'{model_name}_{metric}_{stim_size}.png'), dpi=200)
                fig.show()

#%%



def plot_model_comparison(df, result_dir):
    
    df_model_comparison = df
    
    # combine logs and no transform for each record
    df_no_transform = df_model_comparison[df_model_comparison['transform'] == 'no_transform']
    df_natural_log = df_model_comparison[df_model_comparison['transform'] == 'natural_log']
    df_natural_log = df_natural_log[['accuracy', 'difference_score', 'p_value']]
    df_natural_log = df_natural_log.rename(columns={'accuracy':'log_acc', 'difference_score':'log_diff', 'p_value':'log_p'})
    df_natural_log.reset_index(inplace=True, drop=True)
    df_no_transform.reset_index(inplace=True, drop=True)
    df_model_comp = pd.concat([df_no_transform, df_natural_log], axis=1)
    
    
    for stim_size in ['all', 'small', 'large']:
        df_comp_plot = df_model_comp[df_model_comp['stim_size'] == stim_size]
        
        accuracies = tuple(df_comp_plot['accuracy'])
        differences = tuple(df_comp_plot['difference_score'])
        p_values = tuple(df_comp_plot['p_value'])
        
        log_diffs = df_comp_plot['log_diff']
        log_diffs = log_diffs.iloc[:,0]
        log_ps = df_comp_plot['log_p']
        log_ps = log_ps.iloc[:,0]
        
        comp_model_names = list(df_comp_plot['model_name'])
        
        x = np.arange(4)
        width = 0.3
        colors = ['magenta', 'pink', 'purple', 'coral']
        
        fig, axs = plt.subplots(5, figsize=[6 ,24])
        
        for i in range(5):
            axs[i].grid(True)
            axs[i].set_xticks(np.arange(0, 4, 1))
            # axs[i].set_xticklabels(' ')
        
        axs[0].bar(x, accuracies, width, label='accuracy', color=colors)
        axs[0].set_ylabel('Accuracy', fontsize=16, rotation=90)
        axs[0].set_ylim([0, 1])
        
        axs[1].bar(x, differences, width, label='difference_score', color=colors)
        axs[1].set_ylabel('Mean difference', fontsize=16, rotation=90)
        axs[1].set_ylim([-0.1, 0.1])
        
        axs[2].bar(x, p_values, width, label='p_value', color=colors)
        axs[2].set_ylabel('P value', fontsize=16, rotation=90)
        axs[2].set_ylim([0, 0.2])
        
        axs[3].bar(x, log_diffs, width, label='log_diff', color=colors)
        axs[3].set_ylabel('Log Mean Diff', fontsize=16, rotation=90)
        axs[3].set_ylim([-1, 1])
        
        axs[4].bar(x, log_ps, width, label='log_p', color=colors)
        axs[4].set_ylabel('P after log', fontsize=16, rotation=90)
        axs[4].set_ylim([0, 0.2])
        
        axs[4].set_xticklabels(comp_model_names, rotation=90, fontsize=16)
        # ax.bar(x + 3*width, p_values, width, label='1 - p_value')
        # ax.bar(x + 4*width, log_diffs, width, label='log_difference_score')
        # ax.bar(x + 5*width, log_ps, width, label='1 - p_value after log transform')
        # fig.tight_layout()
        axs[0].set_title(stim_size, fontsize=36)
        plt.savefig(os.path.join(result_dir, f'{stim_size}_barplot.png'), bbox_inches='tight')
        plt.show()
     

