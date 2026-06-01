# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn

from segment_anything.modeling import Sam

from typing import Optional, Tuple

from .utils.transforms import ResizeLongestSide
from .utils.amg import batch_iterator, build_all_layer_point_grids

from pycocotools import mask as mask_utils
import random 
from segment_anything.utils.transforms import Normalize, Unnormalize
from info_nce import InfoNCE
# from segment_anything.runners.diffpure_ddpm import Diffusion
from segment_anything.runners.diffpure_guided import GuidedDiffusion
import torch.nn.functional as F
import time
# from .utils.criterion import sigmoid_focal_loss, dice_loss


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def clip_mse(target, pred, left_thre=-1000, right_thre=1000):
    pred = torch.clamp(pred, min=left_thre, max=right_thre)
    loss = (pred-target)**2
    return loss.mean()


    
class AttackEncoderUAPCL:
    def __init__(
        self,
        sam_model: Sam,
        attack_type = 'clean_gt',
        attack_init=True, 
        epsilon=10.0,
        step_size=2.0,
        num_steps=10,
        uap = None, 
        optimizer = None,
        clean_features_path = None, 
        aug_type = None,
        neg_num = None,
        temperature=None
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        
        # Attack parameters
        dataset_mean = [123.675, 116.28, 103.53]
        dataset_std = [58.395, 57.12, 57.375]

        self.mu = torch.tensor(dataset_mean).view(3,1,1).cuda()
        self.std = torch.tensor(dataset_std).view(3,1,1).cuda()

        self.upper_limit = ((255 - self.mu)/ self.std)
        self.lower_limit = ((0 - self.mu)/ self.std)

        # self.epsilon = epsilon / self.std
        # self.alpha = step_size / self.std

        self.epsilon = epsilon / 255.
        self.alpha = step_size 

        self.attack_init = attack_init 
        self.attack_type = attack_type
        self.num_steps = num_steps

        self.unnorm = Unnormalize(mean=dataset_mean, std=dataset_std)
        self.norm = Normalize(mean=dataset_mean, std=dataset_std)

        self.uap = uap
        self.optimizer = optimizer
        self.clean_features_path=clean_features_path
        self.aug_type = aug_type
        self.neg_num = neg_num
        self.temperature=temperature
        self.image_positive=None
        self.image_positive_source=None
        self.iou_sum=0
        self.times_index=0
    def generate_delta(
        self,
        image: np.ndarray,
        attack_point_per_size: int = 10, 
        attack_point_per_batch: int = 10,
        max_conf_point: bool = False,
        min_conf_point: bool = False,
        random_point: bool = False,
        multimask_output: bool = True,
        image_format: str = "RGB",
        target_features = None,
        low_res_masks_target = None,
        mask_tokens_out_target = None,
        upscaled_embedding_target = None,
        src_target = None,
    ):

        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]  #(1500, 2250, 3)
      
        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)    
        input_image_torch = torch.as_tensor(input_image, device=self.model.device) 
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]  
        self.set_torch_image(input_image_torch, image.shape[:2])

     
        # point selection 
        points_for_image = self.generate_point_mask_pairs(attack_point_per_size, max_conf_point, min_conf_point, random_point, multimask_output)

        target_features = target_features
        low_res_masks_target = low_res_masks_target
        mask_tokens_out_target = mask_tokens_out_target
        upscaled_embedding_target = upscaled_embedding_target
        src_target = src_target
        
        self.train_uap(points_for_image, input_image_torch.shape, attack_point_per_batch, self.attack_type, multimask_output, target_features=target_features,low_res_masks_target = low_res_masks_target, mask_tokens_out_target = mask_tokens_out_target,  upscaled_embedding_target = upscaled_embedding_target, src_target = src_target)


    def attack_image(
        self,
        image: np.ndarray,
        attack_point_per_size: int = 10, 
        attack_point_per_batch: int = 10,
        max_conf_point: bool = False,
        min_conf_point: bool = False,
        random_point: bool = False,
        multimask_output: bool = True,
        image_format: str = "RGB",
    ):
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.model.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]  
        self.set_torch_image(input_image_torch, image.shape[:2])

        # point selection 
        points_for_image = self.generate_point_mask_pairs(attack_point_per_size, max_conf_point, min_conf_point, random_point, multimask_output)
        
        x_adv, masks_adv, masks_clean, clean_features, low_res_masks_gt = self.return_adv_sample(points_for_image, input_image_torch.shape, attack_point_per_batch, self.attack_type, multimask_output)

        return x_adv, masks_adv, masks_clean, points_for_image, clean_features, low_res_masks_gt


    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size  #(1500,2250)
        self.input_size = tuple(transformed_image.shape[-2:]) #(683, 1024)
        input_image = self.model.preprocess(transformed_image) #(1,3,1024,1024)
        self.model_input = input_image
        # self.features = self.model.image_encoder(input_image)
        # self.is_image_set = True


    def generate_point_mask_pairs(self, attack_point_per_size, max_conf_point, min_conf_point, random_point, multimask_output):
        
        seed = 10
        np.random.seed(seed)
        if max_conf_point or min_conf_point:
          point_number_per_size = 2

          ### generate the points 
          point_grids = build_all_layer_point_grids(
                  point_number_per_size,
                  n_layers=0,
                  scale_per_layer=1,
              )
          points_scale = np.array(self.original_size)[None, ::-1] # (1570, 1054)
          points_candidates = point_grids * points_scale  #(1024, 2)
          
          
          ### predict mask batch by batch and find the point with highest confidence 
          point_batch_size = 64
          highest_score = -1.0
          lowest_score = 100.0

          for (points,) in batch_iterator(point_batch_size, points_candidates):
            # transform point in this batch 
            transformed_points = self.transform.apply_coords(points, (self.original_size[-1], self.original_size[-2])) # im_size (1054, 1570)
            in_points = torch.as_tensor(transformed_points, device=self.model.device)
            in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
            final_points = (in_points[:, None, :], in_labels[:, None])

            # generate the ground truth mask for this batch 
            with torch.no_grad():   
              # Embed prompts
              sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                  points=final_points,
                  boxes=None,
                  masks=None,
              )
              # Predict gt masks
              low_res_masks_gt, iou_predictions_gt = self.model.mask_decoder(
                  image_embeddings=self.features,
                  image_pe=self.model.prompt_encoder.get_dense_pe(),
                  sparse_prompt_embeddings=sparse_embeddings,
                  dense_prompt_embeddings=dense_embeddings,
                  multimask_output=multimask_output,
              )
              if max_conf_point:
                if iou_predictions_gt.max() > highest_score:
                  highest_score = iou_predictions_gt.max()
                  points_for_image = points[iou_predictions_gt.argmax()].reshape(1,-1)
              if min_conf_point:
                if iou_predictions_gt.min() < lowest_score:
                  lowest_score = iou_predictions_gt.min()
                  points_for_image = points[iou_predictions_gt.argmin()].reshape(1,-1)

        elif random_point:
          
          # ### generate the points 
          points_scale = np.array(self.original_size)[None, ::-1]
          # point_grids = np.random.random([attack_point_per_size*2]).reshape([-1,2])
          
          #use grid generate point
          num_points_per_dim = attack_point_per_size + 2
          point_interval = 1.0 / (num_points_per_dim - 1)
          point_grids = []
          for i in range(num_points_per_dim):
            for j in range(num_points_per_dim):
                x = i * point_interval
                y = j * point_interval
                if 0 < x < 1 and 0 < y < 1:
                    point_grids.append([x, y])

          points_for_image = point_grids * points_scale

          # print(point_grids)
        else:        
          point_grids = build_all_layer_point_grids(
                  attack_point_per_size,
                  n_layers=0,
                  scale_per_layer=1,
              )
          
          point_grids = np.random.random([attack_point_per_size**2*2]).reshape([-1,2])
          # point_grids = np.random.random([64*2]).reshape([-1,2])
          points_scale = np.array(self.original_size)[None, ::-1] # (2250, 1500)
          #import pdb ;pdb.set_trace()
          #point_grids=[0.8,0.2]# For imageshow
          points_for_image = point_grids * points_scale  #(1125, 750)# [0] #point_grids=[0.77132,0.02075]
          #import pdb;pdb.set_trace()
        return points_for_image
    def return_adv_sample(
        self,
        points_for_image,
        transformed_shape,
        attack_point_per_batch,
        attack_type = 'clean_gt',
        multimask_output = True
        ):
          
      ori_h,ori_w = transformed_shape[-2], transformed_shape[-1]
      input_h, input_w = self.model_input.shape[-2:]
      x = self.model_input.detach()
      self.model.eval()
      
      # generate adv samples by batch 
      for (points,) in batch_iterator(attack_point_per_batch, points_for_image):
        # transform point in this batch 
        transformed_points = self.transform.apply_coords(points, (self.original_size[-1], self.original_size[-2])) # im_size (1054, 1570)
        in_points = torch.as_tensor(transformed_points, device=self.model.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        final_points = (in_points[:, None, :], in_labels[:, None])
 
        # generate the ground truth mask for this batch 
        with torch.no_grad():   
          # Embed prompts
          sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
              points=final_points,
              boxes=None,
              masks=None,
          )
        
          # Predict gt masks
          clean_features = self.model.image_encoder(x)
          low_res_masks_gt, iou_predictions_gt = self.model.mask_decoder(
              image_embeddings=clean_features,
              image_pe=self.model.prompt_encoder.get_dense_pe(),
              sparse_prompt_embeddings=sparse_embeddings,
              dense_prompt_embeddings=dense_embeddings,
              multimask_output=multimask_output,    
          )   

          x_unorm = self.unnorm(x)
          x_adv = self.uap(x_unorm,'test')
          adv_features = self.model.image_encoder(self.norm(x_adv))
          low_res_masks_adv, iou_predictions_adv= self.model.mask_decoder(
              image_embeddings=adv_features,
              image_pe=self.model.prompt_encoder.get_dense_pe(),
              sparse_prompt_embeddings=sparse_embeddings,
              dense_prompt_embeddings=dense_embeddings,
              multimask_output=multimask_output,
          )

        masks_adv = self.model.postprocess_masks(low_res_masks_adv, self.input_size, self.original_size)
        masks_clean = self.model.postprocess_masks(low_res_masks_gt, self.input_size, self.original_size)     
      return x_adv[:,:,:ori_h,:ori_w].to(torch.uint8), masks_adv, masks_clean, clean_features, low_res_masks_gt
  


    def train_uap(
        self,
        points_for_image,
        transformed_shape,
        attack_point_per_batch,
        attack_type = 'clean_gt',
        multimask_output = True,
        preset_mask = None,
        target_features = None,
        low_res_masks_target = None,
        mask_tokens_out_target = None,
        upscaled_embedding_target = None,
        src_target = None,
        ):
      
      ori_h,ori_w = transformed_shape[-2], transformed_shape[-1]  #(683,1024)
      input_h, input_w = self.model_input.shape[-2:]   #(1024,1024)
      x = self.model_input.detach()   #(1,3,1024,1024)
      self.model.eval()

      # generate adv samples by batch 
      for (points,) in batch_iterator(attack_point_per_batch, points_for_image):
        # transform point in this batch 
        transformed_points = self.transform.apply_coords(points, (self.original_size[-1], self.original_size[-2])) # im_size (1054, 1570)
        in_points = torch.as_tensor(transformed_points, device=self.model.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        final_points = (in_points[:, None, :], in_labels[:, None])

        # generate the ground truth mask for this batch 
        with torch.no_grad():   
          # Embed prompts
          sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
              points=final_points,
              boxes=None,
              masks=None,
          )
          
          
          # loas clean features
          clean_features_list = torch.load(self.clean_features_path)
          negative_sample = clean_features_list          
          clean_ft_len = len(clean_features_list)
          negative_sample = torch.cat([torch.flatten(sample, start_dim=1) for sample in negative_sample], dim=0)
          
          embedding_dim = negative_sample.shape[1]
                         
        for att_iter_num in range(self.num_steps):

            with torch.enable_grad(): 
              x_unorm = self.unnorm(x)
              #import pdb;pdb.set_trace()
              x_adv = self.uap(x_unorm)
              # recalculate the image features
              adv_features = self.model.image_encoder(self.norm(x_adv))    

              
            if attack_type == 'enc_rm_inf':
                
                # calculate the delta features
                delta_features = self.model.image_encoder(torch.unsqueeze(self.uap.delta, 0))

                # attack encoder use nce loss
                query = delta_features.view(1, -1) # 1 x embedding_dim
                positive = adv_features.view(1, -1) # 1 x embedding_dim
                negative = negative_sample.view(clean_ft_len, embedding_dim) # clean_ft_len x embedding_dim
                negative = negative[::int(clean_ft_len/self.neg_num)]
                info_loss = InfoNCE(negative_mode='unpaired',temperature=self.temperature)
                start = time.time()
                loss_infoNCE = info_loss(query, positive, negative)
                print('attack time:', time.time() - start)
                loss = loss_infoNCE

                 
            #loos backward                                                                  
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.uap.delta.data = torch.clamp(self.uap.delta.data, -self.epsilon, self.epsilon)

            print(loss.item())



    def reset_image(self) -> None:
      """Resets the currently set image."""
      self.is_image_set = False
      self.features = None
      self.orig_h = None
      self.orig_w = None
      self.input_h = None
      self.input_w = None
      
