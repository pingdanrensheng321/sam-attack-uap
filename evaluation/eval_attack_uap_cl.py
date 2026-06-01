import torch
import torchvision
import torch.nn as nn
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import sys
sys.path.append("..")
# import pdb;pdb.set_trace()
#sys.path = [p for p in sys.path if "segment-anything" not in p]

#han change 20240209, because pip environment occur unknown changed
path_to_remove = '/home/gpuadmin/zs/Target_Attack_SAM'
if path_to_remove in sys.path:
    sys.path.remove(path_to_remove)
    sys.path.append("/home/gpuadmin/zs/sam-uap")
#han end 20240209
# import pdb;pdb.set_trace()
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, AttackEncoderUAPCL
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import json
import time
import argparse
import os
from contextlib import suppress
from datetime import datetime
import pytz
import shutil
from os.path import join
from segment_anything.utils.transforms import ResizeLongestSide
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
import scipy.stats as st

parser = argparse.ArgumentParser(description="Attack SAM")
parser.add_argument("--experiment", type=str, default='attack_mask')
parser.add_argument("--attack_type", type=str, choices=['dec_rm_image', 'enc_rm_inf', 'dec_rm_prompt','enc_rm_inf_2aug','understand_cosine_sim'])
parser.add_argument("--aug_type", type=str, default=None,choices=['flatten','uniform_gaussian_blur','spatial_crop','spatial_cutout','white','uniform','k_uniform', 'k_gaussian','crop','k_dynamic','k_crop_image', 'flatten_add_uniform'])
parser.add_argument("--point_number_size", type=int, default=1)
parser.add_argument('--eps', default=10.0, type=float)
parser.add_argument('--alpha', default=2.0, type=float)
parser.add_argument('--attack_steps', default=10, type=int)
parser.add_argument('--attack_init', default='zero', choices=['zero', 'random'])
parser.add_argument('--auto_seg', action='store_true', help='automatic segmentation')
parser.add_argument('--adv_mask_with_image', action='store_true', help='plot masks on the image')
parser.add_argument('--max_conf_point', action='store_true', help='Select the point prompt which has the highest mask confidence score.')
parser.add_argument('--min_conf_point', action='store_true', help='Select the point prompt which has the lowest mask confidence score.')
parser.add_argument('--random_point', action='store_true', help=' Randomly select the point prompt.')
parser.add_argument('--multimask_output_off', action='store_false', help='How many masks of a single point.')
parser.add_argument('--source_model', type=str, default='vit_b', choices=['vit_b', 'vit_l', 'vit_h','tiny_sam','repvit_sam','edge_sam'], help='source model')
parser.add_argument('--target_model', type=str, default='vit_b', choices=['vit_b', 'vit_l', 'vit_h','tiny_sam','repvit_sam','edge_sam'], help='source model')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--attack_lr', default=0.005, type=float)
parser.add_argument('--train_uap', action='store_true', help='train uap.')
parser.add_argument('--train_img_num', default=100, type=int)
parser.add_argument('--test_img_num', default=100, type=int)
parser.add_argument('--debug', default=1, type=int)
parser.add_argument('--show_img', default=0, type=int)
parser.add_argument('--sh_file_name', type=str)
parser.add_argument('--neg_num',default=1, type=int)
parser.add_argument('--temperature',default=0.1, type=float)
parser.add_argument('--use_another_data', default=0, type=int)
parser.add_argument('--defense', default=0, type=int)

args = parser.parse_args()
class UAP(nn.Module):
    def __init__(self,
                shape=(224, 224),
                num_channels=3):
        super(UAP, self).__init__()
        self.aug_type = args.aug_type
        self.delta = nn.Parameter(torch.zeros(size=(num_channels, *shape), requires_grad=True))
        self.times=0.50
    def augment_method(self,input_image,delta):
        if self.aug_type=='spatial_crop': #sample uniform
            crop_size_range=50#993
            rnd = torch.randint(crop_size_range, 51,size=(1,))[0]
            random_crop = transforms.RandomCrop(size=(rnd,rnd))
            #import pdb;pdb.set_trace()
            
            delta_augment = F.interpolate(random_crop(delta.unsqueeze(dim=0)),size=(1024,1024)).squeeze(0)
            augment=delta_augment.clone()
            print("spatial_crop:",crop_size_range)
        elif self.aug_type=='spatial_cutout': #sample uniform
            cutout_size = (2, 2)  # cutout的大小

            # 随机生成cutout的左上角坐标
            cutout_x = torch.randint(0, 1024 - cutout_size[0] + 1, size=())
            cutout_y = torch.randint(0, 1024 - cutout_size[1] + 1, size=())
            augment=delta.clone()
            # 执行cutout操作，将指定区域的像素值设置为0
            augment[:, cutout_y:(cutout_y + cutout_size[1]), cutout_x:(cutout_x + cutout_size[0])] =0
            
            # han_delta=delta.permute(1,2,0).clone.detach().cpu().numpy()
            # plt.figure()
            # plt.imshow(han_delta.detach())#.cpu()
            # plt.axis('off')
            # plt.savefig("{}.png".format('cut_delta'), bbox_inches='tight', pad_inches = 0.0)
            # plt.show()
            print("spatial_cutout: x,y:",cutout_x,cutout_y)
            #import pdb;pdb.set_trace()
        elif self.aug_type=='flatten': #sample uniform
            random_color1 = torch.randint(0, 256, (1, 1, 1), dtype=torch.uint8)
            random_color2 = torch.randint(0, 256, (1, 1, 1), dtype=torch.uint8)
            random_color3 = torch.randint(0, 256, (1, 1, 1), dtype=torch.uint8)

            # 使用这些随机颜色来创建三个通道的张量
            random_color_tensor = torch.cat([random_color1, random_color2, random_color3], dim=0)

            # 扩展张量的大小以匹配目标形状（3, 1024, 1024）
            random_color_tensor = random_color_tensor.expand(3, 1024, 1024)
            augment=random_color_tensor.detach().cuda()+delta

        if self.aug_type=='flatten_add_uniform': #sample uniform
            random_color1 = torch.randint(0, 256, (1, 1, 1), dtype=torch.uint8)
            random_color2 = torch.randint(0, 256, (1, 1, 1), dtype=torch.uint8)
            random_color3 = torch.randint(0, 256, (1, 1, 1), dtype=torch.uint8)

            # 使用这些随机颜色来创建三个通道的张量
            random_color_tensor = torch.cat([random_color1, random_color2, random_color3], dim=0)

            # 扩展张量的大小以匹配目标形状（3, 1024, 1024）
            random_color_tensor = random_color_tensor.expand(3, 1024, 1024).cuda()
            uniform_noise = torch.randint(0, 11, size=delta.size()).cuda()
            random_color_tensor = random_color_tensor + uniform_noise
            augment=random_color_tensor.detach()+delta
            print('augmentation:flatten_add_uniform ')
        if self.aug_type=='uniform_gaussian_blur': #sample uniform
            ## create by hds
            new_tensor = torch.randint(0, 256, size=delta.size())
            kernel_size = (15, 15)
            blurring=torchvision.transforms.GaussianBlur(kernel_size,sigma=100)
            blur_img=blurring(new_tensor)
            augment=blur_img.detach().cuda()+delta
            ##### 
            # channels=3
            # kernel_size=5
            # kernel = gkern(kernel_size, 3).astype(np.float32)
            # gaussian_kernel = np.stack([kernel, kernel, kernel])
            # gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
            # gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()

            # delta_aug = torch.randint(0, 256, size=delta.size()).to(torch.float32).cuda()

            # delta_aug_smooth = F.conv2d(delta_aug.unsqueeze(0), gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3).squeeze(0)
            # delta_aug = delta_aug.to(torch.int)
            # delta_aug_smooth = delta_aug_smooth.to(torch.int)
            # augment=delta_aug_smooth+delta

            
            #import pdb;pdb.set_trace()
            # plt.figure()
            # plt.imshow(blur_img.permute(1,2,0))#.cpu()
            # plt.axis('off')
            # plt.savefig("{}.png".format('blur_img_new'), bbox_inches='tight', pad_inches = 0.0)
            # plt.show()
            # plt.figure()
            # plt.imshow(new_tensor.permute(1,2,0))#.cpu()
            # plt.axis('off')
            # plt.savefig("{}.png".format('new_tensor_new'), bbox_inches='tight', pad_inches = 0.0)
            # plt.show()
            # import pdb;pdb.set_trace()
        elif self.aug_type=='white': #sample uniform
            white_ts = torch.full_like(delta, 255)
            #import pdb;pdb.set_trace()
            augment=white_ts.detach()+delta
        elif self.aug_type=='uniform': #sample uniform
            new_tensor = torch.randint(0, 11, size=delta.size()).cuda()
            #import pdb;pdb.set_trace()
            augment=new_tensor.detach()+delta
            print('uniform_value')
        elif self.aug_type=='color_uniform': #sample uniform
            new_tensor = torch.randint(0, 11, size=delta.size()).cuda()
            #import pdb;pdb.set_trace()
            augment=new_tensor.detach()+delta
            print('uniform_value')
        elif self.aug_type=='k_uniform': #sample uniform
            random_number = 1.3#(torch.rand(1) * 2).cuda()
            #augment=random_number.detach()*input_image + delta   
            augment=random_number*input_image + delta 
            print('Aug uniform random number',random_number)
        elif self.aug_type=='k_gaussian':  #sample gaussian
            
            mean = torch.tensor([1], dtype=torch.float32)
            std = torch.tensor([1], dtype=torch.float32)
            #torch.cuda.manual_seed(torch.initial_seed())
            random_number = torch.normal(mean, std)
            #import pdb;pdb.set_trace() 
            # torch.cuda.manual_seed(args.seed)
            while random_number < 0 or random_number > 2:
                random_number = torch.normal(mean, std)
            print('Aug gausian random number',random_number)
            augment=random_number.cuda() * input_image +delta    
            ############################################################
            # mean = torch.tensor([1.25], dtype=torch.float32)
            # std = torch.tensor([1], dtype=torch.float32)
            # #torch.cuda.manual_seed(torch.initial_seed())
            # random_number = torch.normal(mean, std)
            # #import pdb;pdb.set_trace() 
            # # torch.cuda.manual_seed(args.seed)
            # while random_number < 1 or random_number > 1.5:
            #     random_number = torch.normal(mean, std)
            # print('Aug gausian random number',random_number)
            # augment=random_number.cuda() * input_image +delta  
        elif self.aug_type=='crop': 
            
            rnd = torch.randint(993, 1024,size=(1,))[0]
            h_rem = 1024 - rnd
            w_rem = 1024 - rnd
            pad_top = torch.randint(0, h_rem,size=(1,))[0]
            pad_bottom = h_rem - pad_top
            pad_left = torch.randint(0, w_rem,size=(1,))[0]
            pad_right = w_rem - pad_left
            
            c = torch.rand(1)
            # random_crop = transforms.RandomCrop(size=(rnd,rnd))
            # delta_augment = (F.pad(random_crop(delta.unsqueeze(dim=0)),(pad_left,pad_right,pad_top,pad_bottom), mode='constant', value=0)).squeeze(0)#,mode='constant', value=0  # crop
            if c <= 0.7:
                delta_augment = (F.pad(F.interpolate(delta.unsqueeze(dim=0), size=(rnd,rnd)),(pad_left,pad_right,pad_top,pad_bottom), mode='constant', value=0)).squeeze(0)
            else:
                delta_augment = delta
            augment = delta_augment + input_image
            print('c value', c)
            print('Aug crop size',rnd)
            #import pdb;pdb.set_trace()
            
            # #transform = transforms.RandomResizedCrop(size=(1024,1024), scale=(0.2,0.5), antialias=True)
            # # Calculate the crop coordinates
            # # crop_top = (delta.shape[2] - target_height) // 2
            # # crop_left = (delta.shape[3] - target_width) // 2
            # scale=(0.2,0.5)*torch.random(1)
            # width=delta.shape[2]
            # height=delta.shape[3]
            
            # crop_size = (1000, 1000)  # Specify the desired crop size
            # random_crop = transforms.RandomCrop(crop_size)
            # augment=F.pad(random_crop,(width,height))
            # # augment = transform(delta) 
            augment.to('cuda')  
        elif self.aug_type=='k_dynamic': 
            self.times=self.times+0.01
            augment = (self.times*input_image).detach() + delta
            print(self.times)
        elif self.aug_type=='k_crop_image': #image crop
            rnd = torch.randint(993, 994,size=(1,))[0]#993
            random_crop = transforms.RandomCrop(size=(rnd,rnd))
            #import pdb;pdb.set_trace()
            crop_image = F.interpolate(random_crop(input_image),size=(1024,1024)).squeeze(0)
            augment = crop_image.detach() + delta
        return augment
    def forward(self, x,type=None):
        # if( type=='test'):
        #     import pdb;pdb.set_trace()
        self.aug_type = None if type == 'test' else self.aug_type
        delta = self.delta * 255
       
        # delta= (torch.rand(1)*20-10).cuda()  #uniform      
        # Add uap to input
        # import pdb;pdb.set_trace()
        if (self.aug_type==None):  
            adv_x = torch.clamp(x + delta, 0, 255)
        else:
            adv_x=self.augment_method(x,delta)
        return adv_x
    
def gaussian_kernel(kernel_size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma**2)) * np.exp(-((x - (kernel_size-1)/2)**2 + (y - (kernel_size-1)/2)**2) / (2 * sigma**2)),
        (kernel_size, kernel_size)
    )
    return torch.tensor(kernel, dtype=torch.float32)

def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def show_attack_anns(anns):
    if anns.size(0) == 0:
        return
    masks = anns[0] > 0.0
    masks = masks.detach().cpu().numpy()

    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for index in range(masks.shape[0]):
        m = masks[index]
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def show_separate_masks(image, anns, points, out_dir, file_name):
    if anns.size(0) == 0:
        return
    point_num = anns.size(0)
    # plt.figure(figsize=(40,40))
    plt.figure()
    for point_index in range(point_num):
        masks = anns[point_index] > 0.0
        masks = masks.detach().cpu().numpy()
        plt.subplot(point_num, masks.shape[0]+1, point_index*(masks.shape[0]+1)+1)
        plt.imshow(image)
        plt.scatter(points[point_index,0], points[point_index,1], color='green', marker='*', s=150, edgecolor='white', linewidth=1.0)
        plt.axis('off')
        zero_image = np.zeros_like(image)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for index in range(masks.shape[0]):
            plt.subplot(point_num, masks.shape[0]+1, point_index*(masks.shape[0]+1) + index+2)
            plt.imshow(zero_image)
            # plt.scatter(points[point_index,0], points[point_index,1], color='green', marker='*', s=150, edgecolor='white', linewidth=1.0)
            color = np.array([1.0, 1.0, 1.0])
            mask_image = masks[index]
            result_image =mask_image[:,:,None]*color.reshape(1, 1, -1) 
            plt.imshow(result_image)
            plt.axis('off')
    plt.savefig("{}.png".format(out_dir + '/adv_mask_' + file_name), bbox_inches='tight', pad_inches = 0.0)

def show_separate_figure(image, anns, points, out_dir, file_name):
    if anns.size(0) == 0:
        return
    plt.figure()
    plt.imshow(image)
    plt.scatter(points[0,0], points[0,1], color='green', marker='*', s=1000, edgecolor='white', linewidth=1.75)
    plt.axis('off')
    ax = plt.gca()
    ax.set_autoscale_on(False)
    plt.savefig("{}.png".format(out_dir + '/clean_image'), bbox_inches='tight', pad_inches = 0.0)

    point_num = anns.size(0)
    # plt.figure(figsize=(40,40))
    plt.figure()
    for point_index in range(point_num):
        masks = anns[point_index] > 0.0
        masks = masks.detach().cpu().numpy()
        polygons = []
        color = []
        zero_image = np.zeros_like(image)
        for index in range(masks.shape[0]):
            plt.imshow(zero_image)
            color = np.array([1.0, 1.0, 1.0])
            mask_image = masks[index]
            result_image =mask_image[:,:,None]*color.reshape(1, 1, -1) 
            plt.imshow(result_image)
            plt.axis('off')
        plt.savefig("{}.png".format(out_dir + '/adv_mask_' + file_name), bbox_inches='tight', pad_inches = 0.0)
        

def calculate_iou(gt, dt):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    intersection = ((gt * dt) > 0).sum()
    union = ((gt + dt) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou

def calculate_iou_tensor(gt, dt):
    """
    Compute boundary iou between two binary masks.
    :param gt (PyTorch tensor, uint8): binary mask
    :param dt (PyTorch tensor, uint8): binary mask
    :return: boundary iou (float)
    """
    intersection = torch.sum((gt * dt) > 0).item()  
    union = torch.sum((gt + dt) > 0).item()  
    if union == 0:
        boundary_iou = 0
    else:
        boundary_iou = intersection / union  
    return boundary_iou

def get_iou_matrix(masks_gt, masks_pred):
    area = []
    # shape = (len(masks_gt), len(masks_pred))
    shape = (masks_gt.shape[0], masks_pred.shape[0])
    iou_matrix = np.zeros(shape=shape)
    for gt_idx in range(shape[0]):
        for pred_idx in range(shape[1]):
            # iou = calculate_iou(masks_gt[0, gt_idx], masks_pred[0, pred_idx])
            iou = calculate_iou(masks_gt[gt_idx,0], masks_pred[pred_idx,0])
            iou_matrix[gt_idx, pred_idx] = iou
        # area.append(masks_gt[gt_idx]['area'])
    
    return iou_matrix, area

def get_iou_matrix_tensor(masks_gt, masks_pred):
    
    shape = (masks_gt.shape[0], masks_pred.shape[0])
    iou_matrix = torch.zeros(shape)
    area = []

    for gt_idx in range(shape[0]):
        for pred_idx in range(shape[1]):
            

            iou = calculate_iou_tensor(masks_gt[gt_idx, 0], masks_pred[pred_idx, 0])
            iou_matrix[gt_idx, pred_idx] = iou

    return iou_matrix, area

def get_max_index(matrix, axis):
    return np.argmax(matrix, axis=axis)

def get_max_value(matrix, axis):
    return np.amax(matrix, axis=axis)



def main(args): 
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    ####################################Save code####################################
   
    sub_set = args.attack_type+'_' +('' if args.aug_type is None else args.aug_type)
    if args.debug:
        mode_output_dir = os.path.join('./results/debug', args.experiment, sub_set)
    else:
        mode_output_dir = os.path.join('./results', args.experiment, sub_set)

    now_utc = datetime.now(pytz.utc)
    # Convert UTC time to KST
    kst = pytz.timezone('Asia/Seoul')
    current_time = now_utc.astimezone(kst).strftime("%m%d%H%M%S")
    current_path = mode_output_dir+current_time+'/'
    if not os.path.exists(current_path):
        os.makedirs(current_path)
        os.makedirs(join(current_path,'code'))
    #save current code   
    source_file = os.path.abspath(__file__)
    destination_file = os.path.join(current_path, 'code', os.path.basename(__file__))
    shutil.copy2(source_file, destination_file)
    #save loss code
    source_file = os.path.join('./segment_anything/','attack_mask_uap_batch_cl.py')
    destination_file = os.path.join(current_path, 'code', 'attack_mask_uap_batch_cl.py')
    shutil.copy2(source_file, destination_file)
    #save sh file
    sh_file_name = args.sh_file_name
    source_file = os.path.join('./experiments/uap', sh_file_name)
    destination_file = os.path.join(current_path, 'code', sh_file_name)
    shutil.copy2(source_file, destination_file)

    ###########################################Initialize UAP###############################################
    uap = UAP(shape=(1024, 1024), num_channels=3).cuda()
    optimizer = torch.optim.Adam(uap.parameters(), lr=args.attack_lr)

    model_dict = {'vit_b': './checkpoints/sam_vit_b_01ec64.pth', 'vit_l': 'sam_vit_l_0b3195.pth', 'vit_h': 'sam_vit_h_4b8939.pth', 'tiny_sam': '/home/gpuadmin/zs/Target_Attack_SAM/checkpoints/mobile_sam/mobile_sam.pt','edge_sam': '/home/gpuadmin/zs/Target_Attack_SAM/checkpoints/edge_sam/edge_sam.pth','repvit_sam': '/home/gpuadmin/zs/Target_Attack_SAM/checkpoints/repvit/repvit_sam.pt'} 
    sam_checkpoint = model_dict[args.source_model]
    device = "cuda"
    sam = sam_model_registry[args.source_model](checkpoint=sam_checkpoint) # model register
    sam.to(device=device)

    ##########################################Train UAP#################################
    clean_features_path='/home/gpuadmin/zs/sam-uap/results/clean_features_new/clean_features_list.pth'#'/home/gpuadmin/zs/sam-uap/results/clean_features_new/repvit.pth'#'/home/gpuadmin/zs/sam-uap/results/clean_features_new/clean_features_list.pth'
    data_dir = './dataset/sam_data100/'
    file_list = np.loadtxt('./evaluation/eval_images/sam_data100.txt', dtype=str)
    
    
    ##handongshen 20250106
    if args.defense==1:
        AttackEncoderUAPCL = load_defense_module()
    else:
        from segment_anything import AttackEncoderUAPCL

    if args.attack_type == 'dec_rm_prompt':
        data_dir_test =  data_dir
        file_list_test = file_list
        train_point_batch=1
        test_random_point=False
    else:
        data_dir_test = './dataset/sam_data100_test/'
        file_list_test = np.loadtxt('./evaluation/eval_images/sam_data100_test.txt',dtype=str)
        train_point_batch=1000
        test_random_point=not(args.attack_type == 'enc_rm_inf' or args.attack_type == 'enc_rm_inf_2aug')
        #import pdb;pdb.set_trace()
    if(args.attack_type == 'enc_rm_inf_2aug' or args.attack_type == 'understand_cosine_sim'):
        achor_aug_list = torch.load('/home/gpuadmin/zs/sam-uap/results/anchor_image_norm/positive_image_norm.pth')
        directory_path='/home/gpuadmin/zs/sam-uap/dataset/sam_data100_positive'
        file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    file_list = np.loadtxt('./evaluation/eval_images/sam_data100.txt', dtype=str)
    
    
    if args.use_another_data== 1:
        data_dir_test =  '/home/gpuadmin/zs/sam-uap/dataset/SA-1B'
        folder_path = '/home/gpuadmin/zs/sam-uap/dataset/SA-1B'
        file_list_test = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        file_list_test = np.array(file_list_test)
    
    mask_generator_uap = AttackEncoderUAPCL(sam, attack_type=args.attack_type, 
                                        attack_init = args.attack_init, 
                                        epsilon=args.eps,
                                        step_size=args.alpha,
                                        num_steps=args.attack_steps,
                                        uap=uap, 
                                        optimizer=optimizer,
                                        clean_features_path=clean_features_path, 
                                        aug_type= args.aug_type,
                                        neg_num = args.neg_num,
                                        temperature=args.temperature
                                        )
    
    
    
    if args.train_uap:
        aa = time.time()
        for image_i in range(args.train_img_num):
            #Load Train Image
            
            image_name = file_list[image_i]
            image_file = os.path.join(data_dir, file_list[image_i])
            image = cv2.imread(image_file)
            image = cv2.resize(image, (1024, 1024))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if(args.attack_type == 'enc_rm_inf_2aug' or args.attack_type == 'understand_cosine_sim'):
                image_pos = cv2.imread(file_paths[image_i])
                image_pos = cv2.resize(image_pos, (1024, 1024))
                image_pos = cv2.cvtColor(image_pos, cv2.COLOR_BGR2RGB)
                mask_generator_uap.image_positive_source = torch.from_numpy(image_pos).permute(2, 0, 1).cuda().contiguous()
                mask_generator_uap.image_positive = achor_aug_list[image_i].squeeze(0)
                #import pdb;pdb.set_trace()
            mask_generator_uap.generate_delta(image, 
                                              attack_point_per_size=args.point_number_size, 
                                              attack_point_per_batch=train_point_batch, 
                                              max_conf_point =args.max_conf_point, 
                                              min_conf_point=args.min_conf_point, 
                                              random_point=args.random_point, 
                                              multimask_output=args.multimask_output_off)
            # break
            # print('attack time:', time.time() - start)
        #Save UAP
        destination_file_delta = os.path.join(current_path, 'delta')
        if not os.path.exists(destination_file_delta):
                os.makedirs(destination_file_delta) 
        # import pdb;pdb.set_trace()
        file_path_delta = os.path.join(destination_file_delta, 'uap.pth')
        torch.save(uap.delta.data, file_path_delta)
        del destination_file_delta
        bb = time.time()
        ttt = bb-aa
        print ('time is ??',ttt)
        import pdb;pdb.set_trace()
    ################################### Test UAP #############################
    # delta = torch.load('uap.pth')
    
    # Set Variable
    total_iou = []
    total_area_ratio = []
    total_area_gt = []
    total_area_adv = []

    #Save test result annd image Path
    # out_dir = os.path.join("results/", args.experiment)
    destination_file_test_results = os.path.join(current_path, 'test_results')
    if not os.path.exists(destination_file_test_results):
        os.makedirs(destination_file_test_results)
    
    file_name = '{}_source={}_target={}_point={}_steps={}_init={}_eps={}_alpha={}_max_conf={}_min_conf={}_rand_point={}_multimask_output_off={}_seed={}'.format(
        args.attack_type, args.source_model, args.target_model, args.point_number_size, args.attack_steps,  args.attack_init, args.eps, args.alpha, args.max_conf_point, args.min_conf_point, args.random_point,args.multimask_output_off, args.seed)
    print(file_name)  
    result_file = open(os.path.join(destination_file_test_results, file_name + '.txt'), 'w')
    
    destination_file_test_img = os.path.join(current_path, 'test_image')
    if not os.path.exists(destination_file_test_img):
        os.makedirs(destination_file_test_img) 
    
    
    #################Start Test################## 
    for image_test_i in range(args.test_img_num):
        #Load Test Image
        image_name_test = file_list_test[image_test_i]
        image_file_test = os.path.join(data_dir_test, file_list_test[image_test_i])
        image_test = cv2.imread(image_file_test)
        image_test = cv2.resize(image_test, (1024, 1024))
        image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
        
        
        #generate adversarial image and masks
        adv_image, masks_adv, masks_clean, points, _, _ = mask_generator_uap.attack_image(image_test, attack_point_per_size=args.point_number_size, attack_point_per_batch=1000, max_conf_point =args.max_conf_point, min_conf_point=args.min_conf_point,random_point=test_random_point, multimask_output=args.multimask_output_off)
        adv_image = adv_image[0,:,:,:].permute(1, 2, 0).cpu().detach().numpy()
        
        #image preprocess
        transform = ResizeLongestSide(1024)
        transformed_image = transform.apply_image(image_test)
        transformed_point = transform.apply_coords(points, (image_test.shape[0], image_test.shape[1]))
        # import pdb;pdb.set_trace()
        #visualize results
        if args.show_img==1:
            for i in range(transformed_point.shape[0]):
                if not os.path.exists(destination_file_test_img + '/clean_image/'):
                    os.makedirs(destination_file_test_img + '/clean_image/')
                    os.makedirs(destination_file_test_img + '/clean_seg_result/')
                    os.makedirs(destination_file_test_img + '/adv_image/')
                    os.makedirs(destination_file_test_img + '/adv_seg_result/')
                # save clean image
                #import pdb;pdb.set_trace()
                plt.figure()
                plt.imshow(transformed_image)
                plt.scatter(transformed_point[i,0], transformed_point[i,1], color='green', marker='*', s=1000, edgecolor='white', linewidth=1.75)
                plt.axis('off')
                plt.savefig("{}.png".format(destination_file_test_img + '/clean_image/'+image_name_test+ str(i) +image_name_test+ file_name), bbox_inches='tight', pad_inches = 0.0)
                plt.show()

                # save clean masks
                blank_image = np.zeros_like(image_test)
                plt.figure()
                plt.imshow(blank_image)
                masks = masks_clean[i] > 0.0
                masks = masks.detach().cpu().numpy()
                color = np.array([1.0, 1.0, 1.0])
                mask_image = masks[0]
                result_image =mask_image[:,:,None]*color.reshape(1, 1, -1) 
                plt.imshow(result_image)
                plt.axis('off')
                plt.savefig("{}.png".format(destination_file_test_img + '/clean_seg_result/'+image_name_test + str(i)+ file_name), bbox_inches='tight', pad_inches = 0.0)
                print("save clean seg result success")

                # save adv image
                plt.figure()
                plt.imshow(adv_image)
                plt.scatter(transformed_point[i,0], transformed_point[i,1], color='green', marker='*', s=1000, edgecolor='white', linewidth=1.75)
                plt.axis('off')
                plt.savefig("{}.png".format(destination_file_test_img + '/adv_image/'+image_name_test + str(i) + file_name), bbox_inches='tight', pad_inches = 0.0)
                plt.show()

                # save adv mask
                blank_image = np.zeros_like(image_test)
                plt.figure()
                plt.imshow(blank_image)
                masks = masks_adv[i] > 0.0
                masks = masks.detach().cpu().numpy()
                color = np.array([1.0, 1.0, 1.0])
                mask_image = masks[0]
                result_image =mask_image[:,:,None]*color.reshape(1, 1, -1) 
                plt.imshow(result_image)
                plt.axis('off')
                plt.savefig("{}.png".format(destination_file_test_img + '/adv_seg_result/'+image_name_test + str(i) +file_name), bbox_inches='tight', pad_inches = 0.0)
                print("save adv seg result success")
                if i ==5:
                    break
            
            #save delta image
            delta_image = (uap.delta.data*255 * 15  + 120).permute(1,2,0).cpu().detach().numpy()
            cv2.imwrite("{}.png".format(destination_file_test_img + '/delta_image_' + file_name), delta_image)
            #import pdb;pdb.set_trace()

      

        masks_clean = masks_clean > 0.0
        masks_clean = masks_clean.detach()
        masks_adv = masks_adv > 0.0
        masks_adv = masks_adv.detach()
        #import pdb;pdb.set_trace()
        iou_matrix, area = get_iou_matrix_tensor(masks_clean, masks_adv)
        #import pdb;pdb.set_trace()
        iou = torch.max(iou_matrix, dim=1)[0].tolist()
        #import pdb; pdb.set_trace()
        total_iou += iou

        area_gt = torch.sum(masks_clean).item()
        area_adv = torch.sum(masks_adv).item()
        area_ratio = area_adv / area_gt

        total_area_ratio.append(area_ratio)
        total_area_gt.append(area_gt)
        total_area_adv.append(area_adv)

        print(f'iou: {iou[0]} area_ratio: {area_ratio} test_number: {image_test_i} image_name: {image_name_test}')
    #####tensor
    score = torch.mean(torch.tensor(total_iou)).item()
    area_ratio_sum = torch.sum(torch.tensor(total_area_adv)) / torch.sum(torch.tensor(total_area_gt)).item()
    
    print(f"end_miou: {score}")
    result_file.write('miou: ' + str(score) + '\n')
    result_file.write('area_ratio_sum: ' + str(area_ratio_sum) + '\n')

    result_file.close()

if __name__ == '__main__':
    main(args)


