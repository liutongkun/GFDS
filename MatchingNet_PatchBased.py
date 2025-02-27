import torch
from torch import nn
import torch.nn.functional as F

def get_pdn_large(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)
    )

def get_pdn_small(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    basechannel = 128
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=basechannel, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=basechannel, out_channels=basechannel*2, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=basechannel*2, out_channels=basechannel*2, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=basechannel*2, out_channels=out_channels, kernel_size=4)
    )

class MatchingNet(nn.Module):
    def __init__(self,distance='cosine',backbone_name = 'large'):
        super(MatchingNet, self).__init__()
        self.backbone_name = backbone_name
        self.distance = distance
        if self.backbone_name =='large':
            backbone = get_pdn_large(384, padding=True)
            checkpath = 'model_large.pth'
            backbone.load_state_dict(torch.load(checkpath,map_location='cuda:0'))
            self.network = backbone

        if self.backbone_name =='small':
            backbone = get_pdn_small(384, padding=True)
            checkpath = 'model_small.pth'
            backbone.load_state_dict(torch.load(checkpath,map_location='cuda:0'))
            self.network = backbone


    def forward(self, img_s_list, mask_s_list, img_q):
        h, w = img_q.shape[-2:]
        # feature maps of support images
        feature_s_list = []
        for k in range(len(img_s_list)):
            with torch.no_grad():
                s_0 = self.network(img_s_list[k])
            feature_s_list.append(s_0)
            del s_0
        # feature map of query image
        with torch.no_grad():
            feature_q = self.network(img_q)
        # foreground(target class) and background prototypes pooled from K support features
        feature_fg_list = []
        feature_bg_list = []

        for k in range(len(img_s_list)):
            mask = (mask_s_list[k] == 1).float()
            fg_mask_indices = (mask[0, :, :] == 1).nonzero()
            scale = mask.shape[1] / (feature_s_list[k].shape[2] + 0.000001)
            fg_mask_indices_resize = (fg_mask_indices / scale).to(torch.int64)
            fg_mask_indices_resize = torch.clamp(fg_mask_indices_resize, max=feature_q.shape[2]-1)
            fg_mask_indices_resize = torch.unique(fg_mask_indices_resize,dim=0)
            fg_mask_indices_resize = fg_mask_indices_resize.to(torch.int64)

            bg_mask = torch.zeros([feature_q.shape[2],feature_q.shape[2]])
            bg_mask[fg_mask_indices_resize[:,0],fg_mask_indices_resize[:,1]] = 1
            bg_mask_indices_resize = (bg_mask==0).nonzero()
            bg_mask_indices_resize = bg_mask_indices_resize.to(torch.int64)
            fg_extracted_values = feature_s_list[k][:,:,fg_mask_indices_resize[:,0],fg_mask_indices_resize[:,1]]#(B,C,N)

            #'''
            kernel_size = 3
            stride = 1
            if fg_extracted_values.shape[-1]>10:
                fg_extracted_values = fg_extracted_values.unfold(2, kernel_size, stride)
                fg_extracted_values = fg_extracted_values.mean(dim=-1)
            #'''

            bg_extracted_values = feature_s_list[k][:,:,bg_mask_indices_resize[:,0],bg_mask_indices_resize[:,1]]

            '''
            kernel_size = 3
            stride = 1
            if bg_extracted_values.shape[-1]>10:
                bg_extracted_values = bg_extracted_values.unfold(2, kernel_size, stride)
                bg_extracted_values = bg_extracted_values.mean(dim=-1)
            '''

            feature_fg_list.append(fg_extracted_values)
            feature_bg_list.append(bg_extracted_values)

        FP = torch.cat(feature_fg_list, dim=-1)
        BP = torch.cat(feature_bg_list, dim=-1)
        #avg
        #FP = torch.mean(FP,dim = -1,keepdim=True)
        #BP = torch.mean(BP,dim = -1,keepdim=True)

        # measure the similarity of query features to fg/bg prototypes
        if self.distance == 'cosine':
            out_0_feature = self.similarity_func_dense(feature_q, FP, BP)
        if self.distance == 'l2':
            out_0_feature = self.similarity_func_dense_l2(feature_q, FP, BP)

        out_0 = F.interpolate(out_0_feature, size=(h, w), mode="bilinear", align_corners=True)
        return out_0

    def similarity_func_dense(self, feature_q, fg_proto, bg_proto):
        similarity_fg = cosine_similarity(feature_q,fg_proto)
        similarity_fg, _ = torch.max(similarity_fg,dim=1)
        similarity_bg = cosine_similarity(feature_q,bg_proto)
        similarity_bg, _ = torch.max(similarity_bg,dim=1)
        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out

    def similarity_func_dense_l2(self, feature_q, fg_proto, bg_proto):
        similarity_fg = l2_distance(feature_q,fg_proto)
        similarity_fg = 1/(similarity_fg+0.00000001)
        similarity_fg, _ = torch.max(similarity_fg,dim=1)
        similarity_bg = l2_distance(feature_q,bg_proto)
        similarity_bg = 1/(similarity_bg+0.00000001)
        similarity_bg, _ = torch.max(similarity_bg,dim=1)
        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out


def cosine_similarity(tensor1, tensor2):
    tensor1_norm = torch.nn.functional.normalize(tensor1, p=2, dim=1)
    tensor2_norm = torch.nn.functional.normalize(tensor2, p=2, dim=1)
    tensor1_flat = tensor1_norm.view(1, tensor1.shape[1], -1).transpose(1, 2)
    cos_sim_flat = torch.bmm(tensor1_flat, tensor2_norm)
    cos_sim = cos_sim_flat.transpose(1, 2).view(1, tensor2.shape[2], tensor1.shape[2], tensor1.shape[3])
    return cos_sim

def l2_distance(tensor1, tensor2):
    tensor1_re = tensor1.permute(0,2,3,1).reshape(tensor1.shape[0],-1,tensor1.shape[1])
    tensor2 = tensor2.permute(0,2,1)
    dist = torch.cdist(tensor1_re,tensor2)
    dist = dist.permute(0,2,1)
    dist = dist.reshape(tensor1.shape[0],tensor2.shape[1],tensor1.shape[2],tensor1.shape[3])
    return dist
