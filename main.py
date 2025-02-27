from MatchingNet_PatchBased import MatchingNet
from utils import set_seed, mIOU, see_img,see_img_heatmap_onlyseg,remove_overlap
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from fewshot_defectloader import FewshotDataset
import numpy as np
from FastSAM_main.fastsam import FastSAM, FastSAMPrompt #please download it from https://github.com/CASIA-IVA-Lab/FastSAM
from scipy.ndimage import label
import torch.nn.functional as F

dilation_mask = torch.ones(21,21).cuda() #used for morphological filter
def main(testcategory,filedir,savedir,resize_shape,model_name='large',useSAM=False,shot=1,visualize=False,sam_model_path=None):
    seed = 2
    set_seed(seed)
    '''
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    '''
    device = torch.device(
        "cuda")
    if useSAM:
        # warmup FastSAM
        sam_iou = 0.5
        sam_conf = 0.1
        sam_retina_mask = True
        sam_imgsz = resize_shape[0]
        model = FastSAM(sam_model_path)
        model.set_predict(
            device = device,
            retina_masks = sam_retina_mask,
            imgsz = sam_imgsz,
            conf = sam_conf,
            iou = sam_iou
        )
        warmup_img = torch.zeros([1,3,sam_imgsz,sam_imgsz]).cuda()
        predictor = model.predictor
        predictor.batch = ['']  # path
        predictor.imgsz = resize_shape
        warm_numpy = np.zeros([sam_imgsz,sam_imgsz,3])
        preds_sam_warmup = predictor.model(warmup_img, augment=predictor.args.augment, visualize=False)
        results_sam = predictor.postprocess(preds_sam_warmup, warmup_img, [warm_numpy])

    testset = FewshotDataset(filedir, resize_shape,shot=shot,testcategory=testcategory)

    testloader = DataLoader(testset, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=0, drop_last=False)
    model = MatchingNet(backbone_name=model_name,distance = distance)
    model = model.cuda()
    tbar = tqdm(testloader)
    num_classes = len(testcategory)+1
    metric = mIOU(num_classes)
    mean2 = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).cuda()
    std2 = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).cuda()
    for i, (img_s_list, mask_s_list, img_q, mask_q, cls, _, id_q,key) in enumerate(tbar):
        img_q, mask_q = img_q.cuda(), mask_q.cuda()
        #sam_module
        img_q_denor = img_q * std2 + mean2
        img_q_numpy = img_q_denor.squeeze().permute(1,2,0).cpu().numpy()
        img_q_numpy = img_q_numpy*255
        img_q_numpy = img_q_numpy.astype(np.uint8)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if useSAM:
                preds_sam = predictor.model(img_q_denor, augment=predictor.args.augment, visualize=False)
                everything_results = predictor.postprocess(preds_sam, img_q_denor, [warm_numpy])
                prompt_process = FastSAMPrompt(img_q_numpy, everything_results, device=device)
                if len(everything_results) > 0:
                    sam_mask = everything_results[0].masks.data.permute(1,2,0)
                else:
                    useSAM = False

        for k in range(len(img_s_list)):
            img_s_list[k], mask_s_list[k] = img_s_list[k].cuda(), mask_s_list[k].cuda()
        classname = key[0].split('_')[0]
        cls = cls[0].item()
        key = key[0]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(img_s_list, mask_s_list, img_q)
            pred = torch.argmax(pred, dim=1)
            pred[pred == 1] = cls
            mask_q[mask_q == 1] = cls

            if visualize:
                sub_dir = f'{savedir}/'+key
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)
                sub_dir = sub_dir + '/'
                predresult = pred[0, :, :].cpu().numpy()
                mask_s_numpy =mask_s_list[0][0, :, :].cpu().numpy()
                mask_q_numpy = mask_q[0, :, :].cpu().numpy()
                see_img_heatmap_onlyseg(predresult,sub_dir, i, 'nosam_result') #original
                see_img_heatmap_onlyseg(mask_s_numpy, sub_dir, i,'supportmask')
                see_img_heatmap_onlyseg(mask_q_numpy, sub_dir, i,'gtmask')
            if useSAM == True:
                pred_permute = pred.permute(1,2,0)
                sam_mask = remove_overlap(sam_mask)
                intersection = torch.logical_and(pred_permute, sam_mask)
                intersection_sum = torch.sum(intersection, dim=(0,1))
                mask_sum = torch.sum(sam_mask,dim=(0,1)).float()
                over_lap = intersection_sum.float()/ mask_sum
                indices_area = torch.nonzero((over_lap > 0.2)&((mask_sum<0.4*sam_imgsz*sam_imgsz)|(over_lap>0.6)))
                indices = indices_area[:,0]
                if useBoundary:
                    pred_r = pred_permute.clone()
                    pred_r_dil = pred.unsqueeze(0).unsqueeze(0)
                    pred_r_dil = pred_r_dil.squeeze().cpu().numpy()
                    labeled, num_r = label(pred_r_dil)
                    labeled = torch.from_numpy(labeled).cuda().unsqueeze(-1)
                    pred_r = pred_r*labeled
                    s_mask_all = torch.zeros_like(pred_r)
                    mask_refine_all = torch.zeros_like(pred_r)

                if indices.shape[0] != 0:
                    torch.cuda.synchronize()
                    selected_sam_mask = sam_mask[:, :, indices]
                    selected_sam_mask_dil = selected_sam_mask.unsqueeze(0)
                    selected_sam_mask_dil = selected_sam_mask_dil.permute(3,0,1,2)

                    selected_sam_mask_dil = F.max_pool2d(selected_sam_mask_dil, kernel_size=21, stride=1,
                                           padding=10)
                    for j in range(selected_sam_mask.shape[2]):
                        s_mask = selected_sam_mask[:,:,j].unsqueeze(-1)
                        pred_permute = torch.logical_or(pred_permute,s_mask).to(torch.int64)
                        if useBoundary:
                            s_mask_dil = selected_sam_mask_dil[j,:,:,:]
                            s_mask_dil = s_mask_dil.permute(1,2,0)
                            multi = s_mask_dil*pred_r
                            multi = multi.to(torch.int64)
                            non_zero_tensor = multi[multi!=0]
                            counts = torch.bincount(non_zero_tensor)
                            label_mask = torch.argmax(counts).item()
                            pred_r_label = (pred_r == label_mask).int()
                            inter2 = torch.logical_and(pred_r_label,s_mask_dil)
                            inter2_sum = torch.sum(inter2)
                            sum2_pred = torch.sum(pred_r_label)
                            overlap2 = inter2_sum/sum2_pred
                            if overlap2>0.9:
                                mask_refine = s_mask_dil*(1-s_mask)
                                s_mask_all = torch.logical_or(s_mask_all,s_mask).to(torch.int64)
                                mask_refine_all = torch.logical_or(mask_refine,mask_refine_all).to(torch.int64)
                    if useBoundary:
                        mask_refine_all = mask_refine_all*(1-s_mask_all)
                        pred_permute = pred_permute*(1-mask_refine_all)
                    pred = pred_permute.permute(2,0,1)
            pred[pred == 1] = cls
            mask_q[mask_q == 1] = cls
            metric.add_batch(pred.cpu().numpy(), mask_q.cpu().numpy())
            mIoU,FB_IoU, pre, rec = metric.evaluate_withrecall()
            tbar.set_description("Testing mIOU: %.2f, FBIOU: %.2f" % (mIoU * 100.0,FB_IoU*100))
            if visualize:
                sub_dir = f'{savedir}/'+key
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)
                sub_dir = sub_dir + '/'
                predresult = pred[0, :, :].cpu().numpy()
                see_img(img_s_list[0],sub_dir,i,f'support') #only foreground similarity
                see_img(img_q,sub_dir,i,f'query') #only background similarity
                see_img_heatmap_onlyseg(predresult, sub_dir, i,'predresult_final')
                if useSAM:
                    ann = prompt_process.everything_prompt()
                    prompt_process.plot(
                        annotations=ann,
                        output_path= sub_dir+'/'+f'{i}_samimg.png',
                        bboxes=None,
                        points=None,
                        point_label=None,
                        withContours=False,
                        better_quality=False,
                    )
    mIoU, FB_IoU, pre, rec = metric.evaluate_withrecall()
    print("Final Testing mIOU: %.2f, FBIOU: %.2f, pre: %.2f, recall: %.2f" % (mIoU * 100.0, FB_IoU * 100, pre*100, rec*100))
    return mIoU, FB_IoU, pre, rec

if __name__ == '__main__':
    iou_list = []
    fb_iou_list = []

    ''' 
    These datasets can be download from:
    https://www.mvtec.com/company/research/datasets/mvtec-ad
    https://github.com/amazon-science/spot-diff
    https://www.dagm.de/the-german-association-for-pattern-recognition
    https://github.com/hmyao22/PSP-DS
    https://github.com/jianzhang96/MSD
    https://github.com/bbbbby-99/TGRNet-Surface-Defect-Segmentation
    all_list =[
        ['tile_crack', 'tile_glue_strip', 'tile_gray_stroke', 'tile_oil', 'tile_rough'],
        ['grid_bent','grid_broken','grid_glue','grid_metal_contamination','grid_thread'],
        ['Steel_Am', 'Steel_Ld', 'Steel_Sc'],
        ['KolektorSDD_bad'],
        ['PSP_class_1','PSP_class_2','PSP_class_3','PSP_class_4','PSP_class_5','PSP_class_6','PSP_class_7'],
        ['Class1_bad','Class2_bad','Class3_bad','Class4_bad','Class5_bad','Class6_bad','Class7_bad','Class8_bad',
        'Class9_bad','Class10_bad'],
        ['Small_fins','Small_pit','Small_scratch'],
         ['Side_holes','Side_boundary'],
         ['Large_bubble','Large_pit','Large_pressure','Large_wear'],
        ['Phone_oil', 'Phone_scratch', 'Phone_stain'],
        ['capsules_0','capsules_1','capsules_2','capsules_3','capsules_4'],
        ['macaroni2_0','macaroni2_1','macaroni2_2','macaroni2_4','macaroni2_5','macaroni2_6']
    ]
    miou_weight = [5, 5, 3, 1, 7, 10, 3, 4, 2, 3, 5, 6]
    fbiou_weight = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    '''
    all_list = [ ['Small_fins','Small_pit','Small_scratch'],
                 ['Side_holes','Side_boundary'],
                 ['Large_bubble','Large_pit','Large_pressure','Large_wear']]
    sam_model_path = '/mnt/data/LIUTONGKUNNEW/Open-set-segmentation/FastSAM-main/weights/FastSAM.pt' ##please download it from https://github.com/CASIA-IVA-Lab/FastSAM
    visualize = True #True: save images
    useSAM = False
    useBoundary = False #True: use morphological fusion
    model_name = 'large'
    distance = 'cosine'
    category = 'FMl'
    shot = 1
    filedir = '/home/b211-3090ti/Dataset/' #change it to your own path, The `Dataset' should contain subdatasets like 'Dataset/Large, Dataset/Small, Dataset/Side (The rubberring dataset)'
    savedir = f'{category}_{model_name}_dshot{shot}'
    resize_shape = [256, 256]
    for testfold in all_list:
        iou,fbiou,pre,recall = main(testfold,filedir,savedir,resize_shape,model_name,useSAM,shot,visualize,sam_model_path)
        print(f'miou{iou},fbiou{fbiou}')

        iou_list.append(iou)
        fb_iou_list.append(fbiou)

    ''' for the entire benchmark
    mioup = np.asarray(iou_list)
    mioupw = mioup * miou_weight
    mioup_avg = np.sum(mioupw) / 54
    fbiou = np.asarray(fb_iou_list)
    fbiouw = fbiou * fbiou_weight
    fbiou_avg = np.sum(fbiouw) / 12
    print(f'final_miou{mioup_avg},final_fbiou{fbiou_avg}')
    '''