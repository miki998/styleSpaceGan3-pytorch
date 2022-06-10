import torch
import pandas as pd
import numpy as np 
import pickle
import io

# ref: https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

# refs: (1st: compute IOU script | 2nd: rank the different values over the generated images/style channels )
# - https://github.com/betterze/StyleSpace/blob/main/align_mask.py
# - https://github.com/betterze/StyleSpace/blob/main/semantic_channel.py 

def ExpendSMask(semantic_masks,num_semantic):
    
    semantic_masks2=[]
    for i in range(num_semantic):
        tmp=semantic_masks==i
        semantic_masks2.append(tmp)
    semantic_masks2=np.array(semantic_masks2)
    semantic_masks2=np.transpose(semantic_masks2, [1,0,2,3])
    return semantic_masks2
    

def OverlapScore(mask2,tmp_mask):
    o=tmp_mask.sum() #size of semantic mask
    if o==0:
        return np.nan,np.nan,np.nan
    
    p=o/(mask2.shape[0]*mask2.shape[1])
    
    threshold=np.percentile(mask2.reshape(-1),(1-p)*100)
    gmask=mask2>threshold

    n=np.logical_and(gmask,tmp_mask).sum().item()
    u=np.logical_or(gmask,tmp_mask).sum().item()
    o = o.item()

    return n,u,o
    
def GetScore(mask2,semantic_mask2):
    scores=[]
    for i in range(len(semantic_mask2)):
        tmp_mask=semantic_mask2[i]
        n,u,o=OverlapScore(mask2,tmp_mask)
        scores.append([n,u,o])
    scores=np.array(scores)
    return scores

def TopRate(all_var_grad):
    num_layer=len(all_var_grad)
    num_semantic=all_var_grad[0].shape[2]
    discount_factor=2 #large number means pay higher weight precision (prefer small area) 
    all_count_top=[]
    for lindex in range(num_layer):
        layer_g=all_var_grad[lindex]
        num_channel=layer_g.shape[1]
        count_top=np.zeros([num_channel,num_semantic])
        for cindex in range(num_channel):
            semantic_in=layer_g[:,cindex,:,0]/(layer_g[:,cindex,:,2]**discount_factor)
            semantic_top=np.nanargmax(np.abs(semantic_in),axis=1)
            
            semantic_top=pd.Series(semantic_top)
            tmp=semantic_top.value_counts()
            count_top[cindex,tmp.index]=tmp.values
        all_count_top.append(count_top)
    
    tmp=all_var_grad[0][:,0,:,2]
    mask_counts2=~np.isnan(tmp)
    mask_counts3=mask_counts2.sum(axis=0)
    mask_counts3[mask_counts3==0]=1 # ignore 0
    
    all_count_top2=[]
    for lindex in range(len(all_count_top)):
        all_count_top2.append(all_count_top[lindex]/mask_counts3)
    return all_count_top2

def computeIU(semantic_masks, gradMaps):
    out_size = 32
    num_per  = 100

    num_semantic=int(semantic_masks.max()+1)
    semantic_masks2=ExpendSMask(semantic_masks.numpy(),num_semantic)

    mask_size=semantic_masks2.shape[-1]
    step=int(mask_size/out_size)

    semantic_masks2=semantic_masks2.reshape(int(num_per),num_semantic,out_size,step,out_size,step)

    semantic_masks2=np.sum(semantic_masks2,axis=(3,5))
    semantic_masks2_single=np.argmax(semantic_masks2,axis=1)

    semantic_masks2=ExpendSMask(semantic_masks2_single,num_semantic)

    all_scores=[]
    for linex in range(len(gradMaps)):
        print('layer index: ',linex)
        layer_g=gradMaps[linex]
        num_img,num_channel,_=layer_g.shape
        
        scores2=np.zeros((num_img,num_channel,num_semantic,3))
        for img_index in range(num_img):
            semantic_mask2=semantic_masks2[img_index]
            for cindex in range(num_channel):
                mask=layer_g[img_index,cindex].reshape((3,out_size,out_size))
                mask2=np.abs(mask).mean(axis=0)  #need code 
                
                scores=GetScore(mask2,semantic_mask2)
                scores2[img_index,cindex,:,:]=scores
        all_scores.append(scores2)
    
    return all_scores    