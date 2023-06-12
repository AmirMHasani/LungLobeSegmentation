from lungmask import mask
import SimpleITK as sitk
import pydicom
import numpy as np 
import cv2
import os 
import tqdm 
import shutil 
import torch 
import pandas as pd 
import glob 
import gc
import pydicom
import matplotlib.pyplot as plt 

from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from skimage import measure

print("Please clone this repo to segment the lobes in lungs https://github.com/JoHof/lungmask")
input_csv_file = "input_data.csv" # Contains patient ids and corresponding dcm slices of full lungs 
output_csv_file = "output_data.csv" # output file with dilation to save the data 

model_name = 'LTRCLobes'
main_output_dir = "output_dir"
cyst_HU = -900
cyst_area_threshold = 400 # If a segmented region has area greater than this then it is not a cyst
cyst_area_min =10 

if(not os.path.exists(main_output_dir)):
    os.mkdir(main_output_dir)
else:
    shutil.rmtree(main_output_dir)
    os.mkdir(main_output_dir)
    
lobes_ids = {"1":"left-upper",
             "2":"left-lower",
             "3": "right-upper",
             "4": "right-middle",
             "5": "right-lower"}

output_df = pd.DataFrame(columns=["Patient_ID"]+list(lobes_ids.values()) + ["left-total","right-total","whole-lung"])


def show_predictions(dcm_path,save_file=None):
    """
    for dcm_path of full lung predict lobes of lungs using lungmask repo. 
    """
    model = mask.get_model('unet',"LTRCLobes")
    image = sitk.GetArrayFromImage(sitk.ReadImage(dcm_path))
    segment = mask.apply(image,model,volume_postprocessing=True)  # default model is U-net(R231)

    plt.figure(figsize=(40,20))

    plt.subplot(2,4,1)
    plt.title("Full image",fontsize=48)
    image = sitk.GetArrayFromImage(sitk.ReadImage(dcm_path))
    plt.imshow(image[0,:,:],cmap="gray")
    plt.axis("off")
    plt.subplot(2,4,2)
    plt.title("all lobes",fontsize=48)
    plt.imshow(segment[0,:,:])
    plt.axis("off")

    unique_ids = np.unique(segment)


    if(len(unique_ids) > 1):
        for idx in unique_ids[1:]:
            plt.subplot(2,4,idx+2)
            plt.title(lobes_ids[str(idx)],fontsize=48)
            lobe = (segment==idx).astype(int)
            plt.imshow(lobe[0,:,:])
            plt.axis("off")
        if(save_file):
            plt.savefig(f"{save_file}",bbox_inches="tight")
            plt.close()
            plt.close("all")
            plt.clf()
            gc.collect

            
    return 

def get_modified_dcm(dcm_path,image_array):
    """
    get new dcm file with mask as the array 
    """
    dcm_image = sitk.ReadImage(dcm_path)
    modified_image = sitk.GetImageFromArray(image_array)
    modified_image.CopyInformation(dcm_image)
    return modified_image

def get_masks(dcm_path,output_dir):
    """
    predict and save all the predicted lobes using the pretrained model 

    dcm_path: dcm path of the full lung dcm file. Please make sure to have the file in 
    transfer style 1.2.840.10008.1.2.1 . The lung segmentation model does not work well for RLE encoded files. 

    output: A new directory which contains all the 5 lobes along with total left,right and lung. 
    If the model does not predict any of the 5 lobes it means the the lobe cannot be segmented in the images 
    Please note there might be empty directories since the model could not segment any of the lobes. 
    
    """
    model = mask.get_model('unet',"LTRCLobes")
    image = sitk.GetArrayFromImage(sitk.ReadImage(dcm_path))
    segment = mask.apply(image,model,volume_postprocessing=True)  # default model is U-net(R231)
    image_array = sitk.GetArrayFromImage(sitk.ReadImage(dcm_path))
    unique_ids = np.unique(segment)

    fname = dcm_path.split("/")[-1] # might need to fix for windows 
    out_path= f"{output_dir}/{fname[:-4]}"
    if(os.path.exists(out_path)):
        shutil.rmtree(out_path)
    os.mkdir(out_path)
    
    sitk.WriteImage(sitk.ReadImage(dcm_path),os.path.join(out_path,fname)) # write original file 
    
    if(len(unique_ids) > 1): # Save only when something is predicted 
        show_predictions(dcm_path,save_file=os.path.join(out_path,"ML_lobes_segmentation.png"))
        for idx in unique_ids[1:]: # Check which lobes are predicted and save them 
            lobe = (segment==idx).astype(int)
            modified_image = get_modified_dcm(dcm_path,np.array(lobe,dtype=np.int8))
            new_fname = fname.replace(".dcm",f"-{lobes_ids[str(idx)]}.dcm")
            sitk.WriteImage(modified_image,os.path.join(out_path,new_fname))
        
        # Total left lung 
        lobe_lower = (segment==1).astype(int)
        lobe_upper = (segment==2).astype(int)
        total_left = lobe_lower+lobe_upper
        total_left[total_left>1] = 1
        modified_image = get_modified_dcm(dcm_path,np.array(total_left,dtype=np.int8))
        new_fname = fname.replace(".dcm",f"-left-total.dcm")
        sitk.WriteImage(modified_image,os.path.join(out_path,new_fname))
        
        # Total right lung 
        lobe_middle = (segment==3).astype(int)
        lobe_lower = (segment==4).astype(int)
        lobe_upper = (segment==5).astype(int)
        total_right = lobe_middle + lobe_lower+lobe_upper
        total_right[total_right>1] = 1
        modified_image = get_modified_dcm(dcm_path,np.array(total_right,dtype=np.int8))
        new_fname = fname.replace(".dcm",f"-right-total.dcm")
        sitk.WriteImage(modified_image,os.path.join(out_path,new_fname))

        
        # Total lung 
        total_lung = total_left + total_right
        total_lung [total_lung >1] = 1
        modified_image = get_modified_dcm(dcm_path,np.array(total_lung ,dtype=np.int8))
        new_fname = fname.replace(".dcm",f"-whole-lung.dcm")
        sitk.WriteImage(modified_image,os.path.join(out_path,new_fname))
        
    return out_path, fname



def get_cyst_lobe_area(dcm_path,lobe_dcmpath,region,show=True):
    """
    dcm_path: dcm image of full lung 
    lobe_dcmpath: dcm image for segmented mask of a lobe 
    region: lobe name 

    For the dcm image of segmented lung 
    1. Segment the cysts using thresholds 
    2. Calculate area of cysts 
    
    return lobe area and total cyst area 
    """
    full_image = sitk.GetArrayFromImage(sitk.ReadImage(dcm_path)) # Full image 
    lobe_image = sitk.GetArrayFromImage(sitk.ReadImage(lobe_dcmpath)) # Segmented lobe image, has values 0 and 1 
    # Calculation of lobe area 
    contours = measure.find_contours(lobe_image[0],0.95)    
    lobe_area = 0
    for contour in contours:
        hull = ConvexHull(contour)
        lobe_area+=hull.volume
    
    # segmentation of cysts 
    valid_image = full_image*lobe_image
    cysts =(valid_image>cyst_HU).astype(int)
    
    # Calculatin of cysts area 
    contours = measure.find_contours(cysts[0],0.95)
    cysts_area = 0
    for contour in contours:
        hull = ConvexHull(contour)
        if(hull.volume > cyst_area_min):
            cysts_area+=hull.volume
    
    if(show):
        plt.figure(figsize=(40,40))
        plt.subplot(2,2,1)
        plt.title("Full image",fontsize=32)
        plt.imshow(full_image[0,:,:],cmap="gray")
        plt.axis("off")
        
        plt.subplot(2,2,2)
        plt.title(f"{region}  area: {lobe_area:0.4f}",fontsize=32)
        plt.imshow(lobe_image[0,:,:],cmap="gray")
        plt.axis("off")  
        
        plt.subplot(2,2,3)
        plt.title("valid_image",fontsize=32)
        plt.imshow(valid_image[0,:,:],cmap="gray")
        plt.axis("off")  
        
        
        plt.subplot(2,2,4)
        plt.title(f"cysts using HU value < {cyst_HU} \n area:{cysts_area:0.3f}",fontsize=32)
        plt.imshow(cysts[0,:,:],cmap="gray")
#         plt.plot(bound_contour[:, 1], bound_contour[:, 0], linewidth=2,color="red")
        plt.axis("off")  
        
        plt.savefig(lobe_dcmpath.replace(".dcm","cyst_segmentation.png"),bbox_inches="tight")
        plt.close()
        plt.close("all")
    
    return lobe_area, cysts_area, valid_image[0,:,:]


input_data=pd.read_csv(input_csv_file)
print(f"Found data of {len(input_data)} patients in csv file ")

for row_num in range(len(input_data)):
    patient_id = input_data.iloc[row_num]["Patient_ID"] # Patient id 
    dcm_folder = input_data.iloc[row_num]["Full_DCM_Path"] # Image title full_dcm_data = sorted(list(glob.glob(f"{dcm_folder}*.dcm")))
    full_dcm_data = sorted(list(glob.glob(f"{dcm_folder}/*.dcm")))
    print(f"Found {len(full_dcm_data)} dcm files for patient {patient_id} in folder: {dcm_folder}")
    output_dir = os.path.join(main_output_dir,str(patient_id))
    os.mkdir(output_dir)
    
    cyst_scores = np.zeros((8))
    for j, dcm_path in enumerate(tqdm.tqdm(full_dcm_data)):
        slice_path,fname = get_masks(dcm_path,output_dir)
    
        dcm_path = os.path.join(slice_path,fname) # full image dcm image 
        regions = list(lobes_ids.values()) + ["left-total","right-total","whole-lung"] # all the regions for calculation 
        for idx,region in enumerate(regions):
            try:
                lobe_dcmpath = os.path.join(slice_path,fname.replace(".dcm",f"-{region}.dcm")) # lobes dcm image
                lobe_area, cysts_area, valid_image = get_cyst_lobe_area(dcm_path,lobe_dcmpath,region)
            except:
                lobe_area, cysts_area = 1.0, 0.0 
            lobe_area = max(1,lobe_area)
            cyst_scores[idx] = cyst_scores[idx] + (cysts_area/lobe_area)
    cyst_scores = cyst_scores/len(full_dcm_data) # Average over all the slices of the patient 
    output_df.loc[row_num] = [patient_id] + list(cyst_scores)


output_df.to_csv(output_csv_file,index=False)



