import os,shutil,glob
from pathlib import Path
from bdd_category_dict import bdd_val_video_id_dict

# this file rename_bdd_data.py is to change bdd file structure
# to be the same with kitti, so deep_sort can take it as input.
def rename_folder(folders):
    for folder_one in folders:
        if folder_one in bdd_val_video_id_dict.keys():
            folder_path = bdd_path+'/'+folder_one
            target_name = bdd_path+'/'+str(bdd_val_video_id_dict[folder_one]).zfill(4)
            shutil.move(folder_path,target_name)
def rename_imgs(folders):
    for folder_one in folders:
        folder_path = bdd_path+'/'+folder_one
        file_names = glob.glob(folder_path+'/*.jpg')
        for file_name_one in file_names:
            img_name = Path(file_name_one).stem.split('-')[2]
            target_name = folder_path+'/'+img_name+'.jpg'
            shutil.move(file_name_one,target_name)
def add_folder(folders):
    # add folder img1.
    for folder_one in folders:
        folder_path = bdd_path+'/'+folder_one
        file_names = glob.glob(folder_path+'/*.jpg')
        for file_name_one in file_names:
            img_name = Path(file_name_one).name
            target_folder = folder_path+'/img1/'
            target_name = target_folder+img_name
            os.makedirs(target_folder,exist_ok=True)
            shutil.move(file_name_one,target_name)
if __name__=='__main__':
    # first rename folder, then rename imgs, then add img1 folder.
    RENAME_FOLDER_FLAG=False
    RENAME_IMGS_FLAG=False
    ADD_FOLDER=True
    bdd_path = '../dataset/BDD_MOTS/val_set'
    folders = os.listdir(bdd_path)
    if RENAME_FOLDER_FLAG:
        rename_folder(folders)
    if RENAME_IMGS_FLAG:
        rename_imgs(folders)
    if ADD_FOLDER:
        add_folder(folders)
