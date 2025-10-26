#!/usr/bin/env python3
import os
import csv
from PIL import Image

""" --------------- GLOBAL VARIABLES --------------- """

SRC_ROOT = 'input/' # top-level directory containing datasets
OUTPUT_ROOT = 'dataset/data/' # directory where pytorch-readable datasets will live
DATA_DIRS = [
        'ai-generated-images-vs-real-images',
        'cifake-real-and-ai-generated-synthetic-images']
DATA_LABELS = {
    0: 'real', 
    1: 'fake'
}
DATA_TYPE = 'images'

""" --------------- FUNCTIONS --------------- """

def load_image(img_path):
    """
    Loads an image 
    
    Converts non-RGB, non-grayscale types to RGB 
    """
    try:
        img = Image.open(img_path)
        img.load()
    except OSError:
        return None
    
    if img.mode != 'RGB' and img.mode != 'L':
        if img.info.get("transparency") is not None:
            img = img.convert('RGBA').convert('RGB')
        else:
            img = img.convert('RGB')
    return img



def structure_data():
    """ 
    Converts dataset folders of the form:

        SRC_ROOT/dataset/{train|test}/{label}/image_file

    into: 

        OUTPUT_ROOT/dataset/{train|test}/images/image_file
        OUTPUT_ROOT/dataset/{train|test}/labels.csv

    Each 'labels.csv' file contains two columns:
        filename,label

    For example:
        image_0001.png,real
        image_0002.png,fake

    Args:
        None (uses global constants: SRC_ROOT, OUTPUT_ROOT, DATA_DIRS, TRAIN_DIR, TEST_DIR)

    Returns:
        None
    """   
    data_src = [os.path.join(SRC_ROOT, dataset) for dataset in DATA_DIRS]
    
    for data_dir in data_src:
        dataset = os.path.relpath(data_dir, SRC_ROOT)
        for split in ['test', 'train']: 
            dest = os.path.join(OUTPUT_ROOT, dataset, split, DATA_TYPE)
            labels = {}
            for label in DATA_LABELS:
                src_dir = os.path.join(data_dir, split, DATA_LABELS[label])

                for dirpath, _, filenames in os.walk(src_dir):
                    for filename in filenames:
                        new_filename = f'{DATA_LABELS[label]}_{filename}'
                        img_path = os.path.join(dirpath, filename)
                        try:
                            img = load_image(img_path)
                            if img is None:
                                continue
                            os.makedirs(dest, exist_ok=True)
                            dest_path = os.path.join(dest, new_filename)
                            if os.path.exists(dest_path):
                                continue
                            img.save(dest_path)
                            labels[new_filename] = label
                            img.close()
                        except Exception as e:
                            print(e)
                        
            csv_path = os.path.join(os.path.dirname(dest), 'labels.csv')
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['file', 'label'])
                for filename, label in labels.items():
                    writer.writerow([filename, label])
             
if __name__ == '__main__':
    os.makedirs('dataset/data', exist_ok=True)
    structure_data()

            
    
    


