import cv2
import os
import pandas as pd
from PIL import Image
import pytesseract
from sklearn.model_selection import train_test_split
from tqdm import tqdm

pytesseract.pytesseract.tesseract_cmd = '/usr/share/tesseract/tesseract-ocr'

def image2text(img_path):
    img = cv2.imread(img_path, 0)   
    pil_image = Image.fromarray(img)
    text = pytesseract.image_to_string(pil_image)
    text = ' '.join((text.split()))
    return text


def read_text(txt_path):
    with open(txt_path, 'r') as f:
        text = f.readlines()
    text = ' '.join(text)
    text = ' '.join((text.split()))
    return text


def create_ocr_parquet_files():

    for target in ['0','2','4','6','9']:

        dir_path = os.path.join(r'data/images', target)
        feature_targets, feature_texts = [], []

        image_names = os.listdir(dir_path)

        for image_name in tqdm(image_names):
            image_path = os.path.join(dir_path, image_name)

            try:
                text = image2text(image_path)
                feature_texts.append(text)
                feature_targets.append(target)
            except:
                print(image_path, 'not processed')
                
        print('Processed:', target)
        data = pd.DataFrame(feature_targets,columns=['class'])
        data['text'] = feature_texts
        data.to_parquet('image2text/'+target+'_data.parquet')


def create_txt_parquet_files():

    for target in ['0','2','4','6','9']:

        dir_path = os.path.join(r'data/ocr', target)
        feature_targets, feature_texts = [], []

        text_files = os.listdir(dir_path)
        for text_file in tqdm(text_files):
            text_path = os.path.join(dir_path, text_file)
            try:
                text = read_text(text_path)
                feature_texts.append(text)
                feature_targets.append(target)
            except:
                print(text_path, 'not processed')

        print('Processed:', target)
        data = pd.DataFrame(feature_targets,columns=['class'])
        data['text'] = feature_texts
        data.to_parquet('read_text/'+target+'_data.parquet')


def data_split(dataframe):

    train_indices, val_indices = train_test_split(
                                                dataframe,
                                                test_size=0.2,
                                                stratify=dataframe['class']
                                                )
    train_indices, test_indices = train_test_split(
                                                train_indices,
                                                test_size=0.25,
                                                stratify=train_indices['class']
                                                )

    return train_indices, test_indices, val_indices 


def data_loader(dataset_type = 'txt'):
    
    if dataset_type == 'txt':

        if len(os.listdir('read_text'))==0:
            create_txt_parquet_files()

        dataset = pd.DataFrame(columns = ['class', 'text'])

        for target in ['0','2','4','6','9']:
            chunk = pd.read_parquet('read_text/'+target+'_data.parquet')
            dataset = pd.concat([dataset, chunk])

    elif dataset_type == 'ocr':

        if len(os.listdir('image2text'))==0:
            create_ocr_parquet_files()
        
        dataset = pd.DataFrame(columns = ['class', 'text'])

        for target in ['0','2','4','6','9']:
            chunk = pd.read_parquet('image2text/'+target+'_data.parquet')
            dataset = pd.concat([dataset, chunk])

    else:

        raise Exception("Format data not deined ('ocr' or 'txt')")

    train_set, test_set, val_set = data_split(dataset)

    return train_set, test_set, val_set


if __name__=="__main__":
    
    # create_ocr_parquet_files()

    # create_txt_parquet_files()

    tr, ts, va = data_loader()
