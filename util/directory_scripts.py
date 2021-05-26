import os 
import shutil
import argparse
from scipy.sparse import data
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--path',type=str)
parser.add_argument('--split',type=float,default=0.2)
parser.add_argument('--train_val_split', default=False, action='store_true')
parser.add_argument('--rename_cp_to_model', default=False, action='store_true')

def train_val_split(path_to_data,split):
    '''A function that splits a directory into a train and a val directory
    Args:
        path_to_data(str): the path to the data to be split
        split(float): the ratio of val data of whole data, setting to 
            0 leads to no validation data'''
    
    # set paths and make directories
    train_path = os.path.join(path_to_data,'train')
    val_path = os.path.join(path_to_data,'val')
    os.makedirs(train_path)
    os.makedirs(val_path)

    data_paths = os.listdir(path_to_data)
    train_data,val_data=train_test_split(data_paths,test_size=split)

    print(train_data)
    for data_path in train_data:
        source_path = os.path.join(path_to_data,data_path)
        dest_path = os.path.join(train_path,data_path)
        shutil.copy(source_path,dest_path)

    for data_path in val_data:
        source_path = os.path.join(path_to_data,data_path)
        dest_path = os.path.join(val_path,data_path)
        shutil.copy(source_path,dest_path)

def rename_cp_to_model(path_to_data):
    '''goes through the directory and renames checkpoint.th into model.th'''
    for path in os.listdir(path_to_data):
        path_to_model = os.path.join(path_to_data,path,'model.th')
        path_to_cp = os.path.join(path_to_data,path,'checkpoint.th')
        if not os.path.exists(path_to_model):
            print(path_to_cp)
            os.rename(path_to_cp,path_to_model)
            
def main():
    global args
    args = parser.parse_args()
    if args.train_val_split:
        train_val_split(args.path,args.split)
    if args.rename_cp_to_model:
        rename_cp_to_model(args.path)

if __name__ == '__main__':
    main()
