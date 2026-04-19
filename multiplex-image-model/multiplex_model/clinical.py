import os
import pandas as pd
import numpy as np

def concat_metadata(emb_dir: str, train: bool):
    """ 
    Args:
        emb_dir (str):  path to directory with embeddings and metadata
        train (bool):   if True, train split metadata files are chosen
                        else test split metadata are chosen
    Returns: 
        full_emb_metadata (pd.DataFrame):   all chosen metadata files concatenated
    """
    parts = []
    for file in os.listdir(emb_dir):
        file = os.path.join(emb_dir,file)
        if file.endswith(".csv"):
            if train==True and 'test' in file:
                continue
            elif train==False and 'train' in file:
                continue
            part = pd.read_csv(file)
            embedding_file = file.replace('metadata','embeddings').replace('csv','npy')
            if not os.path.exists(embedding_file):
                    continue
            part["embeddings_file"] = embedding_file
            part["embedding_idx"] = part.index
            parts.append(part)
    full_emb_metadata = pd.concat(parts)
    return full_emb_metadata

def concat_virtues_metadata(emb_dir: str, train: bool):
    """ 
    Args:
        emb_dir (str):  path to directory with embeddings and metadata
        train (bool):   if True, train split metadata files are chosen
                        else test split metadata are chosen
    Returns: 
        full_emb_metadata (pd.DataFrame):   all chosen metadata files concatenated
    """
    datasets = []
    for dataset_dir in os.listdir(emb_dir):
        dataset_dir = os.path.join(emb_dir,dataset_dir)
        for file in os.listdir(dataset_dir):
            file = os.path.join(dataset_dir,file)
            if file.endswith(".csv"):
                if train==True and 'test' in file:
                    continue
                elif train==False and 'train' in file:
                    continue
                dataset = pd.read_csv(file)
                embedding_file = file.replace('metadata','embeddings').replace('csv','npy')
                if not os.path.exists(embedding_file):
                    continue
                dataset["embeddings_file"] = embedding_file
                dataset["embedding_idx"] = dataset.index
                datasets.append(dataset)
    full_emb_metadata = pd.concat(datasets)
    return full_emb_metadata

def merge_metadata_with_melted(emb_metadata: pd.DataFrame, melted_table: pd.DataFrame, meta_table_path: str):
    """ 
    Args:
        emb_metadata (pd.DataFrame):  dataframe with embeddings metadata
        melted_table (pd.DataFrame):  melted table with single feature per single image as row
        meta_table_path (str):        path to destination file for the result
    Returns: 
        merged_table (pd.DataFrame):  melted table with single feature per single embedding as row 
                                    + path to file with embeddings and idx of embedding in that file
                                    (meta table)
    """
    emb_metadata = emb_metadata.copy()
    img_col = "img_path" if "img_path" in emb_metadata.columns else "image_paths" # col with img path in emb_metadata file
    emb_metadata["img_path"] = emb_metadata[img_col].apply(lambda x: x.split("/")[-1])
    merged_table = pd.merge(emb_metadata, melted_table, left_on="img_path", right_on='image_path')
    merged_table = merged_table.drop(columns=['panel','image_path'], errors='ignore')
    os.makedirs(os.path.dirname(meta_table_path), exist_ok=True)
    merged_table.to_csv(meta_table_path)
    return merged_table

def get_a_subset(meta_table: pd.DataFrame, column: str, value: str|list):
    if isinstance(value, str):
        return meta_table[meta_table[column].astype(str)==value]
    if isinstance(value, list):
        return meta_table[meta_table[column].astype(str).isin(value)]
        

class LabelEncoder():
    def __init__(self, classes):
        self.dict = {cls:i for i,cls in enumerate(sorted(set(classes)))}
        self.inv_dict = {i:cls for cls,i in self.dict.items()}

    def encode(self, labels):
        return [self.dict[label] for label in labels]
    
    def decode(self, labels):
        return [self.inv_dict[label] for label in labels]
    
    def get_dict(self):
        return self.dict
