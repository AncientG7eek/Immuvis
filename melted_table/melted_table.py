import os
import pandas as pd
import re
import ast
import sys

def is_float_convertible(var):
    try:
        float(var)
        return True
    except (ValueError, TypeError):
        return False
def is_int_convertible(var):
    try:
        int(var)
        return True
    except (ValueError, TypeError):
        return False

IMG_PATHS = "img_data/full_images_list_20122025_mapping_to_full_dir.csv" # stores paths to images and img shapes in each dataset
TABLES_DIR = "clinical_data"
DATASETS_SUBTYPES = ['jackson-zurich', 'jackson-basel', 'damond-lung'] # some datasets like jackson consist of subtypes with separate clinical files. Each of subtypes is treated as a seperate dataset by the model  
CONFIG_PATH = 'master_clinical.tsv' # contains mapping from img filename to row in clinical data table
OUTPUT_PATH = "results/melted_table.csv"

info_columns = ["dataset", "img_name", "image_shape", "patient", "merge"]
feature_columns = ['feature','value']
all_columns = info_columns[:]
all_columns.extend(feature_columns)
print(all_columns)
print(info_columns)
pd.DataFrame([all_columns]).to_csv(OUTPUT_PATH, index=False, header=0)

config = pd.read_csv(CONFIG_PATH, sep='\t', dtype='str', index_col=0)

################################### initialize files

# melted_list_img_header = [['dataset', 'image_path', 'image_shape', 'feature', 'feature_value']]
# melted_list_var_type_header = [['dataset','feature','variable_type']]

# melted_images_csv = "melted_table_images3.csv"
# melted_var_type_csv = "melted_table_var_types3.csv"

# if os.path.exists(melted_images_csv):
#     os.remove(melted_images_csv)
# if os.path.exists(melted_var_type_csv):
#     os.remove(melted_var_type_csv)

# melted_table = pd.DataFrame(melted_list_img_header)
# melted_table_var_type = pd.DataFrame(melted_list_var_type_header)
# melted_table.to_csv(melted_images_csv, index=False, header=0)
# melted_table_var_type.to_csv(melted_var_type_csv, index=False, header=0)

################################### Loop over clinical data csvs

for file in os.listdir(TABLES_DIR):
    print(file)
    paths = pd.read_csv(IMG_PATHS, dtype=str)
    dataset = file.split('.')[0]
    
    file = os.path.join(TABLES_DIR,file)
    conf = config[config['dataset']==dataset]

    data = pd.read_csv(file, dtype=str)
    
    #num_factors_per_feature = [data[col].nunique() for col in data.columns] # for automated variable type assesment


    if dataset in DATASETS_SUBTYPES:
        # usually the naming is: subtype = 'full_dataset-something'; hoch is an exception (hoch-rna is a full_dataset name), hence need for DATASETS_SUBTYPES and this if statement
        paths = paths[paths['dataset_subtype']==dataset]
    elif dataset in ['immucan-p1','immucan-p2']:
        paths = paths[(paths['dataset_subtype'] == 'nsclc2') & (paths['dataset'] == dataset)]
        dataset = "nsclc2-panel" + dataset[-1]
    else:
        paths = paths[paths['dataset']==dataset] # get only desired dataset's rows from the master mapping file

    old_path = paths['tiff_path'].astype(str) # meaningful image naming to map clinical values on
    new_path = paths['new_path'].astype(str) # image naming used in split, present in embedding file; master mapping file maps old_path to it
    img_shape = paths['shape']
    
    new_names = [x.split("/")[-1].split('.')[0] for x in new_path] # get image names from image path in form they're written in embeddings file
    path_mapping = dict(zip(new_names,old_path)) # this dict will allow for mapping each embedding in embeddings file to old_path (containing IDs corresponding to clinical features)
    old_2_new = {v:k for k,v in path_mapping.items()}
    shape_mapping = dict(zip(new_names,img_shape))

    image_name_pattern = conf['image_name_pattern'].iloc[0] # for extracting IDs from image old_path
    transform_rules = ast.literal_eval(conf['transform_rules'].iloc[0]) # for conversion of old_path IDs notation to clinical file values notation
    
    all_imgs_IDs = [] # [dict]
    
    ID_column_names = ast.literal_eval(conf['ID_column_names'].iloc[0]) # names of columns in clinical file storing IDs

    for img_path in old_path:
        # gets all IDs specified in config and converts to notation used in clinical file
        IDs = re.findall(image_name_pattern, img_path)
        if isinstance(IDs[0],tuple):
            IDs = [id for id in IDs[0]]
        
        transformed_IDs = [id.replace(before,after) for id, (before,after) in zip(IDs, transform_rules)]
        #all_imgs_IDs[img_path] = transformed_IDs

        row = dict(zip(ID_column_names, transformed_IDs))
        row["img_name"] = old_2_new[img_path]
        
        all_imgs_IDs.append(row)
    
    img_df = pd.DataFrame(all_imgs_IDs).astype(str)
    img_df["dataset"] = dataset
    img_df["image_shape"] = img_df["img_name"].map(shape_mapping)

    for col in ID_column_names:
        img_df[col] = img_df[col].astype(str)
        data[col] = data[col].astype(str)
    
    merged = img_df.merge(data, on=ID_column_names, how="inner", indicator="merge")
    # print(conf)
    # print(conf['patient_ID_column_name'])
    merged['patient'] = merged[conf['patient_ID_column_name']]
   
    print(merged["merge"].value_counts())
    
    
    ################################ this melted table tells about type of feature (continuous, cathegorical, ordinal)

    # melted_list_var_type = []
        
    # for feature in data.columns:

    #     if is_float_convertible(feature):
    #         var_type = 'continous'
    #     elif is_int_convertible(feature):
    #         if len(num_factors_per_feature[idx]) >= 3:
    #             var_type = 'ordinal'
    #     else:
    #         var_type = 'categorical'

    #     new_row = [dataset, feature, var_type]
    #     melted_list_var_type.append(new_row)

    ################################ this table contains separate row for each dataset,patient,feature

    # melted_list = []
    # not_found = []
    # # print(len(data))
    # # print(all_imgs_IDs)
    # for img,IDs in all_imgs_IDs.items():
    # # for each old_path iterates over rows in clinical file to find one which matches all IDs to append the match to a {old_path : pd.DataFrame([universal names of clinical features instances for a given image])} dict
    #     matched_row = None # placeholder for a matched row
    #     new_name = old_2_new[img]
    #     shape = shape_mapping[new_name]
        
    #     for idx, row in data.iterrows(): 
    #         #print(ID_column_names,IDs)
    #         #print([(row[column], id) for column, id in zip(ID_column_names,IDs)])
    #         # finding image in clinical features file
    #         if all(str(row[column]).replace('.tiff','')==str(id) for column,id in zip(ID_column_names,IDs)):
    #             matched_row = row
                
                
    #             for idx,value in enumerate(matched_row):
                    
    #                 new_row = [dataset, str(new_name)+'.tiff', shape, data.columns[idx], value]

    #                 melted_list.append(new_row)
            
    #             break # because match has been found
    #     if matched_row is None:
    #         # handles a case when a match wasn't found by assigning nan for each feature for the image
    #         not_found.append(img)

    melted = merged.melt(
        id_vars=info_columns,
        value_vars=data.columns,
        var_name="feature",
        value_name="value"
    )

    melted = melted.sort_values(["dataset", "img_name"])
            
    ############################# append dataset to melted tables
                 
    # melted_table = pd.DataFrame(melted_list)
    # melted_table_var_type = pd.DataFrame(melted_list_var_type)
    # melted_table.to_csv(melted_images_csv, mode='a', index=False, header=False)
    # melted_table_var_type.to_csv(melted_var_type_csv, mode='a', index=False, header=False)
    # print(f"Appended {dataset} to melted tables. {len(not_found)} images were not matched to any row in clinical data file:")
    # print(not_found)
    melted['patient'] = melted['patient'].astype(str)
    melted.to_csv(OUTPUT_PATH, mode='a', index=False, header=False)
