import pandas as pd
import numpy as np
import os
import tensorflow as tf

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''

#     newcol=[]
#     for i in range(len(df)):
#         code=df['ndc_code'].iloc[i]
#         ind=ndc_df.index[ndc_df['NDC_Code']==code].tolist()

#         if not ind:
#             newcol.append('nan')
#         else:      
#             newcol.append(ndc_df["Proprietary Name"].iloc[ind[0]]) #df['generic_drug_name'].iloc[i]
        
#     df['generic_drug_name']=newcol
    mapping = dict(ndc_df[['NDC_Code', 'Non-proprietary Name']].values)
    mapping['nan'] = np.nan
    df['generic_drug_name'] = df['ndc_code'].astype(str).apply(lambda x : mapping[x])
    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''

    df = df.sort_values('encounter_id')
    first_encounter_values = df.groupby("patient_nbr")["encounter_id"].first().values
    df=df[df["encounter_id"].isin(first_encounter_values)].reset_index()
    df=df.drop("index",1)
    first_encounter_df = df.groupby((df["encounter_id"] != df["encounter_id"].shift()).cumsum().values).first()
    
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, student_numerical_col_list, test_percentage=0.2):#patient_key='patient_nbr', test_percentage=0.2):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
#     df = df.iloc[np.random.permutation(len(df))]
#     unique_values = df[patient_key].unique()
#     total_values = len(unique_values)
#     sample_size = round(total_values * (1 - test_percentage ))
#     trainval = df[df[patient_key].isin(unique_values[:sample_size])].reset_index(drop=True)
#     test = df[df[patient_key].isin(unique_values[sample_size:])].reset_index(drop=True)
#     trainvaltot=len(trainval[patient_key].unique())
#     newsample=round(trainvaltot * (1 - test_percentage ))
#     train=trainval[trainval[patient_key].isin(unique_values[:newsample])].reset_index(drop=True)
#     validation=trainval[trainval[patient_key].isin(unique_values[newsample:])].reset_index(drop=True)
#     from sklearn.model_selection import train_test_split
#     trainval, test = train_test_split(
#         df, 
#         test_size=test_percentage,
#         random_state=42,
#         shuffle=True)#,
#         #stratify=df[patient_key])

#     train, validation = train_test_split(
#         trainval, 
#         test_size=test_percentage,
#         random_state=42,
#         shuffle=True)#,
#         #stratify=trainval[patient_key]) #['time_in_hospital']

    df[student_numerical_col_list] = df[student_numerical_col_list].astype(float)
    train_val_df = df.sample(frac = 0.8, random_state=3)
    train_df = train_val_df.sample(frac = 0.8, random_state=3)
    val_df = train_val_df.drop(train_df.index)
    test_df = df.drop(train_val_df.index)
    return train_df.reset_index(drop = True), val_df.reset_index(drop = True), test_df.reset_index(drop = True)
#     return train, validation, test


#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        principal_diagnosis_vocab = tf.feature_column.categorical_column_with_vocabulary_file(c, 
            vocab_file_path, num_oov_buckets=1)
        
        tf_categorical_feature_column = tf.feature_column.indicator_column(principal_diagnosis_vocab)
        
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    import functools
  
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature=tf.feature_column.numeric_column(
    key=col, default_value = default_value, normalizer_fn=normalizer, dtype=tf.float64)

    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    student_binary_prediction = df[col].apply(lambda x : 1 if x >=6 else 0)

    return student_binary_prediction
