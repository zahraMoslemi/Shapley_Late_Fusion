# Utility functions for data cleaning
# Yueqi Ren, 2023-10-17

#############################################

import pandas as pd 
import numpy as np 
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.model_selection import train_test_split

#############################################

def get_feature_types(df,tier2=True):
    """ Obtain lists of features that fall into different data types.
        Args:
            df: pandas dataframe
            tier2: boolean, whether to include tier2 features (default is True),
                can leave as is without change
    """
    # one hot encode, without dropping categories
    cat_demo_feats = ['NACCNIHR','PRIMLANG','SEX','HISPANIC']    # demographic
    ord_demo_feats = ['MARISTAT','NACCLIVS','INDEPEND','RESIDENC']   # living situation
    num_demo_feats = ['NACCAGE','EDUC']   # demographic

    # health history
    cat_phist_feats = ['TOBAC30', 'TOBAC100','NACCTBI','DEP2YRS', 'DEPOTHR',
            'ANYMEDS','NACCAAAS', 'NACCAANX', 'NACCAC', 'NACCACEI', 
            'NACCADEP', 'NACCAHTN', 'NACCANGI', 'NACCAPSY',
            'NACCBETA', 'NACCCCBS', 'NACCDBMD', 'NACCDIUR', 'NACCEMD',
            'NACCEPMD', 'NACCHTNC', 'NACCLIPL', 'NACCNSD', 'NACCPDMD',
            'NACCVASD']
    ord_phist_feats = ['NACCAMD','PACKSPER','CVHATT', 'CVAFIB', 'CVANGIO', 'CVBYPASS', 'CBTIA',
            'CVPACE', 'CVCHF', 'CVOTHR', 'CBSTROKE','SEIZURES','NCOTHR',
            'DIABETES','HYPERTEN', 'HYPERCHO', 'B12DEF','THYROID', 'INCONTU', 
            'INCONTF','ALCOHOL', 'ABUSOTHR','PSYCDIS']
    num_phist_feats = ['SMOKYRS','NACCSTYR','NACCTIYR']

    cat_fhist_feats = ['NACCFADM', 'NACCFFTD']
    ord_fhist_feats = ['NACCFAM', 'NACCMOM', 'NACCDAD']  # family history

    cat_phys_feats = ['NACCNREX','FOCLSYM','FOCLSIGN']
    ord_phys_feats = ['DECSUB','VISION', 'VISCORR','VISWCORR','HEARING', 'HEARAID', 'HEARWAID'] # 'APOERISK'
    num_phys_feats = ['HEIGHT', 'WEIGHT','BPSYS', 'BPDIAS', 'HRATE','NACCBMI']

    ord_gds_feats = ['NOGDS', # -4 = NaN - binary
            'SATIS', 'DROPACT', 'EMPTY', 'BORED', 'SPIRITS', 'AFRAID',
            'HAPPY', 'HELPLESS', 'STAYHOME', 'MEMPROB', 'WONDRFUL', 'WRTHLESS',
            'ENERGY', 'HOPELESS', 'BETTER', # recode 9 as -1, -4 = NaN - ordinal
            'NACCGDS'] # 88 or -4 = NaN - ordinal
    ord_faq_feats = ['BILLS', 'TAXES','SHOPPING', 'GAMES', 'STOVE', 
            'MEALPREP', 'EVENTS', 'PAYATTN','REMDATES', 'TRAVEL'] # recode 8 as -1, 9 or -4 = NaN
    # Decide to drop the binary variables in NPI
    ord_npi_feats = ['DELSEV', 'HALLSEV', 'AGITSEV', 'DEPDSEV', 'ANXSEV',
                'ELATSEV', 'APASEV', 'DISNSEV', 'IRRSEV', 'MOTSEV', 'NITESEV',
                'APPSEV',]

    ord_np_feats = ['MMSEORDA','MMSEORLO']

    label_feat = ['NACCUDSD'] # for prediction tasks (never mising, ordinal)

    # CDR features are never missing and are all ordinal
    cdr_feats = ['MEMORY', 'ORIENT', 'JUDGMENT', 'COMMUN', 'HOMEHOBB', 
                        'PERSCARE', 'CDRSUM', 'CDRGLOB'] 

    # combined
    cat_feats = cat_demo_feats + cat_phist_feats + cat_fhist_feats + cat_phys_feats
    # neuropschological testing
    if tier2:
        num_np_feats = ['NACCMMSE','MEMUNITS','DIGIF', 'DIGIFLEN', 'DIGIB', 'DIGIBLEN',
                    'ANIMALS', 'VEG','BOSTON', 'TRAILA', 'TRAILB']
    else:
        num_np_feats = ['NACCMMSE']
    ord_feats = list(ord_demo_feats + ord_phist_feats + ord_fhist_feats + ord_phys_feats + 
                    ord_npi_feats + ord_gds_feats + ord_faq_feats + ord_np_feats)
    num_feats = num_demo_feats + num_phist_feats + num_phys_feats + num_np_feats

    # check that the features are in the dataframe
    cat_feats = [x for x in cat_feats if x in df.columns.values]
    ord_feats = [x for x in ord_feats if x in df.columns.values]
    num_feats = [x for x in num_feats if x in df.columns.values]
    label_feat = [x for x in label_feat if x in df.columns.values]
    cdr_feats = [x for x in cdr_feats if x in df.columns.values]

    miss_cat_feats = [x for x in cat_feats if x in df.columns.values]
    miss_ord_feats = [x for x in ord_feats if x in df.columns.values]
    miss_num_feats = [x for x in num_feats if x in df.columns.values]
    miss_label_feat = [x for x in label_feat if x in df.columns.values]
    miss_cdr_feats = [x for x in cdr_feats if x in df.columns.values]
    
    # returns: categorical features, ordinal features, numerical features, label feature, cdr features
    return cat_feats, ord_feats, num_feats, label_feat, cdr_feats


def get_modality_features(df,tier2=True, fine_grained=False):
    """ Obtain lists of features that fall into different modalities.
    This function divide the UDS data into two modalities: 
    Modality 1: history-based information including basic demographics, health history, family history etc. 
        Args:
            df: pandas dataframe
            tier2: boolean, whether to include tier2 features (default is True),
                can leave as is without change
    """
    # one hot encode, without dropping categories
    cat_demo_feats = ['NACCNIHR','PRIMLANG','SEX','HISPANIC']    # demographic
    ord_demo_feats = ['MARISTAT','NACCLIVS','INDEPEND','RESIDENC']   # living situation
    num_demo_feats = ['NACCAGE','EDUC']   # demographic

    # health history
    cat_phist_feats = ['TOBAC30', 'TOBAC100','NACCTBI','DEP2YRS', 'DEPOTHR',
            'ANYMEDS','NACCAAAS', 'NACCAANX', 'NACCAC', 'NACCACEI', 
            'NACCADEP', 'NACCAHTN', 'NACCANGI', 'NACCAPSY',
            'NACCBETA', 'NACCCCBS', 'NACCDBMD', 'NACCDIUR', 'NACCEMD',
            'NACCEPMD', 'NACCHTNC', 'NACCLIPL', 'NACCNSD', 'NACCPDMD',
            'NACCVASD']
    ord_phist_feats = ['NACCAMD','PACKSPER','CVHATT', 'CVAFIB', 'CVANGIO', 'CVBYPASS', 'CBTIA',
            'CVPACE', 'CVCHF', 'CVOTHR', 'CBSTROKE','SEIZURES','NCOTHR',
            'DIABETES','HYPERTEN', 'HYPERCHO', 'B12DEF','THYROID', 'INCONTU', 
            'INCONTF','ALCOHOL', 'ABUSOTHR','PSYCDIS']
    num_phist_feats = ['SMOKYRS','NACCSTYR','NACCTIYR']

    cat_fhist_feats = ['NACCFADM', 'NACCFFTD']
    ord_fhist_feats = ['NACCFAM', 'NACCMOM', 'NACCDAD']  # family history

    cat_phys_feats = ['NACCNREX','FOCLSYM','FOCLSIGN']
    ord_phys_feats = ['DECSUB','VISION', 'VISCORR','VISWCORR','HEARING', 'HEARAID', 'HEARWAID'] # 'APOERISK'
    num_phys_feats = ['HEIGHT', 'WEIGHT','BPSYS', 'BPDIAS', 'HRATE','NACCBMI']

    # neuropsychiatric & behavioral
    ord_gds_feats = ['NOGDS', # -4 = NaN - binary
            'SATIS', 'DROPACT', 'EMPTY', 'BORED', 'SPIRITS', 'AFRAID',
            'HAPPY', 'HELPLESS', 'STAYHOME', 'MEMPROB', 'WONDRFUL', 'WRTHLESS',
            'ENERGY', 'HOPELESS', 'BETTER', # recode 9 as -1, -4 = NaN - ordinal
            'NACCGDS'] # 88 or -4 = NaN - ordinal
    ord_faq_feats = ['BILLS', 'TAXES','SHOPPING', 'GAMES', 'STOVE', 
            'MEALPREP', 'EVENTS', 'PAYATTN','REMDATES', 'TRAVEL'] # recode 8 as -1, 9 or -4 = NaN
    # Decide to drop the binary variables in NPI
    ord_npi_feats = ['DELSEV', 'HALLSEV', 'AGITSEV', 'DEPDSEV', 'ANXSEV',
                'ELATSEV', 'APASEV', 'DISNSEV', 'IRRSEV', 'MOTSEV', 'NITESEV',
                'APPSEV',]

    # neuropschological testing
    ord_np_feats = ['MMSEORDA','MMSEORLO']

    if tier2:
        num_np_feats = ['NACCMMSE','MEMUNITS','DIGIF', 'DIGIFLEN', 'DIGIB', 'DIGIBLEN',
                    'ANIMALS', 'VEG','BOSTON', 'TRAILA', 'TRAILB']
    else:
        num_np_feats = ['NACCMMSE']

    label_feat = ['NACCUDSD'] # for prediction tasks (never mising, ordinal)

    # CDR features are never missing and are all ordinal
    cdr_feats = ['MEMORY', 'ORIENT', 'JUDGMENT', 'COMMUN', 'HOMEHOBB', 
                        'PERSCARE', 'CDRSUM', 'CDRGLOB'] 

    # Modality 1 comtains all history related information
    demo_feats = cat_demo_feats + ord_demo_feats + num_demo_feats
    phist_feats = cat_phist_feats + ord_phist_feats + num_phist_feats
    fhist_feats = cat_fhist_feats + ord_fhist_feats

    mod_history = demo_feats + phist_feats + fhist_feats

    # Modality 2 contains all physical or psychological testings/surveys
    phys_feats = cat_phys_feats + ord_phys_feats + num_phys_feats
    npi_feats = ord_gds_feats + ord_faq_feats + ord_npi_feats
    npt_feats = ord_np_feats + num_np_feats

    # If fine grained, further split into survey modality and clinical testing modality
    if fine_grained:
        mod_survey = npi_feats
        mod_testing = phys_feats + npt_feats
        return  mod_history, mod_survey, mod_testing, label_feat
    else:
        mod_survey = phys_feats + npi_feats + npt_feats
        return mod_history, mod_survey, label_feat

    
 
def encode_feature_types(X_data,tier2=True):
    """ Encode features by data type for later processing.
        Args: 
            X_data: pandas dataframe, n x p
            tier2: boolean, whether to include tier2 features (default is True),
                can leave as is without change
        Notes:
            All the imputation steps are not used here but gives you an idea 
            of how you can incorporate it in the same set up later on
            (imputation is done in the pipeline AFTER train/test split).
            If you have CDR or diagnosis features, uncomment the final part of the function.
    """
    # Use specific transformers for each type of data
    cat_feats, ord_feats, num_feats, label_feat, cdr_feats = get_feature_types(X_data,tier2)

    # Obtain the ordinal-encoded features to get the names of the columns
    ordinal_transformer = Pipeline( 
       steps=[
           #('imputer', SimpleImputer(strategy='most_frequent')),
           ('encoder', OrdinalEncoder()), 
       ]
    )
    ordinal_preprocessor = ColumnTransformer(
        transformers=[
            ('ord', ordinal_transformer, ord_feats),
            ('cat', 'passthrough', cat_feats),
            ('num', 'passthrough', num_feats),
            # If your data frame includes other features, uncomment the line below
            ('other', 'passthrough', label_feat + cdr_feats) 
        ],
        verbose_feature_names_out=False
    )
    ordinal_preprocessor = ordinal_preprocessor.fit(X_data)

    return ordinal_preprocessor


def split_data(data,test_percent=0.2,random_seed=None):
    """ Split the data into training and testing sets.
        Args:
            data: pandas dataframe, n x p of all features (including the clinical diagnosis label)
            test_percent: float, percentage of data to use for testing
            random_seed: int or None (default), to ensure reproducibility for splits if given a specific seed
    """
    X_data = data.copy()
    X_subj = np.unique(X_data.index.values)
    subj_train, subj_test = train_test_split(X_subj, test_size=test_percent, random_state=random_seed)
    X_train = X_data.loc[subj_train]
    X_test = X_data.loc[subj_test]
    return X_train, X_test


def match_invariant_feature_names(transformed_features):
    """ Due to one-hot-encoding, make sure to match feature names
        Args:
            transformed_features: list of feature names after encoding
    """
    # Find which features are time invariant
    #feature_details = pd.read_excel('Project Information/feature_details.xlsx',index_col=0,header=0,engine='openpyxl')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'feature_details.xlsx')
    feature_details = pd.read_excel(file_path,index_col=0,header=0)
    invariant_feats = feature_details[feature_details['time-varying']==0].index.values
    transformed_invariant_feats = []
    for feat in transformed_features:
        if feat.split('_')[0] in invariant_feats:
            transformed_invariant_feats.append(feat)
    return transformed_invariant_feats


def impute_invariant_features(X_train, X_test):
    """ Impute all invariant features using training data.
        Args:
            X_train: pandas dataframe, n x p of training data
            X_test: pandas dataframe, n x p of testing data
        Notes:
            You may need to change the file directory for the feature_details file.
    """
    # Find which features are time invariant
    transformed_features = X_train.columns.values
    invariant_feats = match_invariant_feature_names(transformed_features)
    # Impute invariant features
    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()
    for feat in invariant_feats:
        # filling in using the mode ensures that all missing instances receive the same value
        # feel free to edit this step to use other types of imputation methods
        # while ensuring that all values of the feature are the same per subject across visits
        X_train_imputed[feat] = X_train_imputed[feat].fillna(X_train[feat].mode()[0])
        X_test_imputed[feat] = X_test_imputed[feat].fillna(X_train[feat].mode()[0])
    return X_train_imputed, X_test_imputed


def imputer_by_feature_type(X_data,tier2=True):
    """ Impute features by data type using training data.
        Args: 
            X_data: pandas dataframe, n x p of training data
            tier2: boolean, whether to include tier2 features (default is True),
                can leave as is without change
        Notes:
            All the training data should have already be encoded by data type prior to this step.
            If you have CDR or diagnosis features, uncomment the final part of the function.
    """
    # Use specific transformers for each type of data
    cat_feats, ord_feats, num_feats, label_feat, cdr_feats = get_feature_types(X_data,tier2)

    numeric_transformer = Pipeline(
        steps=[
            ('scaler', StandardScaler()), # z transform
            ('imputer', SimpleImputer(strategy='median')), 
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore',drop='first')),
        ]
    )
    ordinal_transformer = Pipeline( 
       steps=[
           ('imputer', SimpleImputer(strategy='most_frequent')),
           #('encoder', OrdinalEncoder()), 
       ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ('ord', ordinal_transformer, ord_feats),
            ('cat', categorical_transformer, cat_feats),
            ('num', numeric_transformer, num_feats),
            # If your data frame includes other features, uncomment the line below
            ('other', 'passthrough', label_feat + cdr_feats) 
        ],
        verbose_feature_names_out=False
    )
    return preprocessor



def impute_mrisbm_features(X_train, X_test):
    # Compute the mode for each column
    train_modes = X_train.mode().iloc[0]  # `.iloc[0]` extracts the mode for each column
    test_modes = X_test.mode().iloc[0]
    # Fill missing values with the mode

    X_train_imputed = X_train.fillna(train_modes)
    X_test_imputed = X_test.fillna(test_modes)

    # Apply z-transform (standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Convert the numpy arrays back to pandas DataFrames, keeping the original column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled



#############################################

if __name__ == '__main__':
    # Perform the data cleaning steps as outlined in the Data Cleaning document

    # Load the data
    training_data = pd.read_csv('TrainTestData/seven_visits_train_new.csv',index_col=0,header=0)
    testing_data = pd.read_csv('TrainTestData/seven_visits_test_new.csv',index_col=0,header=0)
    # Merge the training and testing data for data cleaning
    data = pd.concat([training_data,testing_data])
    # Select the features to use
    cat_feats, ord_feats, num_feats, label_feat, cdr_feats = get_feature_types(data)
    data = data[cat_feats+ord_feats+num_feats+label_feat+cdr_feats]
    # Encode the ordinal data type (prior to split to ensure all categories are included)
    ordinal_preprocessor = encode_feature_types(data)
    encoded_data = pd.DataFrame(ordinal_preprocessor.transform(data),columns=ordinal_preprocessor.get_feature_names_out(),index=data.index.values)
    # Split the data into training and testing sets
    random_seed = np.random.randint(10000) # random seed and save name for each split
    X_train, X_test = split_data(encoded_data,random_seed=random_seed)
    # Impute the invariant features
    X_train_imputed, X_test_imputed = impute_invariant_features(X_train, X_test)
    # Impute the remaining features
    preprocessor = imputer_by_feature_type(X_train_imputed)
    preprocessor = preprocessor.fit(X_train_imputed)
    X_train_imputed = pd.DataFrame(preprocessor.transform(X_train_imputed), columns=preprocessor.get_feature_names_out(), index=X_train_imputed.index.values)
    X_test_imputed = pd.DataFrame(preprocessor.transform(X_test_imputed), columns=preprocessor.get_feature_names_out(), index=X_test_imputed.index.values)
    # Save the data
    X_train_imputed.to_csv('TrainTestData/seven_visits_train_cleaned_'+str(random_seed)+'.csv')
    X_test_imputed.to_csv('TrainTestData/seven_visits_test_cleaned_'+str(random_seed)+'.csv')