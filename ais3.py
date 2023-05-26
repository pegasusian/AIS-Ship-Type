import datetime

import numpy as np # Linear Algebra
import pandas as pd # data processing


# importing the important Libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.metrics import classification_report

print('All the Libraries have been imported !!')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



print(" All the models Imported ")

# Slight Adjustments to the screen settings

from warnings import filterwarnings
filterwarnings('ignore')
pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
pd.set_option('display.float_format',lambda x: f'{x:.4f}')

# Loading the data


# lets do the Exploratory data analysis
# cat_th is short for "categorical threshold"
# car_th is short for "cardinality threshold"

def thresholds(col, data, d, u):
    q3 = data[col].quantile(u)
    q1 = data[col].quantile(d)
    down = q1 - (q3 - q1) * 1.5
    up = q1 + (q3 - q1) * 1.5
    return down, up


def check_outliers(col, data, d=0.25, u=0.75, plot=True):
    down, up = thresholds(col, data, d, u)
    ind = data[(data[col] < down) | (data[col] > up)].index
    if plot:
        sns.boxplot(x=col, data=data)
        plt.show()
    if len(ind) != 0:
        print(f"\n Number of outliers for {col} : {len(ind)}")
        return col

def corr_analyzer(data, corr_th=0.7, plot = False):
    corr_matrix = pd.DataFrame(np.tril(data.corr(), k=-1), columns=data.corr().columns, index= data.corr().columns)
    corr = corr_matrix[corr_matrix > corr_th].stack()
    print(corr)

    if plot:
        sns.heatmap(corr_matrix, cmap='Blues')
    return corr[corr>corr_th].index.to_list()

def missing_values_table(data):
    m = data.isnull().sum()
    print(pd.DataFrame({'n_miss' : m[m!=0], 'ratio' : m[m!=0]/len(data)}))

def grab_col_names(dataframe, cat_th = 10, car_th=20):
    #cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes =="O"] #Object data type
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes !="O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in  cat_but_car]

    #num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations : {dataframe.shape[0]}")
    print(f"Variables :{dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

def cat_summary(dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(), "Ratio" : 100*dataframe[col_name].value_counts() / len(dataframe)}))

    print(" ")
    print(" _____________________________________________________ ")
    print(" ")

    if plot:
        sns.countplot(x=dataframe[col_name], data = dataframe)
        plt.show()

    def rare_encoder(dataframe, rare_perc):
        temp_df = dataframe.copy()
        rare_columns = [col for col in temp_df.columns if
                    temp_df[col].dtypes == 'O' and col != 'shiptype' and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(
                        axis=None)]

        for var in rare_columns:
            tmp = temp_df[var].value_counts() / len(temp_df)
            rare_labels = tmp[tmp < rare_perc].index
            temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

        return temp_df

    def one_hot_encode(dataframe, categorical_cols, drop_first = True):
        dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
        return dataframe


def one_hot_encode(dataframe, categorical_cols, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if
                    temp_df[col].dtypes == 'O' and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(
                        axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

def prepare_dat_ais1 (df):

    # Let's Remove the unwanted Columns
    df.drop(['Unnamed: 0'], axis=1, inplace=True)

    # Let's look at the null values
    df.isnull().sum()

    df.info()

    df.head()

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    df.groupby('mmsi').count().shape

    df['mmsi'].value_counts()

    df['navigationalstatus'].value_counts()

    # lets see the number of missing values in the dataset

    missing_values_table(df)

    for col in num_cols:
        check_outliers(col, df, 0.01, 0.99)

    check_outliers(num_cols[0], df, 0.01, 0.99)

    check_outliers(num_cols[1], df, 0.2, 0.8)

    check_outliers(num_cols[2], df, 0.01, 0.99)

    check_outliers(num_cols[3], df, 0.01, 0.99)

    check_outliers(num_cols[4], df, 0.01, 0.99)

    check_outliers(num_cols[5], df, 0.01, 0.99)

    check_outliers(num_cols[6], df, 0.01, 0.99)

    #Correlations between variables



    corr_analyzer(df, corr_th=0.75, plot=True)

    # First, the filling was made according to those in the 'heading' but not in the 'cog'. So we have fewer missings at 'cog'

    #df['route'] = np.where(df['cog'].isnull(), df['heading'], df['cog'])

    df["route"] = df["cog"]

    df["route"] = df["route"].apply(lambda x: df["heading"] if x=="NaN" else x)

    missing_values_table(df)

    # Secondly, we divided the 360-degree route into 8 regions.
    rot= [-1, 45, 90, 135, 180, 225, 270, 315, 360]
    df['waypoint'] = pd.cut(df['route'], rot, labels=['NNE','ENE','ESE','SSE','SSW','WSW','WNW','NNW'])

    # Finally, the ships with less than 5.5kts speed and no route information were tagged as 'FIX'.
    df['waypoint'] = np.where((df['sog']<5.5) & (df['waypoint'].isnull()), 'FIX', df['waypoint'])
    df.head()

    #df['speed'] = df["sog"].fillna(df.groupby(['shiptype', 'waypoint'])['sog'].transform('mean'))
    #df.head()

    #speed mean ve max değerlerini feature olarak ata mean - median
    df = df.join(df.groupby('shiptype')['sog'].transform('mean').rename('sogmean'.format()))

    #df = df.join(df.groupby('shiptype')['sog'].transform('max').rename('sogmax'.format()))

    #dimension is not meaningful
    df['dimension'] = df['length'] * df['width']
    df.head()

    df = df.drop_duplicates(subset='mmsi') # one vessel has more than one data
    df.groupby(['shiptype'])['mmsi'].count()

    df.groupby(['shiptype']).agg({"sog":"mean"}).head(20)

    # no need anymore
    df.drop(['cog', 'heading', 'route', 'sog', 'mmsi', 'draught', 'width', 'length'], axis=1, inplace=True)

    df.head()

    # taking account for the new variables
    cat_cols, num_cols, cat_but_car = grab_col_names(df)


    for col in cat_cols:
        cat_summary(df,col)




    #rare encode gemi tipinde bilgi kaybına yol açıyor
    df = rare_encoder(df, 0.02)

    df.head()

    df.groupby(['shiptype'])['sogmean'].count()

    # combining the 'Unknown value' and 'Rare' groups at 'navigationalstatus' variable
    #df['status'] = df['navigationalstatus'].where(
    #~((df['navigationalstatus']=='Unknown value') | (df['navigationalstatus']=='Rare')), 'Other')

    df.head()

    #length and with not dropped

    df.drop(['navigationalstatus'], axis=1, inplace=True)
    df.head()

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    # with rare class
    for col in cat_cols:
        cat_summary(df, col)

    missing_values_table(df)

    df.info()

    #missing values
    df = df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    check_outliers(num_cols[0], df, 0.2, 0.8)

    check_outliers(num_cols[1], df, 0.2, 0.8)



    # Splitting data as an output and predictors;
    y = df['shiptype']
    X= df.drop('shiptype', axis=1)

    #dlist = ["vesseltype", "cargo", "date", "callsign", "name"]

    #X= df.drop(dlist, axis=1)

    # One hot encoding



    X = one_hot_encode(X, ['waypoint']) #, 'status'])
    X.head(3)

    X.info()

    scaler = RobustScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X.head()


    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=42)

    # lets see our training data
    print('Training data shape => ',X_train.shape)
    print('Testing data shape => ', X_test.shape)
    return X,y, X_train, X_test, y_train, y_test


#####
##prepare ais2
def prepare_dat_ais2 (df):

    # Let's Remove the unwanted Columns
    df.drop(['Unnamed: 0'], axis=1, inplace=True)

    # Let's look at the null values
    df.isnull().sum()

    df.info()

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    df.groupby('mmsi').count().shape

    df['mmsi'].value_counts()

    df['navstatus'].value_counts()

    # lets see the number of missing values in the dataset

    missing_values_table(df)

    for col in num_cols:
        check_outliers(col, df, 0.01, 0.99)


    #Correlations between variables



    corr_analyzer(df, corr_th=0.75, plot=True)

    # First, the filling was made according to those in the 'heading' but not in the 'cog'. So we have fewer missings at 'cog'

    #df['route'] = np.where(df['cog'].isnull(), df['heading'], df['cog'])

    df["route"] = df["cog"]

    df["route"] = df["route"].apply(lambda x: df["heading"] if x=="NaN" else x)

    missing_values_table(df)

    # Secondly, we divided the 360-degree route into 8 regions.
    rot= [-1, 45, 90, 135, 180, 225, 270, 315, 360]
    df['waypoint'] = pd.cut(df['route'], rot, labels=['NNE','ENE','ESE','SSE','SSW','WSW','WNW','NNW'])

    # Finally, the ships with less than 5.5kts speed and no route information were tagged as 'FIX'.
    df['waypoint'] = np.where((df['sog']<5.5) & (df['waypoint'].isnull()), 'FIX', df['waypoint'])
    df.head()

    #df['speed'] = df["sog"].fillna(df.groupby(['shiptype', 'waypoint'])['sog'].transform('mean'))
    #df.head()

    df.rename(columns= {'vesseltype': 'shiptype'}, inplace=True)

    #speed mean ve max değerlerini feature olarak ata
    df = df.join(df.groupby('shiptype')['sog'].transform('mean').rename('sogmean'.format()))

    #df = df.join(df.groupby('shiptype')['sog'].transform('max').rename('sogmax'.format()))

    #dimension is not meaningful
    df['dimension'] = df['length'] * df['beam']
    df.head()

    df = df.drop_duplicates(subset='mmsi') # one vessel has more than one data
    df.groupby(['shiptype'])['mmsi'].count()

    # no need anymore
    df.drop(['cog', 'heading', 'callsign', 'route', 'imonumber', 'latitude', 'longitude', 'cargo', 'name', 'timeoffix', 'date', 'sog', 'mmsi', 'beam', 'length'], axis=1, inplace=True)

    df.head()

    # taking account for the new variables
    cat_cols, num_cols, cat_but_car = grab_col_names(df)


    for col in cat_cols:
        cat_summary(df,col)




    #rare encode gemi tipinde bilgi kaybına yol açıyor
    df = rare_encoder(df, 0.02)

    df.head()

    # combining the 'Unknown value' and 'Rare' groups at 'navigationalstatus' variable
    df['status'] = df['navstatus'].where(
    ~((df['navstatus']=='Unknown value') | (df['navstatus']=='Rare')), 'Other')

    df.head()

    #length and with not dropped

    df.drop(['navstatus'], axis=1, inplace=True)
    df.head()

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    # with rare class
    for col in cat_cols:
        cat_summary(df, col)

    missing_values_table(df)

    df.info()

    #missing values
    df = df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)


    df.reset_index(drop=True)

    df["shiptype"]

    df = df.dropna(axis=0)

    # Splitting data as an output and predictors;
    y = df['shiptype']
    X= df.drop('shiptype', axis=1)

    #dlist = ["vesseltype", "cargo", "date", "callsign", "name"]

    #X= df.drop(dlist, axis=1)

    # One hot encoding



    X = one_hot_encode(X, ['waypoint', 'status'])
    X.head(3)

    X.info()

    scaler = RobustScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X.head()


    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)

    # lets see our training data
    print('Training data shape => ',X_train.shape)
    print('Testing data shape => ', X_test.shape)
    return X,y, X_train, X_test, y_train, y_test
#####

#####
def prepare_dat_ais3(df, scale=True):

    # Let's Remove the unwanted Columns
    #df.drop(['Unnamed: 0'], axis=1, inplace=True)

    # Let's look at the null values
    df.isnull().sum()

    df.info()

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    df.groupby('MMSI').count().shape

    df['MMSI'].value_counts()

    plt.rcParams['figure.figsize'] = [4, 4]

    df.groupby(['Ship type']).agg({"SOG": ["mean", "max"]}).plot(legend=True)

    df.groupby(['Ship type']).agg({"Width": ["mean", "max", "min"]}).plot(legend=True)

    df.groupby(['Ship type']).agg({"Length": ["mean", "max", "min"]}).plot(legend=True)

    df.groupby(['Ship type']).agg({"A": ["mean", "max"]}).plot(legend=True)

    df['Navigational status'].value_counts()

    # lets see the number of missing values in the dataset

    missing_values_table(df)

    for col in num_cols:
        check_outliers(col, df, 0.01, 0.99)


    #Correlations between variables



    corr_analyzer(df, corr_th=0.75, plot=True)

    # First, the filling was made according to those in the 'heading' but not in the 'cog'. So we have fewer missings at 'cog'

    #df['route'] = np.where(df['cog'].isnull(), df['heading'], df['cog'])

    df["route"] = df["COG"]

    df["route"] = df["route"].apply(lambda x: df["Heading"] if x=="NaN" else x)

    missing_values_table(df)

    # Secondly, we divided the 360-degree route into 8 regions.
    rot= [-1, 45, 90, 135, 180, 225, 270, 315, 360]
    df['waypoint'] = pd.cut(df['route'], rot, labels=['NNE','ENE','ESE','SSE','SSW','WSW','WNW','NNW'])

    # Finally, the ships with less than 5.5kts speed and no route information were tagged as 'FIX'.
    df['waypoint'] = np.where((df['SOG']<5.5) & (df['waypoint'].isnull()), 'FIX', df['waypoint'])
    df.head()

    #df['speed'] = df["sog"].fillna(df.groupby(['shiptype', 'waypoint'])['sog'].transform('mean'))
    #df.head()

    #speed mean ve max değerlerini feature olarak ata
    df = df.join(df.groupby('Ship type')['SOG'].transform('mean').rename('sogmean'.format()))

    df = df.join(df.groupby('Ship type')['SOG'].transform('max').rename('sogmax'.format()))

    df = df.join(df.groupby('Ship type')['Width'].transform('max').rename('WidthMax'.format()))

    df = df.join(df.groupby('Ship type')['Width'].transform('min').rename('WidthMin'.format()))

    df = df.join(df.groupby('Ship type')['Length'].transform('max').rename('LenMax'.format()))

    df = df.join(df.groupby('Ship type')['Length'].transform('min').rename('LenMin'.format()))

    df = df.join(df.groupby('Ship type')['Draught'].transform('max').rename('DraMax'.format()))

    #df = df.join(df.groupby('Ship type')['ROT'].transform('max').rename('ROTMax'.format()))

    #df = df.join(df.groupby('Ship type')['ROT'].transform('mean').rename('ROTMean'.format()))

    #dimension is not meaningful
    #df['dimension'] = df['Length'] * df['Width']
    df.head()

    df = df.drop_duplicates(subset='MMSI') # one vessel has more than one data

    df = df[(df["Ship type"] != "Undefined")]

    df = df[(df["Ship type"] != "Other")]

    df.groupby(['Ship type'])['MMSI'].count()

    # no need anymore
    # 'Draught', 'Width', 'Length'
    df.drop(['Callsign', 'Name', 'Cargo type', 'Destination', 'ETA', 'COG', 'Heading', 'route', 'SOG', 'MMSI'], axis=1, inplace=True)

    #df.drop(['Callsign', 'Name', 'Cargo type', 'Destination', 'ETA'], axis=1, inplace=True)

    df.drop(['# Timestamp', 'Type of mobile', 'Latitude', 'Longitude', 'ROT', 'IMO', 'Type of position fixing device', 'Data source type', 'A','B', 'C', 'D' ], axis=1, inplace=True)

    df.head()

    # taking account for the new variables
    cat_cols, num_cols, cat_but_car = grab_col_names(df)


    for col in cat_cols:
        cat_summary(df,col)




    #rare encode gemi tipinde bilgi kaybına yol açıyor
    df = rare_encoder(df, 0.04)

    df.head()

    df.groupby(['Ship type'])['sogmean'].count()

    #status'u iptal et
    # combining the 'Unknown value' and 'Rare' groups at 'navigationalstatus' variable
    #df['status'] = df['Navigational status'].where(
    #~((df['Navigational status']=='Unknown value') | (df['Navigational status']=='Rare')), 'Other')

    df.head()

    #length and with not dropped

    df.drop(['Navigational status'], axis=1, inplace=True)
    df.head()

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    # with rare class
    for col in cat_cols:
        cat_summary(df, col)

    missing_values_table(df)

    df.info()

    #missing values
    df = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

    #df.groupby(['Ship type']).agg({"SOG": ["mean", "max"]}).plot(legend=True)

    df.groupby(['Ship type']).agg({"Width": ["mean", "max", "min"]}).plot(legend=True)

    df.groupby(['Ship type']).agg({"Length": ["mean", "max", "min"]}).plot(legend=True)

    #df.groupby(['Ship type']).agg({"A": ["mean", "max"]}).head(20)

    # Splitting data as an output and predictors;
    y = df['Ship type']
    X= df.drop('Ship type', axis=1)

    #dlist = ["vesseltype", "cargo", "date", "callsign", "name"]

    #X= df.drop(dlist, axis=1)

    # One hot encoding



    X = one_hot_encode(X, ['waypoint'])#, 'status'])
    X.head(3)

    X.info()

    #scaler = RobustScaler()
    #X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    #X.head()


    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

    if(scale==True):
        scaler = RobustScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X.head()

    # lets see our training data
    print('Training data shape => ',X_train.shape)
    print('Testing data shape => ', X_test.shape)
    if(scale==True):
        return X,y, X_train, X_test, y_train, y_test, df, scaler
    else:
        return X, y, X_train, X_test, y_train, y_test, df

#####

    # lgbm = LGBMClassifier(random_state=17)
    # lgbm

# defining the hyper-parameters for lgbm

# lgbm_params = {'max_depth': [2,5,8,10],
#                'learning_rate':[0.05,0.1],
#                'n_estimators':[200, 400],
#                 'colsample_bytree':[0.3, 0.5, 1]}
#
# lgbm_cv = GridSearchCV(lgbm, lgbm_params, cv=10, n_jobs=-1, verbose=True)
# lgbm_cv.fit(X,y)

# def plot_importance(model, features, num = len(X)):
#     feature_imp = pd.DataFrame({'Value' : model.feature_importances_, 'Feature': features.columns})
#     plt.figure(figsize=(10,10))
#     sns.set(font_scale=1)
#     sns.barplot(x = 'Value', y = 'Feature', data = feature_imp.sort_values(by='Value', ascending=False)[0:num])
#     plt.title('Features')
#     plt.tight_layout()
#     plt.show()


# lgbm_f = lgbm.set_params(**lgbm_cv.best_params_, random_state=17).fit(X_train,y_train)
# plot_importance(lgbm_f, X_train)
#
# lgbm_final = lgbm.set_params(**lgbm_cv.best_params_, random_state=17)
# lgbm_results = cross_val_score(lgbm_final, X_test, y_test, cv =10, scoring="accuracy").mean()
# lgbm_results

from sklearn.impute import SimpleImputer

# Create an imputer object with median strategy
#imputer = SimpleImputer(strategy='median')

# Fit the imputer on X_train
#imputer.fit(X_train)

# Transform X_train and X_test using the trained imputer
#X_train = imputer.transform(X_train)
#X_test = imputer.transform(X_test)

#imputer.fit(X)
#X = imputer.transform(X)

#X = pd.DataFrame(X)
#X.head()

# Now the input data should be free of NaN or infinity values


# Base Models
def base_models(X, y, scoring="accuracy"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   #('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

# Hyperparameter Optimization

# config.py

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, 24, None],
             "max_features": [4, 5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [40, 80, 200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

gbm_params = {"learning_rate": [0.1, 0.5, 1],
                   "min_samples_split": [200, 300, 400, 500],
                   "max_depth": [7, 8, 9, 10]
}

lightgbm_params = { "max_depth": [4,5,6],
                    "learning_rate": [0.01, 0.1, 0.3, 0.5, 0.7, 1],
                   "n_estimators": [50, 100, 200, 300, 400],
                   "colsample_bytree": [0.3, 0.5, 0.7]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ("GBM", GradientBoostingClassifier(), gbm_params),
               #('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]

def hyperparameter_optimization(X, y, cv=3, scoring="accuracy"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        time1 = datetime.datetime.now()
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        time2 = datetime.datetime.now()
        print("Duration:", time2-time1)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

# Stacking & Ensemble Learning
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    print(best_models)
    print(X.head())
    print(y.head())
    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]), ('RF', best_models["RF"]),
                                              ('GBM', best_models["GBM"])],
                                  voting='soft').fit(X, y)

    print(voting_clf)

    cv_results = cross_validate(voting_clf, X, y, cv=3, error_score="raise", scoring="accuracy")

    cv_results = pd.DataFrame(cv_results)
    print(cv_results.head())

    #cv_results = cross_validate(voting_clf, X, y, cv=3, error_score="raise", scoring="f1_micro")


    #print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    #print(f"F1Score: {cv_results['test_f1'].mean()}")
    #print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

def plot_importance(model, features):
        num = len(features)
        feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False)[0:num])
        plt.title('Features')
        plt.tight_layout()
        plt.show()


def estimator(sample, y):
    print(f"Predicted Ship type \t : {lgbm_final.predict(random_ship)[0]}")
    temp = y.reset_index()
    print(f"Actual ship type \t : {temp[temp.index == sample.index[0]].iloc[0, 1]} ")

def main():

    #df= pd.read_csv("AIS-Ship_classification--main/AIS_UNACORN.csv")
    #df= pd.read_csv("AIS-Ship_classification--main/ais_data.csv")

    df= pd.read_csv("ais/aisdk-2023-04-19.csv", nrows=2000000)

    df_20= pd.read_csv("ais/aisdk-2023-04-20.csv", nrows=2000000)

    df_21 = pd.read_csv("ais/aisdk-2023-04-21.csv", nrows=2000000)

    df_22 = pd.read_csv("ais/aisdk-2023-04-22.csv", nrows=2000000)

    df_test = pd.read_csv("ais/aisdk-2023-04-23.csv", nrows=4000000)

    df = pd.concat([df,df_20, df_21, df_22])


    df.reset_index(drop=True, inplace=True)


    df.head()

    df.tail()

    df.info



#X,y, X_train, X_test, y_train, y_test = prepare_dat_ais2(df)
#X,y, X_train, X_test, y_train, y_test = prepare_dat_ais1(df)

    X,y, X_train, X_test, y_train, y_test, df, scaler = prepare_dat_ais3(df)

    X_f,y_f, X_train_f, X_test_f, y_train_f, y_test_f, df_f = prepare_dat_ais3(df_test,scale=False)

    df.groupby(['Ship type'])['sogmean'].count()

    df.describe().T


#df_f.groupby(['Ship type']).agg({"sogmax":"mean"}).head(20)


    df_f.describe().T

    X.info()

#base_models(X, y)
#best_models = hyperparameter_optimization(X, y)
#voting_clf = voting_classifier(best_models, X, y)

#base_models(X_f, y_f)
#best_models = hyperparameter_optimization(X_f, y_f)
#voting_clf = voting_classifier(best_models, X_f, y_f)

    base_models(X_train, y_train)
    best_models = hyperparameter_optimization(X_train, y_train)
    voting_clf = voting_classifier(best_models, X_train, y_train)


#join with final series

    lgbm_params = {'max_depth': [2,4,6, 8,10],
                    'learning_rate':[0.05,0.1, 0.2, 0.3],
                    'n_estimators':[10, 20, 30, 40, 50, 100, 200, 300, 400],
                     'colsample_bytree':[0.3, 0.5, 0.7, 1]}

    lgbm_best_params =  {'colsample_bytree': 0.3, 'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 100}

    lgbm = LGBMClassifier(random_state=17)

    lgbm_cv = GridSearchCV(lgbm, lgbm_params, cv=4, n_jobs=-1, verbose=True)
    lgbm_cv.fit(X_train,y_train)
    lgbm_final = lgbm.set_params(**lgbm_best_params).fit(X_train,y_train)

    plot_importance(lgbm_final, X_train)

    #scaler = RobustScaler()

    X_test.reset_index(drop=True, inplace=True)
    X_test = pd.DataFrame(scaler.transform(X_test))


    y_pred = lgbm_final.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(f"model training complete...")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(" ")
    print(report)



    pred_counts = pd.Series(y_pred).value_counts()
    plt.pie(pred_counts, labels=pred_counts.index, autopct='%1.2f%%')
    plt.title('Predicted Labels')
    plt.show()

# create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

# plot confusion matrix
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(set(y)))
    plt.xticks(tick_marks, set(y), rotation=90)
    plt.yticks(tick_marks, set(y))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.plot(include_values=True)
    plt.show()


    import seaborn as sns
    sns.heatmap(cm,
            annot=True, fmt=".4g")
    plt.ylabel('Prediction',fontsize=13)
    plt.xlabel('Actual',fontsize=13)
    plt.title('Confusion Matrix',fontsize=17)
    plt.show()

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    confusion_matrices = multilabel_confusion_matrix(y_test, y_pred)
    for confusion_matrix_ in confusion_matrices:
        disp = ConfusionMatrixDisplay(confusion_matrix_, display_labels=y_test)
        disp.plot(include_values=True, cmap="viridis", ax=None, xticks_rotation="vertical")
        plt.show()


    random_ship = X_test.sample(1)
    random_ship

    print (estimator(random_ship,y_test))

#scaler = RobustScaler()

    X_test_scaled = pd.DataFrame(scaler.transform(X_f))
#random_ship = X_test_scaled.sample(1)
#    random_ship

#print (estimator(random_ship,y_test_f))
    y_pred_f = lgbm_final.predict(X_test_scaled)
    report = classification_report(y_f, y_pred_f)
    print(f"model training complete...")
    accuracy = accuracy_score(y_f, y_pred_f)
    print(f"Accuracy: {accuracy}")
    print(" ")
    print(report)


    import pandas as pd

#other data with same model
#df_final= pd.read_csv("ais/aisdk-2023-04-19.csv", nrows=100000)

    df_final= pd.read_csv("ais/aisdk-2023-05-11.csv", nrows=500000)

    df_final.head()



    df_final.groupby(['Ship type'])['MMSI'].count()


#df_final = df_final[(df_final["Ship type"] != "Undefined")]

#df_final = df_final[(df_final["Ship type"] != "Dredging")]

#df_final = df_final[(df_final["Ship type"] != "HSC")]

#df_final = df_final[(df_final["Ship type"] != "Other")]

#df_final = df_final[(df_final["Ship type"] != "Pilot")]

    df_final.info()

#df = df.dropna()

    df_final.shape

    X,y, X_train, X_test, y_train, y_test, df_final = prepare_dat_ais3(df_final)

    df_final["Ship type"].value_counts()

    df_final.groupby(['Ship type'])['sogmean'].count()

    random_ship = X_f.sample(1)
    random_ship

    print (estimator(random_ship,y))

    X_test

#plot trajectory
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    def load_data(file_path):
    """
    import trajectory data
    :return: trajectory data
    df= pd.read_csv("ais/aisdk-2023-04-19.csv")
    df.head()
    """


    #file_path = "ais/aisdk-2023-04-19.csv"
        data = pd.read_csv(file_path, usecols=['# Timestamp', 'MMSI', 'Latitude', 'Longitude'])
        data.rename(columns={'Longitude': 'long', 'Latitude': 'lat', '# Timestamp': 't', 'MMSI': 'mmsi'}, inplace=True)
        data['long'] = data['long'].map(lambda x: x / 600000.0)
        data['lat'] = data['lat'].map(lambda x: x / 600000.0)
        return data


    if __name__ == '__main__':
        trajectories = load_data('ais/aisdk-2023-04-19.csv')
        params = {'axes.titlesize': 'large',
              'legend.fontsize': 14,
              'legend.handlelength': 3}
        plt.rcParams.update(params)

        for shipmmsi, dt in trajectories.groupby('mmsi'):
            plt.plot(dt['long'].values, dt['lat'].values, color='green', linewidth=0.5)

        plt.yticks(fontproperties='Times New Roman', size=12)
        plt.xticks(fontproperties='Times New Roman', size=12)
        plt.xlabel('Longitude', fontdict={'family': 'Times New Roman', 'size': 14})
        plt.ylabel('Latitude', fontdict={'family': 'Times New Roman', 'size': 14})
        plt.title('Preprocessed Trajectories', fontdict={'family': 'Times New Roman', 'size': 14})
        plt.ticklabel_format(useOffset=False, style='plain')
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    #drawtrajectory
        import os
        import pandas as pd
        import numpy as np
        import webbrowser as wb
        import folium
        from folium.plugins import HeatMap, MiniMap, MarkerCluster


    # draw a heatmap
        def draw_heatmap(map):
            data = (
                np.random.normal(size=(100, 3)) *
                np.array([[1, 1, 1]]) +
                np.array([[30.9, 122.52, 1]])
            ).tolist()
            HeatMap(data).add_to(map)


    # add minimap
        def draw_minimap(map):
            minimap = MiniMap(toggle_display=True,
                          tile_layer='Stamen Watercolor',
                          position='topleft',
                          width=100,
                          height=100)
            map.add_child(minimap)


        def draw_circlemarker(loc, spd, cog, map):
            tip = 'Coordinates:' + str(loc) + "\t" + 'Speed:' + str(spd) + '\t' + 'COG:' + str(cog)
            folium.CircleMarker(
                location=loc,
                radius=3.6,
                color="blueviolet",
                stroke=True,
                fill_color='white',
                fill=True,
                weight=1.5,
                fill_opacity=1.0,
                opacity=1,
                tooltip=tip
            ).add_to(map)


    # draw a small information marker on the map
    def draw_icon(map, loc):
        mk = folium.features.Marker(loc)
        pp = folium.Popup(str(loc))
        ic = folium.features.Icon(color="blue")
        mk.add_child(ic)
        mk.add_child(pp)
        map.add_child(mk)


    # draw a stop marker on the map
    def draw_stop_icon(map, loc):
        # mk = folium.features.Marker(loc)
        # pp = folium.Popup(str(loc))
        # ic = folium.features.Icon(color='red', icon='anchor', prefix='fa')
        # mk.add_child(ic)
        # mk.add_child(pp)
        # map.add_child(mk)
        folium.Marker(loc).add_to(map)


    def draw_line(map, loc1, loc2):
        kw = {"opacity": 1.0, "weight": 6}
        folium.PolyLine(
            smooth_factor=10,
            locations=[loc1, loc2],
            color="red",
            tooltip="Trajectory",
            **kw,
        ).add_to(map)


    def draw_lines(map, coordinates, c):
        folium.PolyLine(
            smooth_factor=0,
            locations=coordinates,
            color=c,
            weight=0.5
        ).add_to(map)


    # save the result as HTML to the specified path
    def open_html(map, htmlpath):
        map = m
        htmlpath = 'draw.html'
        map.save(htmlpath)
        #search_text = 'cdn.jsdelivr.net'
        #replace_text = 'gcore.jsdelivr.net'
        #with open(htmlpath, 'r', encoding='UTF-8') as file:
        #    data = file.read()
        #    data = data.replace(search_text, replace_text)
        #with open(htmlpath, 'w', encoding='UTF-8') as file:
        #    file.write(data)
        #chromepath = "/Applications/Google Chrome.app"
        #wb.register('chrome', None, wb.BackgroundBrowser(chromepath))
        #wb.get('chrome').open(htmlpath, autoraise=1)
        wb.open_new_tab(htmlpath)

    # read .csv file
    def read_traj_data(path):
        #path = csv_path
        P = pd.read_csv(path, dtype={'Latitude': float, 'Longitude': float},nrows=50000)

        P.isnull().sum()
        P.info()


        #cat_cols, num_cols, cat_but_car = grab_col_names(P)

        #missing_values_table(P)

        #for col in num_cols:
        #    check_outliers(col, P, 0.01, 0.99)

        #P.describe().T

        P = P[(P["Latitude"] > 50) & (P["Latitude"] < 60)]
        P = P[(P["Longitude"] > 1) & (P["Longitude"] < 20)]

        P.info()
        #P.dropna(inplace=True)

        locations_total = P.loc[:, ['Latitude', 'Longitude']].values.tolist()
        speed_total = P.loc[:, ['SOG']].values.tolist()
        cog_total = P.loc[:, ['COG']].values.tolist()
        locations_stay = P.loc[P['Navigational status'] == 1, ['Latitude', 'Longitude']].values.tolist()
        lct = [P['Latitude'].mean(), P['Longitude'].mean()]
        return locations_total, speed_total, cog_total, locations_stay, lct


    def draw_single_traj(csv_path):
        '''
        draw a single trajectory
        :param data: file path
        :return: null
        '''
        locations, spds, cogs, stays, ct = read_traj_data(csv_path)
        m = folium.Map(ct, zoom_start=15, attr='default')  # 中心区域的确定
        folium.PolyLine(  # polyline方法为将坐标用实线形式连接起来
            locations,  # 将坐标点连接起来
            weight=1.0,  # 线的大小为1
            color='blueviolet',  # 线的颜色
            opacity=0.8,  # 线的透明度
        ).add_to(m)  # 将这条线添加到刚才的区域map内
        num = len(locations)
        for i in range(num):
            draw_circlemarker(locations[i], spds[i], cogs[i], m)
        for i in iter(stays):
            draw_stop_icon(m, i)
        output_path = os.getcwd() + 'show.html'
        open_html(m, output_path)


    def draw_trajs(file_path):
        '''
        draw multiple trajectories
        :param data: file path
        :return: null
        '''
        map = folium.Map([31.1, 122.5], zoom_start=10, attr='default')  # 中心区域的确定
        draw_minimap(map)
        fls = os.listdir(file_path)
        scatterColors = ['blue', 'red', 'yellow', 'cyan', 'purple', 'orange', 'olive', 'brown', 'black', 'm']
        i = 0
        for x in fls:
            i = i + 1
            colorSytle = scatterColors[i % len(scatterColors)]
            df = pd.read_csv(file_path + "/" + x, encoding="gbk")
            df['MMSI'] = df['MMSI'].apply(lambda _: str(_))
            df['Longitude'] = df['Longitude'].map(lambda x: x / 1.0)
            df['Latitude'] = df['Latitude'].map(lambda x: x / 1.0)
            for shipmmsi, dt in df.groupby('MMSI'):
                if len(dt) > 2:
                    dt_copy = dt.copy(deep=True)
                    dt_copy.sort_values(by='# Timestamp', ascending=True, inplace=True)
                    locations = dt_copy.loc[:, ['Latitude', 'Longitude']].values.tolist()
                    draw_lines(map, locations, colorSytle)
        output_path = os.getcwd() + 'show.html'
        open_html(map, output_path)


    if __name__ == '__main__':
        csv_path = r'ais/aisdk-2023-04-19.csv'
        draw_single_traj(csv_path)
        #draw_trajs(csv_path)
        # csv_path = r'./data'
        # draw_trajs(csv_path)

