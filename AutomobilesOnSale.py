# Importing Necessary Library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from datetime import datetime
from ipywidgets import interact, widgets

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')


# Importing Data from autos.csv file
df_auto = pd.read_csv(r'C:\Users\Ad\Desktop\bros f proj\Automobiles_Capstone_Project-master\Dataset\autos.csv', encoding='latin-1')


# Investigating the Dimension of Extracted Observations
df_auto.shape


# Checking null values in Columns
df_auto.isnull().sum()


# Defining a function to fill the null values with their max value counts of individual's column
def impute_missing_values(parameter):
    df_auto[parameter] = df_auto[parameter].fillna(df_auto[parameter].value_counts().index[0])


# Missing value imputation using "impute_missing_values" function
impute_missing_values('vehicleType')
impute_missing_values('gearbox')
impute_missing_values('model')
impute_missing_values('fuelType')
impute_missing_values('notRepairedDamage')


 
# Checking all the columns of dataset
df_auto.columns


# Dropping duplicate observation from dataset
df_auto = df_auto.drop_duplicates()

# Checking shape of datset
df_auto.shape

# Converting German word "ja" to 'yes' & "nein" to 'no' in English for better understanding
df_auto['notRepairedDamage'] = df_auto['notRepairedDamage'].map({'ja':'yes', 'nein':'no'})


# Investigating number of cars available for sale

print('Number of cars available in dataset : ', df_auto['name'].nunique())

# Dropping the feature : name as it is unnecessary while building model
df_auto = df_auto.drop(['name'], axis=1)


# Investigating overall structure of feature : monthOfRegistration
df_auto.monthOfRegistration.describe()

# As there are 12 months in Calendar, 13 months can't be right. Removing observations of month = 12, contains ~ 12k observation
df_auto = df_auto[df_auto.monthOfRegistration != 12]


# Univariate Analysis of : Sellers
sns.barplot(df_auto.seller.value_counts().index, df_auto.seller.value_counts().values, alpha=0.9)
plt.xlabel('Sellers')
plt.ylabel('Count')
plt.title('Distribution Of Car Sellers');



# As almost all of the Sellers are from private we can drop this feature
df_auto = df_auto.drop(['seller'], axis=1)



# Univariate Analysis of : Offer Type
sns.barplot(df_auto.offerType.value_counts().index, df_auto.offerType.value_counts().values, alpha=0.9)
plt.xlabel('Offer Type')
plt.ylabel('Count')
plt.title('Distribution Of Car Offers');




# As almost all of the Offers are from Angebot we can drop this feature
df_auto = df_auto.drop(['offerType'], axis=1)




print('Number of observation where price is 0 : ', df_auto[df_auto.price == 0]['price'].count())




# Number of observation where price is > 200000
df_auto[df_auto.price > 200000]['price'].count()




# Number of observation where price is < 200
df_auto[df_auto.price < 200]['price'].count()



# Considering outlier, selecting observations in between $200 & $200000
df_auto = df_auto[(df_auto.price > 200) & (df_auto.price < 200000)]




# Distribution of Price
sns.distplot(df_auto.price)
plt.xlabel("Price")
plt.ylabel('Frequency')
plt.title("Distribution of Car's Price");



# Logarithm of Price Distribution
sns.distplot(np.log(df_auto.price))
plt.xlabel("Logarithm of Car's Price")
plt.ylabel('Frequency')
plt.title("Distribution Log of Car's Price");



# Univariate Analysis of : AB Testing
sns.barplot(df_auto.abtest.value_counts().index, df_auto.abtest.value_counts().values, alpha=0.9)
plt.xlabel('Type of Testing')
plt.ylabel('Count')
plt.title('Distribution Of Car Testing');



# Univariate Analysis of : Vehicle Type
plt.figure(figsize=(12,6))
sns.barplot(df_auto.vehicleType.value_counts().index, df_auto.vehicleType.value_counts().values, alpha=0.9)
plt.xlabel('Type of Vehicle')
plt.ylabel('Count')
plt.title('Distribution Of Vehicle Types');





# Univariate Analysis of : Gear Type
sns.barplot(df_auto.gearbox.value_counts().index, df_auto.gearbox.value_counts().values, alpha=0.9)
plt.xlabel('Type of Gears')
plt.ylabel('Count')
plt.title('Distribution Of Types of Gears');


print('No of PowerPS is having value of 0 : ', df_auto[df_auto.powerPS == 0]['powerPS'].count())




print('No of PowerPS is having value of more than 662 is : ', df_auto[df_auto.powerPS > 662]['powerPS'].count())


# Removng cars having HP of 662 as the latest technology doesn't have HP > 662
# Removing observations having HP of 0 - as its meaningless
df_auto = df_auto[(df_auto.powerPS > 0) & (df_auto.powerPS < 663)]




# Distribution of Top 10 Horse Powered car sold
plt.figure(figsize=(16,6))
sns.lineplot(df_auto[df_auto.powerPS > 0].powerPS.value_counts()[:10].index, 
             df_auto[df_auto.powerPS > 0].powerPS.value_counts()[:10].values)
plt.xticks(df_auto[df_auto.powerPS > 0].powerPS.value_counts()[:10].index)
plt.xlabel('Horse Power')
plt.ylabel('No. of Car Sold With Available Horse Power')
plt.title('Top 10 Car Sold with Horse Power Variation');




# Distribution of Top 10 car's moel sold
sns.lineplot(df_auto.model.value_counts()[:10].index, df_auto.model.value_counts()[:10].values)
plt.xticks(df_auto.model.value_counts()[:10].index)
plt.xlabel('Cars Model')
plt.ylabel('Frequency')
plt.title('Top 10 Cars Model Sold');




# Ditribution of Mesurement of KM a car ran before coming for sale
plt.figure(figsize=(12,6))
sns.distplot(df_auto.kilometer)
plt.xlabel("KM's Car Ran")
plt.ylabel('Frequency')
plt.title('Car was Driven in KM');



# No. of car registerd in a month for sale
plt.figure(figsize=(12,6))
sns.lineplot(df_auto.monthOfRegistration.value_counts().index, df_auto.monthOfRegistration.value_counts().values)
plt.xticks(df_auto.monthOfRegistration.value_counts().index.sort_values(), 
           ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlabel("Month Of Registration")
plt.ylabel('Frequency')
plt.title('No. Of Cars Sold In Month');




# Univariate Analysis of : fuel Type
plt.figure(figsize=(12,6))
sns.barplot(df_auto.fuelType.value_counts().index, df_auto.fuelType.value_counts().values, alpha=0.9)
plt.xlabel('Types of Fuel')
plt.ylabel('Frequency')
plt.title('Distribution Of Car with Types of Fuel');




# Univariate Analysis of : Top 10 Car's Brand
plt.figure(figsize=(12,6))
sns.barplot(df_auto.brand.value_counts()[:10].index, df_auto.brand.value_counts()[:10].values, alpha=0.9)
plt.xlabel("Car's Brand")
plt.ylabel('Frequency')
plt.title("Top 10 Car's Brand Sold");





# Univariate Analysis of : Car was Repaired: yes/no before sale
sns.barplot(df_auto.notRepairedDamage.value_counts().index, df_auto.notRepairedDamage.value_counts().values, alpha=0.9)
plt.xlabel('Repaired Post Damage')
plt.ylabel('Frequency')
plt.title('Distribution Of Car Not Repaired Damaged');





# Investigating overall structure of feature : yearOfRegistration
df_auto.yearOfRegistration.describe()





# Observation which is older than 1989
df_auto[df_auto.yearOfRegistration < 1989]['yearOfRegistration'].count()





# Observation which is more than 2019
df_auto[df_auto.yearOfRegistration > 2019]['yearOfRegistration'].count()



# Taking into considearion which is in the year of between 1989 & 2019
df_auto = df_auto[(df_auto.yearOfRegistration >= 1989) & (df_auto.yearOfRegistration <= 2019)]



# No of car was registered for sale throughout the year
sns.lineplot(df_auto.groupby('yearOfRegistration')['price'].count().index,
            df_auto.groupby('yearOfRegistration')['price'].count().values,
            data=df_auto)
plt.xlabel('Years of Registration')
plt.ylabel('Price')
plt.title('Variation Of Price with Year');




# No of days it took to sold while purchasing from E-bay
days = []
for time1, time2 in zip(df_auto['dateCrawled'], df_auto['lastSeen']):
    time = datetime.strptime(time2, '%Y-%m-%d %H:%M:%S') - datetime.strptime(time1, '%Y-%m-%d %H:%M:%S')
    days.append(time.days)
        
df_auto['Sold_In_Days'] = days


# Investigating the feature : Sold_In_Days
df_auto.Sold_In_Days.describe()


# Removing the observations having negative values as it doesn't make any sense
df_auto = df_auto[df_auto.Sold_In_Days >= 0]


# Distribution of no. of cars sold in days
plt.figure(figsize=(12,6))
sns.barplot(df_auto.Sold_In_Days.value_counts().index, df_auto.Sold_In_Days.value_counts().values, alpha=0.9)
plt.xlabel('Sold In Days')
plt.ylabel('Frequency')
plt.title('No. Of Cars Sold in Days');


# Dropping the below mentioned features as they are unnecesary now while building models
# All the postal code is from Germany only
df_auto = df_auto.drop(['dateCrawled', 'lastSeen', 'dateCreated', 'nrOfPictures', 'model', 'abtest', 'postalCode'], axis=1)


# Corelation matrix with Heatmap annotation
sns.heatmap(df_auto.corr(), annot=True);


# Function to get the Chi Square value & P value
def chi_p_value(cat1, cat2):
    table = pd.crosstab(df_auto[cat1], df_auto[cat2])
    chi2, p, dof, expected = chi2_contingency(table.values)
    if p < 0.05:
        print("Chi Square Statistics and p value of {} and {} is {}, {}".format(cat1, cat2, chi2, p))


# Extracting Chi Square value & p value
for i in range(len(df_auto.select_dtypes(include=['object']).columns)):
    for cat2 in df_auto.select_dtypes(include=['object']).columns[df_auto.select_dtypes(include=['object']).columns != 
                                                                  df_auto.select_dtypes(include=['object']).columns[i]]:
        chi_p_value(df_auto.select_dtypes(include=['object']).columns[i], cat2)


# Taking into consideration of Sold_In_Dyas which is <= 5 days for bi-variate analysis
# It will give us the top most sold cars in first consecutive 5 days
df_auto_sold = df_auto[df_auto.Sold_In_Days < 5]



# Function to visualize bivariate analysis
def bivariate_analysis(param, xlabel):
    df_auto_sold.groupby([param, 'Sold_In_Days'])['price'].count().unstack().plot(kind='bar')
    plt.xticks(rotation=360)
    plt.xlabel(xlabel)
    plt.ylabel('Price')
    plt.title('Price Distribution of ' + xlabel + ' Sold within 0-4 days');


bivariate_analysis('vehicleType', 'Types Of Vehicle')

bivariate_analysis('gearbox', 'Types Of Gear')

bivariate_analysis('fuelType', 'Types Of Fuel')

print("No. Of cars sold on the day the ad was published : ", df_auto[df_auto.Sold_In_Days == 0].count()[0])
print("No. Of cars sold on the 1st day the ad was published : ", df_auto[df_auto.Sold_In_Days == 1].count()[0])
print("No. Of cars sold on the 2nd day the ad was published : ", df_auto[df_auto.Sold_In_Days == 2].count()[0])


df_auto.head()

# Investigating the count of individual Categorical Features
for col in df_auto.select_dtypes(include=['object']).columns:
    print(col, len(df_auto[col].unique()))


# Interactive Distribution of Horsepower with Price
# Visualization possible among year/month/days/gearbox/damage
def plot_year(year, month, days, gearbox, damage):
    
    data = df_auto[(df_auto.yearOfRegistration == year) & (df_auto.monthOfRegistration == month) & 
                   (df_auto.Sold_In_Days == days) & (df_auto.gearbox == gearbox) & 
                   (df_auto.notRepairedDamage == damage)]
    
    area = 2 * df_auto.powerPS
    
    data.plot.scatter('powerPS', 'price', s = area, linewidth = 1, edgecolor='k', figsize=(12,8), alpha=0.7)
    
    plt.xlabel('Horse Power')
    plt.ylabel('Price')
    title = 'Variation of Price with Horse Power in ' +  str(year)
    plt.title(title)


interact(plot_year, year=widgets.IntSlider(min=1989, max=2019, step=1, value=2003, description='Year '), 
         month=widgets.IntSlider(min=1, max=12, step=1, value=2, description='Month '),
        days=widgets.IntSlider(min=0, max=10, step=1, value=0, description='Day '),
        gearbox = widgets.RadioButtons(value='manuell', options=list(df_auto.gearbox.unique()), description="Gear Type "),
        damage = widgets.RadioButtons(value='no', options=list(df_auto.notRepairedDamage.unique()), description="Repaired "))




df_auto.head()

# Importing Necessary Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


X = df_auto.drop(['price'], axis=1)
y = df_auto.price


#X = pd.get_dummies(data=X,columns= ['vehicleType','yearOfRegistration','gearbox','monthOfRegistration','fuelType','brand','notRepairedDamage'],drop_first=True)
X = pd.get_dummies(data=X,columns= ['vehicleType','gearbox','fuelType', 'brand','notRepairedDamage'])

X.to_csv(r'C:\Users\Ad\Desktop\bros f proj\Automobiles_Capstone_Project-master\X_dummy.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


 
rf = RandomForestRegressor(n_estimators = 50)
rf.fit(X_train, y_train)
score = rf.score(X_test, y_test)
print('Accuracy Of Random Forest: ', score)

pred=rf.predict(X_test)
print(pred)


from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(pred,y_test))
print("RMSE of model is : ",rmse)


import joblib

filename = r'C:\Users\Ad\Desktop\bros f proj\Automobiles_Capstone_Project-master\persis model\finalized_model.sav'
joblib.dump(rf, filename)




from IPython.display import Image  
from sklearn import tree
import pydotplus

# Visualize data
data_feature_names=['vehicleType','price','yearOfRegistration','gearbox','powerPS','kilometer','Sold_In_Days','monthOfRegistration','fuelType','brand','notRepairedDamage']
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(X_train, y_train)
tree_small = rf_small.estimators_[5]
dot_data = tree.export_graphviz(tree_small,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

