#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import cufflinks as cf
from sklearn.linear_model import LinearRegression
from collections import Counter
import joblib


# In[2]:


#Importing Train and Test data
df = pd.read_csv(r"D:\MSBA Study Materials\Projects\Data and Programming\train.csv")

df_test = pd.read_csv(r"D:\MSBA Study Materials\Projects\Data and Programming\test.csv")


# In[3]:


def check(df):
    list_check=[]
    columns=df.columns
    
    for col in columns:
        dtypes=df[col].dtypes
        nunique=df[col].nunique()
        sum_null=df[col].isnull().sum()
        list_check.append([col,dtypes,nunique,sum_null])
    df_check=pd.DataFrame(list_check, columns=['column','dtypes','nunique','sum_null'])
        
    return df_check

check(df)


# In[4]:


null_data=(df.isnull().mean().sort_values(ascending=False)*100).reset_index()
null_data.rename(columns={0:"Average"},inplace=True)
null_data
# fig=px.histogram(null_data,x="Average",y="index",title="Percentage of Missing values",color="index",labels={"Average":"Percentage of missing values","index":"Column Names"})

# fig.show()


# In[5]:


diff = dict(df.groupby("satisfaction")["Arrival Delay in Minutes"].mean())
df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df.satisfaction.map(diff))
df.isnull().sum()
df.drop("Unnamed: 0", axis = 1, inplace = True)
df.isnull().sum()

df.head()


# In[6]:


df.columns


# In[7]:


temp = df.groupby('Class')[['Flight Distance','Age']].mean().reset_index()
px.histogram(temp, x = 'Class' , y='Flight Distance', color = 'Age' ,barmode ='group' )


# In[8]:


Eco_class = df[df['Class'] == 'Eco']
Eco_class = Eco_class.mean().reset_index()
Eco_class.columns = ['index', 'Mean']
Eco_class


# In[9]:


Eco_class.drop(index =[0,2], inplace=True)
px.histogram(Eco_class, x='index', y='Mean', color = 'index')


# In[10]:


Business_class = df[df['Class'] == 'Business']
Business_class  = (Business_class.mean()).reset_index()
Business_class.columns = ['index', 'Mean']


# In[11]:


Business_class.drop(index = [0,2], inplace = True)
px.histogram(Business_class, x = 'index', y = 'Mean', color='index')


# In[ ]:





# In[12]:


diff = dict(df_test.groupby("satisfaction")["Arrival Delay in Minutes"].mean())
df_test["Arrival Delay in Minutes"] = df_test["Arrival Delay in Minutes"].fillna(df_test.satisfaction.map(diff))
df_test.isnull().sum()
df_test.drop("Unnamed: 0", axis = 1, inplace = True)
df_test.isnull().sum()


# In[13]:


fig = px.histogram(df, x="satisfaction")
fig.show()


# In[14]:


categorical_columns = df.select_dtypes(include = ['object'])
numerical_columns = df.select_dtypes(exclude =['object']) 

df_num_cat = pd.DataFrame()
df_num_cat['Number of Numerical Columns'] = [numerical_columns.shape[1]]
df_num_cat['Number of Categorical Columns'] = [categorical_columns.shape[1]]
df_num_cat_transpose = df_num_cat.T.reset_index()
df_num_cat_transpose.rename(columns={0:"Count"},inplace=True)

fig=px.bar(df_num_cat_transpose,x="index",y= "Count",title="Count of Categorical and Numerical Columns",color="index",labels={"index":"Categorical or Numerical Columns",0:"Count"})
fig.show()


# In[15]:


categorical_columns.describe().T


# In[16]:


numerical_columns.describe().T


# In[17]:


def detect_outliers(df,features):
    outlier_indices=[]
    
    for c in features:
        Q1=np.percentile(df[c],25)
        
        Q3=np.percentile(df[c],75)
        
        IQR= Q3-Q1
        
        outlier_step= IQR * 1.5
        
        outlier_list_col = df[(df[c]< Q1 - outlier_step)|( df[c] > Q3 + outlier_step)].index
        
        outlier_indices.extend(outlier_list_col)
        
                
    outliers_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outliers_indices.items() if v>2)
    return multiple_outliers

df.loc[detect_outliers(df, ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'])]


# In[18]:


df = df.drop(detect_outliers(df,['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']),axis = 0).reset_index(drop = True)


# In[19]:


fig=px.pie(df,values=df["satisfaction"].value_counts(),
           names=["Satisfied","Dissatisfied"],title="<b>Satisfied And Dissatisfied", hole = 0.5)
fig.show()


# In[20]:


fig=px.violin(df,y="Age",x="Gender",color="satisfaction",title="<b>Age vs Gender vs satisfaction")
fig.update_layout(template="plotly")
fig.show()


# In[21]:


plt.figure(figsize=(20,20))
corr = df.corr()
sns.heatmap(corr, annot=True)


# In[22]:


cols  = ['satisfaction']
dummies = pd.get_dummies(df[cols])
df = pd.concat([df,dummies],axis = 1)
df.head()


# In[23]:


cols  = ['satisfaction']
dummies = pd.get_dummies(df_test[cols])
df_test = pd.concat([df_test,dummies],axis = 1)
df_test.head()


# In[24]:


sns.countplot(data = df ,x = df["Class"], hue = "satisfaction", palette = "mako")


# In[25]:


sns.countplot(data = df ,x = df["Type of Travel"], hue = "satisfaction", palette = "mako")


# In[26]:


sns.displot(df["Flight Distance"], kde= True, bins= 50)


# In[27]:


df.columns


# In[28]:


sns.displot(df['Departure/Arrival time convenient'], kde= True, bins= 50)


# In[29]:


g = sns.FacetGrid(df, row="Class", col="Type of Travel", hue = 'satisfaction')
g.map(plt.scatter, "Departure Delay in Minutes", "Arrival Delay in Minutes")
plt.subplots_adjust(hspace=0.4, wspace=2)
g.add_legend()


# In[30]:


fig = px.histogram(df, x="Customer Type", color="satisfaction")
fig.show()


# In[34]:


sns.displot(df["Age"], kde= True, bins= 10)


# In[35]:


g = sns.FacetGrid(df, row="Class", col="Type of Travel", hue = 'satisfaction')
g.map(plt.scatter, "Age", "Departure Delay in Minutes")
plt.subplots_adjust(hspace=0.4, wspace=2)
g.add_legend()


# In[36]:


#Filtering the Dataset with only Loyal Customers as the Customer Type
dissatisfied_customers_df = df[df["satisfaction"] == "neutral or dissatisfied"]

#Visualising Class wise Satisfaction Rating for Loyal Customers
fig1 = px.sunburst(dissatisfied_customers_df,path=["Class", "Customer Type"],template="plotly", title = "<b>Class wise Customer Type for Dissatisfied Customers")
fig1.show()


# In[51]:


l1 = []
for x in round(df.iloc[:, 7:21].mean(), 2):
    l1.append(x)
l1


# In[52]:


short_medium_flight = df[df['Flight Distance'] < 3000]
l1 = []
for x in round(short_medium_flight.iloc[:, 7:21].mean(), 2):
    l1.append(x)
print(l1)


# In[53]:


long_flight = df[df['Flight Distance'] > 3000]
l1 = []
for x in round(long_flight.iloc[:, 7:21].mean(), 2):
    l1.append(x)
print(l1)


# In[37]:


px.histogram(df, x="Age", nbins = 10, color="satisfaction", color_discrete_sequence=[
                 "orange", "purple"])


# In[54]:


import plotly.graph_objects as go

categories = []
for x in df.iloc[:, 7:21]:
    categories.append(x)

fig = go.Figure()


fig.add_trace(go.Scatterpolar(
      r = [2.74, 2.94, 2.92, 3.0, 3.39, 3.92, 3.91, 3.79, 3.7, 3.73, 3.79, 3.48, 3.79, 3.56],
      theta=categories,
      fill='toself',
      name='Long Distance'

))
fig.add_trace(go.Scatterpolar(
      r = [2.73, 3.07, 2.74, 2.97, 3.19, 3.19, 3.4, 3.32, 3.35, 3.32, 3.62, 3.29, 3.63, 3.26],
      theta=categories,
      fill='toself',
      name='Short or Medium Distance'

))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 5]
    )),title = "<b>Radar Chart of Average ratings of Services based on Flight Distance",
  showlegend=True
)

fig.show()


# In[38]:


fig = px.scatter(df, x="Departure Delay in Minutes", y="Arrival Delay in Minutes",
	         size="Flight Distance", color="satisfaction")
fig.show()


# In[58]:


sat_corr = df.corr()[["satisfaction_satisfied"]].sort_values(by='satisfaction_satisfied', ascending=False)
sns.heatmap(sat_corr, annot=True, cmap="YlGnBu")


# In[59]:


sat_corr = df.corr()[["satisfaction_neutral or dissatisfied"]].sort_values(by='satisfaction_neutral or dissatisfied', ascending=False)
sns.heatmap(sat_corr, annot=True, cmap="YlGnBu", xticklabels=True, yticklabels=True)


# In[39]:


from sklearn.preprocessing import LabelEncoder
df['Gender'] = df['Gender'].astype('str')
df['Class'] = df['Class'].astype('str')
df['Customer Type'] = df['Customer Type'].astype('str')
df['Type of Travel'] = df['Type of Travel'].astype('str')

df['satisfaction'] = df['satisfaction'].astype('str')
        
df[['Gender', 'Customer Type','Type of Travel','Class','satisfaction']] =  df[['Gender', 'Customer Type', 'Type of Travel', 'Class','satisfaction']].apply(LabelEncoder().fit_transform)


# In[40]:


from sklearn.preprocessing import LabelEncoder
df_test['Gender'] = df_test['Gender'].astype('str')
df_test['Class'] = df['Class'].astype('str')
df_test['Customer Type'] = df_test['Customer Type'].astype('str')
df_test['Type of Travel'] = df_test['Type of Travel'].astype('str')

df_test['satisfaction'] = df_test['satisfaction'].astype('str')
        
df_test[['Gender', 'Customer Type','Type of Travel','Class','satisfaction']] =  df_test[['Gender', 'Customer Type', 'Type of Travel', 'Class','satisfaction']].apply(LabelEncoder().fit_transform)


# In[41]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

X = df.drop('satisfaction', axis = 1)
y = df.satisfaction
X_test = df_test.drop('satisfaction', axis = 1)
y_test = df_test.satisfaction


# In[42]:


clf = DecisionTreeClassifier()
clf = clf.fit(X,y)
y_pred = clf.predict(X_test)


# In[43]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[44]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 100) 
 
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
rfc.fit(X, y)
 
# performing predictions on the test dataset
y_pred_rfc = rfc.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_rfc))


# In[45]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf, X_test, y_test)
plt.show()


# In[60]:


from sklearn.linear_model import LinearRegression
X = df.drop('satisfaction' , axis=1)
y = df.satisfaction 
reg = LinearRegression()
reg.fit(X, y)

y_pred_lr = reg.predict(X_test)


# In[61]:


X = df.drop('satisfaction' , axis=1)
y = df.satisfaction

X_test = df_test.drop('satisfaction' , axis=1)
y = df_test.satisfaction


# In[62]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

X = df.drop('satisfaction' , axis=1)
y = df.satisfaction 


# In[63]:


sfs1 = sfs(knn, 
           k_features=(1,13), 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='accuracy',
           cv=5)


# In[64]:


sfs1 = sfs1.fit(X,y)
sfs1


# In[73]:


#sfs2.subsets_


# In[ ]:


from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
fig1 = plot_sfs(sfs2.get_metric_dict(),
                kind='std_dev',
                figsize=(6, 4))


# In[66]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
df = df.drop(['satisfaction_satisfied','satisfaction_neutral or dissatisfied'],axis = 1)
X = df.drop('satisfaction', axis = 1)
y = df.satisfaction
X_test = df_test.drop('satisfaction', axis = 1)
y_test = df_test.satisfaction


# In[68]:


#clf = DecisionTreeClassifier()
#clf = clf.fit(X,y)
#y_pred = clf.predict(X_test)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[69]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap = 'GnBu')
plt.show()


# In[71]:


#from sklearn.ensemble import RandomForestClassifier

#rfc = RandomForestClassifier(n_estimators = 100) 
 
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
#rfc.fit(X, y)
 
# performing predictions on the test dataset
#y_pred_rfc = rfc.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred_rfc))


# In[72]:


cm = confusion_matrix(y_test, y_pred_rfc)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap = 'GnBu')
plt.show()

