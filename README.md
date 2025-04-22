## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
  ```
import pandas as pd
df=pd.read_csv("/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/05c48d30-97ae-44bb-956d-d2656e54415d)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/71cbdab3-06aa-4086-af76-5f1909c589eb)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/47692277-fe76-4d77-8c97-c0afe6507fc9)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/9824ea72-2d26-4d3b-bfee-108b04f952d7)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/a522bbf9-0083-481d-bd6d-4a3ebb6f502a)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/9b36cff0-8c82-4d90-9bb1-7a15f17a4539)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/1eac5578-f286-47dd-bdfa-aed56a832ba6)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/data.csv")
df
```
![image](https://github.com/user-attachments/assets/12d64b72-ea5c-4f26-ba42-3f5765fc3e1f)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/975a48d5-650e-41b1-b877-a85fc92d8141)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/0ae38015-0c1a-4307-a425-2ce8a8a543d0)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/95aa2699-3e6d-4a32-9abb-bc593a73f9a7)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/7552f1d6-20a8-4e1b-bc39-64a0caa94345)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/de3473e8-b526-48dc-a460-e23f5ab4b9c4)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/4a28be5d-5640-4ccb-8dfc-bfa0a676b002)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/ae870465-9806-45a4-a573-4d7d7ab809a4)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/f93083c6-df6a-4948-94d6-f6b20718631f)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/7881a13f-32d5-4496-8aaa-b469c80b4a00)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/9ee33bbb-1df2-485d-86c7-35bef97fd9e6)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/33e989eb-ab51-4aaf-a96b-0250ae0d1ad7)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/8f9d8423-9d17-4f9e-bd9e-aca3d0fd1f9d)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/dd25d00e-95a8-4613-b431-760bccff6169)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/89dcbb4d-6112-4178-bf79-c3379a641b1c)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/75f034f1-0942-48e5-a4cd-0ee1a70e5e3b)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/52a1861f-9339-4141-8def-ecab4dfd5894)

# RESULT:
    Thus, performing Feature Encoding and Transformation process for the given data set is completed.

       
