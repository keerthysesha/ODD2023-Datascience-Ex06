# Feature_Transformation
# Ex-06-Feature-Transformation
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
STEP 1:
Read the given Data

STEP 2:
Clean the Data Set using Data Cleaning Process

STEP 3:
Apply Feature Transformation techniques to all the features of the data set

STEP 4:
Print the transformed features

# PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```
# OUTPUT:


![image](https://github.com/keerthysesha/ODD2023-Datascience-Ex06/assets/125575936/bbf26653-0de7-41eb-95b8-2d2d09d42087)


![image](https://github.com/keerthysesha/ODD2023-Datascience-Ex06/assets/125575936/5db2e761-2d68-4255-b920-f77fbcbbec48)


![image](https://github.com/keerthysesha/ODD2023-Datascience-Ex06/assets/125575936/f9d1f403-a250-48f7-af63-934e5063d1ed)


![image](https://github.com/keerthysesha/ODD2023-Datascience-Ex06/assets/125575936/1e880df2-6a02-4440-a168-874ce102ba35)

![image](https://github.com/keerthysesha/ODD2023-Datascience-Ex06/assets/125575936/4911ea8a-9e5a-4e73-b4ba-9294f6a8bdf8)


![image](https://github.com/keerthysesha/ODD2023-Datascience-Ex06/assets/125575936/ace4119d-5bed-46cb-80dd-2436198e585a)


![image](https://github.com/keerthysesha/ODD2023-Datascience-Ex06/assets/125575936/7c9a4511-6b68-41e1-be05-f9e9a7eb309f)


![image](https://github.com/keerthysesha/ODD2023-Datascience-Ex06/assets/125575936/f452490d-161d-4bd8-b8ca-100188ff91f7)

![image](https://github.com/keerthysesha/ODD2023-Datascience-Ex06/assets/125575936/bc2f865a-df33-438c-ad0e-90d43bbf7cf8)


![image](https://github.com/keerthysesha/ODD2023-Datascience-Ex06/assets/125575936/e6914d0d-8255-4878-923b-362e922eb7d9)

# RESULT:
Thus feature transformation is done for the given dataset.
