# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(path)

p_a=df[df['fico'].astype(float) >700].shape[0]/df.shape[0]
print(p_a)
# probability of purpose == debt_consolidation
p_b = df[df['purpose']== 'debt_consolidation'].shape[0]/df.shape[0]
print(p_b)

# Create new dataframe for condition ['purpose']== 'debt_consolidation' 
df1 = df[df['purpose']== 'debt_consolidation']

# Calculate the P(A|B)
p_a_b = df1[df1['fico'].astype(float) >700].shape[0]/df1.shape[0]
print(p_a_b)
# Check whether the P(A) and P(B) are independent from each other
result = (p_a == p_a_b)
print(result)


# --------------
# code starts here
prob_lp=df[df['paid.back.loan'] == 'Yes'].shape[0] /df.shape[0]
prob_cs=df[df['credit.policy'] == 'Yes'].shape[0] /df.shape[0]

new_df=df[df['paid.back.loan'] == 'Yes']
prob_pd_cs = new_df[new_df['credit.policy'] == 'Yes'].shape[0] /new_df.shape[0]
bayes=(prob_pd_cs * prob_lp)/ prob_cs
print(bayes)
#print(df[df['paid.back.loan'] == 'Yes'].shape[0])
#print(new_df.head())

# code ends here


# --------------
# code starts here
df.purpose.value_counts(normalize=True).plot(kind='bar')
df1=df[df['paid.back.loan']=='No']
df1.plot(kind='bar')
# code ends here


# --------------
# code starts here
inst_median=df.installment.median()
inst_mean=df.installment.mean()
print('ins_median',inst_median)
print('inst_mean',inst_mean)
print(df.columns)
df.installment.plot.hist(bins=12, alpha=0.5)
df['log.annual.inc'].plot.hist(bins=12, alpha=0.5)
# code ends here


