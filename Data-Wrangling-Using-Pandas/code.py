# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 # code starts here
filepath = path
bank=pd.read_csv(filepath)
categorical_var=bank.select_dtypes(include = 'object')
numerical_var=bank.select_dtypes(include='number')
print(numerical_var)








# code ends here


# --------------
# code starts here
#load the dataset and drop the Loan_ID
banks=bank.drop(['Loan_ID'], axis=1)
#check all the missing values filled
print(banks.isnull().sum())
#apply mode
bank_mode=banks.mode().iloc[0]
print(banks.head())
# Fill the missing values with 

banks.fillna(bank_mode, inplace=True)

# check again all the missing values filled.

print(banks.isnull().sum())




# --------------
# Code starts here
import pandas as pandas
import numpy as np

avg_loan_amount=pd.pivot_table(banks, values='LoanAmount', index=['Gender', 'Married', 'Self_Employed'], aggfunc=np.mean)

print(avg_loan_amount)




# code ends here



# --------------
# code starts here

# code for loan aprroved for self employed
loan_approved_se = banks.loc[(banks["Self_Employed"]=="Yes")  & (banks["Loan_Status"]=="Y"), ["Loan_Status"]].count()
print(loan_approved_se)

# code for loan approved for non self employed
loan_approved_nse = banks.loc[(banks["Self_Employed"]=="No")  & (banks["Loan_Status"]=="Y"), ["Loan_Status"]].count()
print(loan_approved_nse)

# percentage of loan approved for self employed
percentage_se = (loan_approved_se * 100 / 614)
percentage_se=percentage_se[0]
# print percentage of loan approved for self employed
print(percentage_se)

#percentage of loan for non self employed
percentage_nse = (loan_approved_nse * 100 / 614)
percentage_nse=percentage_nse[0]
#print percentage of loan for non self employed
print (percentage_nse)

# code ends here


# --------------
# code starts here


# loan amount term 

loan_term = banks['Loan_Amount_Term'].apply(lambda x: int(x)/12 )


big_loan_term=len(loan_term[loan_term>=25])

print(big_loan_term)

# code ends here


# --------------
# code starts here

columns_to_show = ['ApplicantIncome', 'Credit_History']
 
loan_groupby=banks.groupby(['Loan_Status'])

loan_groupby=loan_groupby[columns_to_show]

# Check the mean value 
mean_values=loan_groupby.agg([np.mean])

print(mean_values)

# code ends here


