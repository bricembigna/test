pip install gspread google-auth





import gspread
from google.oauth2.service_account import Credentials

key_file_path = "/Users/briceoliviermbignambakop/Downloads/prova-api-cs-27ad55beccc7.json"

scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

credentials = Credentials.from_service_account_file(key_file_path, scopes=scopes)

client = gspread.authorize(credentials)




sheet = client.open("Dataset").sheet1

data = sheet.get_all_records()






pip install matplotlib seaborn







import matplotlib.pyplot as plt
import pandas as pd

data = sheet.get_all_records()

df = pd.DataFrame(data)

department_counts = df['Department'].value_counts()

percentages = department_counts / department_counts.sum() * 100

plt.figure(figsize=(10, 6))
plt.pie(percentages, labels=percentages.index, autopct='%1.1f%%', startangle=140)
plt.title('Percentage of Employees by Department')
plt.axis('equal')
plt.show()








import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = sheet.get_all_records()

df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))

sns.violinplot(x='Gender', y='Age', data=df, hue='Gender', split=True)

plt.title('Age Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Age')

plt.show()

#SimonK to check whether we can unite two sides closer
#Brice to look it up









import matplotlib.pyplot as plt
import pandas as pd

data = sheet.get_all_records()

df = pd.DataFrame(data)

bins = list(range(18, 61))

plt.figure(figsize=(12, 6))
plt.hist(df['Age'], bins=bins, edgecolor='white', color='skyblue', alpha=0.7, align='left')

plt.title('Age Distribution Histogram (Individual Ages 18 to 60)')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.xticks(bins)

plt.show()










import matplotlib.pyplot as plt

department_counts = df['Department'].value_counts()

plt.figure(figsize=(10, 6))
department_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Employees by Department')
plt.xlabel('Department')
plt.ylabel('Number of Employees')
plt.xticks(rotation=45)
plt.show()






mean_income = df['MonthlyIncome'].mean()
median_income = df['MonthlyIncome'].median()
std_income = df['MonthlyIncome'].std()

print(f"Mean Monthly Income: ${mean_income:.2f}")
print(f"Median Monthly Income: ${median_income:.2f}")
print(f"Standard Deviation of Monthly Income: ${std_income:.2f}")










# Count each unique category in BusinessTravel
business_travel_counts = df['BusinessTravel'].value_counts()

# Plotting the BusinessTravel distribution as a bar chart
plt.figure(figsize=(8, 5))
business_travel_counts.plot(kind='bar', color='lightcoral', edgecolor='black')
plt.title('Business Travel Frequency')
plt.xlabel('Business Travel Category')
plt.ylabel('Number of Employees')
plt.xticks(rotation=45)
plt.show()








plt.figure(figsize=(8, 8))
plt.pie(business_travel_counts, labels=business_travel_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Business Travel Distribution')
plt.axis('equal')
plt.show()






import seaborn as sns
import matplotlib.pyplot as plt

# Plotting the distribution of JobRole using a violin plot
plt.figure(figsize=(12, 8))
sns.violinplot(y='JobRole', x='MonthlyIncome', data=df, scale='count', inner='stick')
plt.title('Distribution of Monthly Income by Job Role')
plt.xlabel('Monthly Income')
plt.ylabel('Job Role')
plt.show()









# Plotting the distribution of JobRole using a boxen plot
plt.figure(figsize=(12, 8))
sns.boxenplot(y='JobRole', x='MonthlyIncome', data=df)
plt.title('Distribution of Monthly Income by Job Role')
plt.xlabel('Monthly Income')
plt.ylabel('Job Role')
plt.show()


#if you show this one show an explanation on how to read it
#reasoning behind is that it gives more information. Make sure to both describe (box on streamlit) + in documentation of source code, explain
#options and why we came out with this decision






import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the distribution of JobRole using a standard box plot
plt.figure(figsize=(12, 8))
sns.boxplot(y='JobRole', x='MonthlyIncome', data=df)
plt.title('Distribution of Monthly Income by Job Role')
plt.xlabel('Monthly Income')
plt.ylabel('Job Role')
plt.show()


#out, use for something else









plt.figure(figsize=(10, 6))
sns.scatterplot(x='YearsAtCompany', y='YearsSinceLastPromotion', hue='Attrition', data=df, palette='coolwarm', alpha=0.7)
plt.title('Years at Company vs. Years Since Last Promotion (Attrition Highlighted)')
plt.xlabel('Years at Company')
plt.ylabel('Years Since Last Promotion')
plt.legend(title='Attrition')
plt.show()


# kill







plt.figure(figsize=(12, 8))
sns.violinplot(y='JobRole', x='WorkLifeBalance', data=df, inner='quartile')
plt.title('Distribution of Work-Life Balance by Job Role')
plt.xlabel('Work-Life Balance')
plt.ylabel('Job Role')
plt.show()


# check whether it makes sense to convert WLB from 1-4 into 1-10 and make sure distribution by JobRole is not the same for every one.
# Makes data more interesting









