import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn.linear_model import LinearRegression

#time series
#pesticide = kg/pc
#fertilizer = kg/pc
#change in tmp = c with respect to 1951-1980 baseline
#average rainfall = inches
#emission = MMT CO2 eq
#crop Yield = index per capitia such that 2014-2016 = 100
pesticide_time_series = np.array([1.53	,	1.57	,	1.51	,	1.62	,	1.59	,	1.61	,	1.6	,	1.54	,	1.55	,	1.52	,	1.44	,	1.45	,	1.44	,	1.45	,	1.31	,	1.31	,	1.32	,	1.24	,	1.16	,	1.2	,	1.25	,	1.29	,	1.31	,	1.28	,	1.3	,	1.38	,	1.36	,	1.38	,	1.37	,	1.36,1.36])
fertilizer_time_series = np.array([74.4	,	74.17	,	76.89	,	74.96	,	74.46	,	75.12	,	74.24	,	72	,	70.27	,	67.53	,	67.67	,	67.11	,	72.83	,	68.35	,	64.93	,	69.31	,	64.33	,	52.46	,	61.32	,	63.2	,	64.88	,	66.03	,	66.02	,	62.04	,	62.26	,	61.41	,	61.09	,	61.34	,	60.79	,	61.59	,	60.26])
temp_time_series = np.array([51.8, 53.6, 50.3, 54.7, 53.9, 54.3, 53.4, 52.9, 53.3, 53.7, 53.8, 52.0, 55.0, 52.1, 53.4, 53.8, 53.8, 51.9, 52.1, 52.7, 51.8, 56.6, 53.3, 55.6, 56.3, 56.2, 56.1, 56.2, 53.6, 55.7, 56.3])
rainfall_time_series = np.array([32.44, 31.26, 32.62, 30.62, 32.69 ,33.7, 31.86, 33.89, 28.47, 28.22, 29.02, 29.05, 30.51, 33.25, 30.08, 29.82, 29.18, 31.24, 32.3, 31.37, 30.1, 27.53, 31.06, 30.85, 34.59, 31.42, 32.31, 34.65, 34.82, 30.38, 30.42])
emissions_time_series = np.array([6418.40619, 6534.880127, 6639.525505, 6745.166791, 6821.933718, 7023.529629, 7079.206863, 7124.684268, 7169.885428, 7369.162972, 7253.522308, 7293.277737, 7352.702361, 7464.364918, 7477.358361, 7407.921495, 7511.447958, 7294.541719, 6840.741542, 7058.197908, 6907.203236, 6670.533532, 6841.661093, 6898.525596, 6737.358644, 6578.432362, 6561.82444, 6754.831648, 6617.916876, 6025.973613, 6340.228292])
corn_time_series = np.array([65.72, 82.14, 54.17, 84.75, 61.61, 75.91, 74.75, 78.25, 74.71, 77.65, 73.62, 68.78, 76.64, 88.83, 82.79, 77.7, 95.25, 87.13, 93.68, 88.31, 86.77, 75.12, 95.77, 97.63, 92.67, 109.7, 97.97, 95.49, 90.1, 92.9, 99.2])

#functions used to calculate for explantory analysis
def summation(x):
    total = 0
    for i in range(len(x)):
        total = total + x[i]
    return(total)

def average(x):
    total = 0
    for i in range(len(x)):
        total = total + x[i]
    average = total / (len(x))
    return average

def center_func(x):
    series = np.array([])
    av = average(x)
    for i in range(len(x)):
        newval = x[i] - av
        series = np.append(series , newval)
    return series

#fuctions to return the percent of total variability of the data
def perTot(x):
    percent_list = []
    for i in range(len(x)):
        total = summation(x)
        percentage = (x[i]/total)*100
        percent_list.append(percentage)
    return percent_list

def perTota(x):
    total2 = 0
    for i in range(len(x)):
        total = summation(x)
        percentage = (x[len(x)-i-1]/total)*100
        total2 = total2 + percentage
        print(total2)
       
#standardizing the time series
center_corn = center_func(corn_time_series)
center_pesticide = center_func(pesticide_time_series)
center_fertilizer = center_func(fertilizer_time_series)
center_temp = center_func(temp_time_series)
center_rainfall = center_func(rainfall_time_series)
center_emissions = center_func(emissions_time_series)

corn_standard = center_corn / np.std(center_corn)
pesticide_standard = center_corn / np.std(center_corn)
fertilizer_standard = center_corn / np.std(center_corn)
temp_standard = center_corn / np.std(center_corn)
rainfall_standard = center_corn / np.std(center_corn)
emissions_standard = center_corn / np.std(center_corn)

#first model

#covariance and correlation matrices
raw_data1 = np.vstack((corn_time_series, pesticide_time_series, fertilizer_time_series, temp_time_series, rainfall_time_series, emissions_time_series))
raw_data_T1 = raw_data1.transpose()
data1 = np.vstack((center_corn, center_pesticide, center_fertilizer, center_temp, center_rainfall, center_emissions))
data_std1 = np.vstack((corn_standard, pesticide_standard, fertilizer_standard, temp_standard, rainfall_standard, emissions_standard))
data_std_T1 = data_std1.transpose()
print(data_std_T1)
cov_matrix1 = np.cov(data1)
corr_matrix1 = np.corrcoef(data1)
print(cov_matrix1)
print(corr_matrix1)

t1,v1=eig(cov_matrix1)
idx = np.argsort(t1)
t1 = t1[idx]
v1 = v1[:,idx]
print('E-value:', t1)
print('E-vector', v1)
PC_list = ["PC6","PC5","PC4","PC3","PC2","PC1"]



plt.bar(PC_list, perTot(t1), color ='black',
        width = 0.4)
 
plt.xlabel("Principal Components")
plt.ylabel("Percentage")
plt.title("PC Plot")
plt.show()

#correlation PCA
print('Corr:', corr_matrix1)
u1,w1=eig(corr_matrix1)
idx = np.argsort(u1)
u1 = u1[idx]
w1 = w1[:,idx]
print('E-value:', u1)
print('E-vector', w1)
PC_list = ["PC6","PC5","PC4","PC3","PC2","PC1"] 
print(perTot(u1))
plt.bar(PC_list, perTot(u1), color ='black',
        width = 0.4)
plt.xlabel("Principal Components")
plt.ylabel("Percentage")
plt.title("PC Plot")

plt.show()

#regression
#4 principal components
print(w1[:, [2, 3, 4, 5]])
principal_comp1 = np.matmul(data_std_T1, w1)
regr_comp1 = principal_comp1[:, [2, 3, 4, 5]]
print('PCs:' , principal_comp1)
print('Regr Comp:',regr_comp1)

model1 = LinearRegression().fit(regr_comp1, corn_standard)
r_sq = model1.score(regr_comp1, corn_standard)
print(r_sq)
adj = 1 - ((1 - r_sq)*(len(corn_time_series) - 1)/(len(corn_time_series) - 5 - 1))  
print(adj)
#second model
raw_data = np.vstack((corn_time_series, pesticide_time_series, fertilizer_time_series, temp_time_series))
raw_data_T = raw_data.transpose()
data = np.vstack((center_corn, center_pesticide, center_fertilizer, center_temp))
data_std = np.vstack((corn_standard, pesticide_standard, fertilizer_standard, temp_standard))
data_std_T = data_std.transpose()
print(data_std_T)
cov_matrix = np.cov(data)
corr_matrix = np.corrcoef(data)
print(cov_matrix)
print(corr_matrix)
#covariance PCA 
t,v=eig(cov_matrix)
idx = np.argsort(t)
t = t[idx]
v = v[:,idx]
print('E-value:', t)
print('E-vector', v)
PC_list = ["PC4","PC3","PC2","PC1"]
plt.bar(PC_list, perTot(t), color ='black',
        width = 0.4)
plt.xlabel("Principal Components")
plt.ylabel("Percentage")
plt.title("PC Plot")
plt.show()

#correlation PCA
print('Corr:', corr_matrix)
u,w=eig(corr_matrix)
idx = np.argsort(u)
u = u[idx]
w = w[:,idx]
print('E-value:', u)
print('E-vector', w)
perTota(u)

PC_list = ["PC4","PC3","PC2","PC1"] 

print(perTot(u))
plt.bar(PC_list, perTot(u), color ='black',
        width = 0.4)
 
plt.xlabel("Principal Components")
plt.ylabel("Percentage")
plt.title("PC Plot")

plt.show()

#regression
#2 principal componenets
print(w[:, [2, 3]])
principal_comp = np.matmul(raw_data_T, w)
regr_comp = principal_comp[:, [2, 3]]
print('PCs:' , principal_comp)
print('Regr Comp:',regr_comp)

model = LinearRegression().fit(regr_comp, corn_time_series)
r_sq = model.score(regr_comp, corn_time_series)
adj = 1 - ((1 - r_sq)*(len(corn_time_series) - 1)/(len(corn_time_series) - 3 - 1)) 
print(adj) 

#Explantory analysis results
print(corn_standard)
print(r_sq)
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")


print("Corn")
print("average:", average(corn_time_series))
print("median:", np.median(corn_time_series))
print("mininum:", np.min(corn_time_series))
print("maximum:", np.max(corn_time_series))
print("standard deviation:", np.std(corn_time_series))

print("Pesticide")
print("average:", average(pesticide_time_series))
print("median:", np.median(pesticide_time_series))
print("mininum:", np.min(pesticide_time_series))
print("maximum:", np.max(pesticide_time_series))
print("standard deviation:", np.std(pesticide_time_series))

print("Fertilizer")
print("average:", average(fertilizer_time_series))
print("median:", np.median(fertilizer_time_series))
print("mininum:", np.min(fertilizer_time_series))
print("maximum:", np.max(fertilizer_time_series))
print("standard deviation:", np.std(fertilizer_time_series))

print("Temperature")
print("average:", average(temp_time_series))
print("median:", np.median(temp_time_series))
print("mininum:", np.min(temp_time_series))
print("maximum:", np.max(temp_time_series))
print("standard deviation:", np.std(temp_time_series))

print("Rainfall")
print("average:", average(rainfall_time_series))
print("median:", np.median(rainfall_time_series))
print("mininum:", np.min(rainfall_time_series))
print("maximum:", np.max(rainfall_time_series))
print("standard deviation:", np.std(rainfall_time_series))

print("Emissions")
print("average:", average(emissions_time_series))
print("median:", np.median(emissions_time_series))
print("mininum:", np.min(emissions_time_series))
print("maximum:", np.max(emissions_time_series))
print("standard deviation:", np.std(emissions_time_series))

 

