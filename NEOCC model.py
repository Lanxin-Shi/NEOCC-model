#!/usr/bin/env python
# coding: utf-8

# In[1]:


#/usr/bin/env python
#coding: utf-8


# In[2]:


import pandas as pd
import numpy as np
import numpy_financial as npf
import math
from scipy.stats import gamma
import os
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# In[3]:


Vehicle_Policy_Scenario = ["Dual-credit Policy", "CAFC Policy Only", "NEV Policy Only", "No Rules"]


# In[4]:


Year = int(2023)


# In[5]:


BEV_FC = "No"


# In[6]:


PEF=8.797811


# "Model" Input

# In[7]:


df_model = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Model.xlsx")

df_model_personal_vehicle_market = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Model.xlsx", sheet_name="Personal Vehicle Market")
df_model_fleet_vehicle_market = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Model.xlsx", sheet_name="Fleet Vehicle Market")

df_model_calibration_total_sales_market = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Model.xlsx", sheet_name="Calibration", index_col=0, 
                                                        usecols="A:E", skiprows=1, nrows=19)
df_model_calibration_personal_vehicle_market = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Model.xlsx", sheet_name="Calibration", index_col=0, 
                                                        usecols=[0, 5, 6, 7, 8], skiprows=1, nrows=19)
df_model_calibration_fleet_vehicle_market = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Model.xlsx", sheet_name="Calibration", index_col=0, 
                                                        usecols=[0, 9, 10, 11, 12], skiprows=1, nrows=19)

df_model_calibration_GAP_diversity_Prodcost = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Model.xlsx", sheet_name="Gap, Diversity, ProdCost")
df_model_vehicle_product_cost = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Model.xlsx", sheet_name="Vehicle Product Cost")


# "Projection" Input

# In[8]:


df_projection = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\projection.xlsx")


# "Constant" Input

# In[9]:


df_constant_personal_vehicles = pd.read_excel(
    r"D:\Users\11379\Desktop\博士\博0\NEOCC model\constant.xlsx",
    usecols = "B:P",
    skiprows = 2,
    nrows = 37
) 

df_constant_fleet_vehicles = pd.read_excel(
    r"D:\Users\11379\Desktop\博士\博0\NEOCC model\constant.xlsx",
    usecols = "Q:AE",
    skiprows = 2,
    nrows = 37
)

new_columns_personal = [
    'Sedan' if 'Sedan' in col else
    'SUV&Crossover&MPV' if 'SUV&Crossover&MPV' in col else col 
    for col in df_constant_personal_vehicles.columns
]

new_columns_fleet = [
    'Sedan' if 'Sedan' in col else
    'SUV&Crossover&MPV' if 'SUV&Crossover&MPV' in col else col 
    for col in df_constant_fleet_vehicles.columns
]
df_constant_personal_vehicles.columns = new_columns_personal
df_constant_fleet_vehicles.columns = new_columns_fleet


# "Consumer" Input

# In[10]:


df_consumer = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Consumer.xlsx", usecols="A:C", skiprows=1, nrows=18)
df_consumer = df_consumer.transpose()
new_header = df_consumer.iloc[0] 
df_consumer = df_consumer[1:]  
df_consumer.columns = new_header


# In[11]:


def Gamma_Scale_theta_Beta(row):
    return row['Driving - mean (km)'] - row['Driving - mode (km)']

def Gamma_Shape_kAlpha(row):
    if row['Gamma-Scale-thetaBeta'] == 0:
        return None  # Avoid division by zero
    return row['Driving - mean (km)'] / row['Gamma-Scale-thetaBeta']

def Lifetime_Kilometers(row):
    # Calculate the annual driving distance
    annual_driving = row['Driving - mean (km)'] * 365
    # Calculate present value
    pv = npf.pv(rate = row['Discount rate'], nper = row['Vehicle lifetime'], pmt = -annual_driving)
    return pv


# In[12]:


df_consumer['Gamma-Scale-thetaBeta'] = df_consumer.apply(Gamma_Scale_theta_Beta, axis=1)
df_consumer['Gamma-Shape-kAlpha'] = df_consumer.apply(Gamma_Shape_kAlpha, axis=1)
df_consumer['Lifetime kilometers'] = df_consumer.apply(Lifetime_Kilometers, axis=1)


# In[13]:


df_DCM_personal_vehicle = pd.DataFrame({
    '4_Technology_IC': -0.000778996634356637/df_consumer[('￥/$ in 2020')],
    '4_Technology_PH': -0.000545183351593316/df_consumer[('￥/$ in 2020')],
    '4_Technology_EV': -0.000476934455320167/df_consumer[('￥/$ in 2020')],
    '4_Technology_FC': -0.000356062789728483/df_consumer[('￥/$ in 2020')],
    
    '3_Powertrain_Conv': -0.00034423179570962/df_consumer[('￥/$ in 2020')],
    '3_Powertrain_Electricity': -0.000247265826200335/df_consumer[('￥/$ in 2020')],
    '3_Powertrain_Hydrogen': -0.000247265826200335/df_consumer[('￥/$ in 2020')],

    '2_Class_Sedan': -0.000173861272935794/df_consumer[('￥/$ in 2020')],
    '2_Class_SUV/Crossover': -0.000173861272935794/df_consumer[('￥/$ in 2020')],

    '1_Class': -0.000173861272935794/df_consumer[('￥/$ in 2020')],

    'Buy/NoBuy': -0.000154509525/df_consumer['GDP per capital ($2020 USD)'].apply(lambda x: np.log(x) if pd.notnull(x) and x != 0 else np.nan),
}) 

pd.set_option('display.float_format', '{:.4e}'.format)

df_DCM_personal_vehicle = df_DCM_personal_vehicle.iloc[0:1] 


# In[14]:


df_DCM_fleet_vehicle = pd.DataFrame({
    '4_Technology_IC': (-0.000778996634356637/df_consumer[('￥/$ in 2020')])*1.01,
    '4_Technology_PH': (-0.000545183351593316/df_consumer[('￥/$ in 2020')])*1.01,
    '4_Technology_EV': (-0.000476934455320167/df_consumer[('￥/$ in 2020')])*1.01,
    '4_Technology_FC': (-0.000356062789728483/df_consumer[('￥/$ in 2020')])*1.01,
    
    '3_Powertrain_Conv': -0.00034423179570962/df_consumer[('￥/$ in 2020')],
    '3_Powertrain_Electricity': -0.000247265826200335/df_consumer[('￥/$ in 2020')],
    '3_Powertrain_Hydrogen': -0.000247265826200335/df_consumer[('￥/$ in 2020')],

    '2_Class_Sedan': (-0.000173861272935794/df_consumer[('￥/$ in 2020')])*1.4,
    '2_Class_SUV/Crossover': (-0.000173861272935794/df_consumer[('￥/$ in 2020')])*1.4,
}) 

pd.set_option('display.float_format', '{:.4e}'.format)

df_DCM_fleet_vehicle = df_DCM_fleet_vehicle.iloc[1:2]


# Industry

# In[15]:


df_industry_vehicle_production = pd.DataFrame({
    'Sed FC Gap (L/100km)': df_model_calibration_GAP_diversity_Prodcost['Sed FC Gap (L/100km)'],
    'SUV FC Gap (L/100km)': df_model_calibration_GAP_diversity_Prodcost['SUV FC Gap (L/100km)'],
    'Sed FC Diversity (L/100km)': df_model_calibration_GAP_diversity_Prodcost['Sed FC Diversity (L/100km)'],
    'SUV FC Diversity (L/100km)': df_model_calibration_GAP_diversity_Prodcost['SUV FC Diversity (L/100km)'],
    
    'Sedan Markup': 1.50,
    'SUV Markup': 1.50,
    'PEV/FCEV Markup': df_projection[df_projection['Year'] == Year]['PEV mark-up factor'].values,

    'Fleet Vehicle Purchase Discount': 0.9,
    
    'ICEV Cost Slope Parameter_a': df_projection[df_projection['Year'] == Year]['ICEV Cost Parameter_a'].values,
    'ICEV Cost Slope Parameter_b': df_projection[df_projection['Year'] == Year]['ICEV Cost Parameter_b'].values,
    'ICEV Cost Slope Parameter_c': df_projection[df_projection['Year'] == Year]['ICEV Cost Parameter_c'].values,

    'ICEV MPG High Incremental Level': df_projection[df_projection['Year'] == Year]['ICEV-Low Fuel Cosum Rate'].values,

    'Sed ProdCost A': df_model_calibration_GAP_diversity_Prodcost['Sed ProdCost A'],
    'Sed BEV Base Cost Index': df_projection[df_projection['Year'] == Year]['Sed BEV Base Cost Index'].values,
    'Sed PHEV Base Cost Index': df_projection[df_projection['Year'] == Year]['Sed PHEV Base Cost Index'].values,
    'SUV ProdCost A': df_model_calibration_GAP_diversity_Prodcost['SUV ProdCost A'],
    'SUV BEV Base Cost Index': df_projection[df_projection['Year'] == Year]['SUV BEV Base Cost Index'].values,
    'SUV PHEV Base Cost Index': df_projection[df_projection['Year'] == Year]['SUV PHEV Base Cost Index'].values,

    'Personal purchase share in sales market': (df_model_calibration_personal_vehicle_market.loc['Total sales of sedans', 'History.1'] + 
                           df_model_calibration_personal_vehicle_market.loc['Total sales of SUVs', 'History.1']) / \
                          (df_model_calibration_total_sales_market.loc['Total sales of sedans', 'History'] + 
                           df_model_calibration_total_sales_market.loc['Total sales of SUVs', 'History']),
}) 


# In[16]:


Subsidy_Change = {
    'Vehicle Type': ['Sedan', 'SUV'],
    'Subsidy Change_1': [0, 0],
    'Engine Type_1': ['ICE', 'NEV'],
    'Subsidy Change_2': [0, 0],
    'Model_1': ['tgL&tg', 'tgH'],
    'Subsidy Change_3': [0, 0]
}

df_industry_subsidy_change = pd.DataFrame(Subsidy_Change)


# In[17]:


df_industry = pd.DataFrame({
    'Battery pack costs in 2020 ($/kWh)': 137,
    'Battery cost coefficient by year': df_projection[df_projection['Year'] == Year]['Battery Cost (benchmark = 2017)'].values,
    'Fuel cell system costs in 2020 ($/kW)': 300,
    'Fuel cell system cost coefficient by year': df_projection[df_projection['Year'] == Year]['Fuel Cell System Cost (benchmark=2020)'].values,
    'Annual sales growth rate': 0.02,
    'PEV credit trading efficiency': 0.90,
    'PEV credit multiplier': df_projection[df_projection['Year'] == Year]['NEV Credit multiplier'].values,
    'Electrcity consumption efficiency improvement': df_projection[df_projection['Year'] == Year]['Elec Consum efficiency improvement'].values,
}) 


# "Rules" Input

# In[18]:


df_rules = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Rules.xlsx", usecols="A:K", skiprows=1, nrows=17) 
df_rules_Phase_4_CAFC = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Rules.xlsx", usecols="B:D", skiprows=40, nrows=17)
df_rules_Phase_4_CAFC_WLTC_adjusted = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Rules.xlsx", usecols="B:D", skiprows=21, nrows=17)
df_rules_Phase_5_CAFC_Target = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Rules.xlsx", usecols="F:I", skiprows=21, nrows=4) 
df_rules_Phase_5_CAFC_Limit = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Rules.xlsx", usecols="F:I", skiprows=28, nrows=4) 
df_rules_CAFC_Multiplier = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Rules.xlsx", usecols="K:M", skiprows=21, nrows=9)


# In[19]:


if Year < 2021:
    ICEV_Low_Benefit_coefficient = 1
elif Year == 2021:
    ICEV_Low_Benefit_coefficient = 0.5
elif Year == 2022:
    ICEV_Low_Benefit_coefficient = 0.3
else:
    ICEV_Low_Benefit_coefficient = 0.2 


# In[20]:


WLTC_NEDC_coefficient = 1.1057 if Year <= 2020 else 1


# In[21]:


# Merge the two Phase 4 CAFC DataFrames on the 'CM' column to align rows based on the 'CM' values
merged_df = pd.merge(df_rules_Phase_4_CAFC_WLTC_adjusted, df_rules_Phase_4_CAFC, on='CM', suffixes=('_adjusted', ''))

# Update 'FC Target' and 'FC Limit' in the adjusted DataFrame
merged_df['FC Target_adjusted'] = merged_df['FC Target'] * WLTC_NEDC_coefficient
merged_df['FC Limit_adjusted'] = merged_df['FC Limit'] * WLTC_NEDC_coefficient

# Replace the original df_rules_Phase_4_CAFC_WLTC_adjusted DataFrame with the updated values:
df_rules_Phase_4_CAFC_WLTC_adjusted = merged_df[['CM', 'FC Target_adjusted', 'FC Limit_adjusted']].copy()
df_rules_Phase_4_CAFC_WLTC_adjusted.columns = ['CM', 'FC Target', 'FC Limit']  # Rename columns to original


# “Model” Supplement 1

# In[22]:


def Sales_Weighted_Weight(x):
    equation_results = np.zeros(16)

    equation_results[0] = 1384.88
    equation_results[1] = 1384.88
    equation_results[2] = 1524.76
    equation_results[3] = 1802.83372636263
    equation_results[4:7] = 2.6753 * x[4:7] + 302.85
    equation_results[7] = 1850
    equation_results[8] = 1572.26
    equation_results[9] = 1572.26
    equation_results[10] = 1769.42
    equation_results[11] = 2085.22
    equation_results[12:15] = 1.9163 * x[12:15] + 992.34
    equation_results[15] = 2210
    
    return equation_results

df_model['Sales-weighted Weight (kg)'] = Sales_Weighted_Weight(df_model['Elec Range (km)'])


# In[23]:


def FC_Target(x1, x2, x3):
    equation_results = np.zeros_like(x1)

    x3_interp = np.interp(x1[1], x2, x3)
    idx = np.searchsorted(x2, x1[1]) - 1 
    if idx >= 0 and idx < len(x2) - 1 and x1[1] != x2[idx]:
        x3_interp = x3[idx]

    equation_results[:16] = np.where(
        Year <= 2020,
        x3_interp,
        np.where(
            x1[:16] <= 1090,
            4.02,
            np.where(
                x1[:16] <= 2510,
                0.0018 * (x1[:16] - 1415) + 4.6,
                6.57
            )
        )
    )
   
    return equation_results

df_model['FC Target (L/100km)'] = FC_Target(df_model['Sales-weighted Weight (kg)'], df_rules_Phase_4_CAFC_WLTC_adjusted['CM'], df_rules_Phase_4_CAFC_WLTC_adjusted['FC Target'])


# In[24]:


def EC_CD(x):
    equation_results = np.zeros(16)

    equation_results[:3] = 0
    equation_results[3] = 14.27 * x.iloc[0]
    equation_results[4] = 9.65354771871201 * x.iloc[0]
    equation_results[5] = 12.0928129610104 * x.iloc[0]
    equation_results[6] = 13.45 * x.iloc[0]
    equation_results[7:11] = 0
    equation_results[11] = 16.57 * x.iloc[0]
    equation_results[12] = 13.72 * x.iloc[0]
    equation_results[13] = 13.28 * x.iloc[0]
    equation_results[14] = 15.33 * x.iloc[0]
    equation_results[15] = 0
    
    return equation_results

df_model['EC CD (kWh/100km)'] = EC_CD(df_industry['Electrcity consumption efficiency improvement'])


# In[25]:


def FC_CS_Gasoline_Equivalent(x):
    equation_results = np.zeros(16)

    equation_results[:3] = x[:3]
    equation_results[3] = 5.50579117241379
    equation_results[4:7] = 0
    equation_results[7:11] = x[7:11]
    equation_results[11] = 7.794976
    equation_results[12:15] = 0
    equation_results[15] = x[15]
    
    return equation_results

df_model['FC CS Gasoline Equivalent (L/100km)'] = FC_CS_Gasoline_Equivalent(df_model['FC Achieved Gasoline Equivalent (L/100km)'])


# In[26]:


def FC_Achieved_Gasoline_Equivalent(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14):
    equation_results = np.zeros(16)

    x3_interp = np.interp(x1[1], x2, x3)
    idx = np.searchsorted(x2, x1[1]) - 1 
    if idx >= 0 and idx < len(x2) - 1 and x1[1] != x2[idx]:
        x3_interp = x3[idx]
    equation_results[1] = (x3_interp + x4.iloc[0]) * (1 - (1 - np.minimum(x5.iloc[0], 1)) ** 1.5)
    
    equation_results[0] = equation_results[1] / (1 - (1 - np.minimum(x5.iloc[0], 1)) ** 1.5) * (1 - (1 - np.minimum(x5.iloc[0], 1)) ** 1.8) + x6.iloc[0]
    equation_results[2] = np.maximum(4.17, equation_results[1] - x6.iloc[0]) * x5.iloc[0]
    
    equation_results[3] = np.where(
        Year > 2020,
        np.where(
            BEV_FC == "Yes",
            (x7[3] * 25 + x8[3] * x9[3] / x10.iloc[0] / PEF) / (25 + x9[3] / x10.iloc[0]),
             x7[3] * 25 / (25 + x9[3] / x10.iloc[0])
        ),
        1.48
    )

    equation_results[4:7] = np.where(Year > 2020, np.where(BEV_FC == "Yes", x8[4:7] / PEF, 0), 0)

    equation_results[7] = 3.886 * np.interp(Year, x12, x11)
    
    x3_interp = np.interp(x1[9], x2, x3)
    idx = np.searchsorted(x2, x1[9]) - 1 
    if idx >= 0 and idx < len(x2) - 1 and x1[9] != x2[idx]:
        x3_interp = x3[idx]
    equation_results[9] = (x3_interp + x13.iloc[0]) * (1 - (1 - np.minimum(x5.iloc[0], 1)) ** 1.5)

    equation_results[8] = equation_results[9] / (1 - (1 - np.minimum(x5.iloc[0], 1)) ** 1.5) * (1 - (1 - np.minimum(x5.iloc[0], 1)) ** 1.8) + x14.iloc[0]
    equation_results[10] = np.maximum(5.3, equation_results[9] - x14.iloc[0]) * x5.iloc[0]

    equation_results[11] = np.where(
        Year > 2020,
        np.where(
            BEV_FC == "Yes",
            (x7[11] * 25 + x8[3] * x9[11] / x10.iloc[0] / PEF) / (25 + x9[11] / x10.iloc[0]),
             x7[11] * 25 / (25 + x9[11] / x10.iloc[0])
        ),
        1.48
    )

    equation_results[12:15] = np.where(Year > 2020, np.where(BEV_FC == "Yes", x8[12:15] / PEF, 0), 0)

    equation_results[15] = 4.556 * np.interp(Year, x12, x11)
    
    return equation_results

df_model['FC Achieved Gasoline Equivalent (L/100km)'] = FC_Achieved_Gasoline_Equivalent(df_model['Sales-weighted Weight (kg)'],
                                                                                        df_rules_Phase_4_CAFC_WLTC_adjusted['CM'],
                                                                                        df_rules_Phase_4_CAFC_WLTC_adjusted['FC Target'],
                                                                                        df_industry_vehicle_production['Sed FC Gap (L/100km)'],
                                                                                        df_industry_vehicle_production['ICEV MPG High Incremental Level'],
                                                                                        df_industry_vehicle_production['Sed FC Diversity (L/100km)'],
                                                                                        df_model['FC CS Gasoline Equivalent (L/100km)'],
                                                                                        df_model['EC CD (kWh/100km)'],
                                                                                        df_model['Elec Range (km)'],
                                                                                        df_industry['Electrcity consumption efficiency improvement'],
                                                                                        df_projection['Hydrogen Consum efficiency improvement'],
                                                                                        df_projection['Year'],
                                                                                        df_industry_vehicle_production['SUV FC Gap (L/100km)'],
                                                                                        df_industry_vehicle_production['SUV FC Diversity (L/100km)'])


# In[27]:


def CAFC_Multiplier(x1, x2):
    equation_results = np.zeros(16)

    equation_results[:3] = np.where(
        x1[:3] <= 3.2,
        np.interp(Year, x2.iloc[:, 0], x2.iloc[:, 2]),
        1
    )

    equation_results[3:8] = np.interp(Year, x2.iloc[:, 0], x2.iloc[:, 1])

    equation_results[8:10] = np.where(
        x1[:2] <= 3.2,
        np.interp(Year, x2.iloc[:, 0], x2.iloc[:, 2]),
        1
    )

    equation_results[10:16] = np.interp(Year, x2.iloc[:, 0], x2.iloc[:, 1])
    
    return equation_results

df_model['CAFC Multiplier'] = CAFC_Multiplier(df_model['FC Achieved Gasoline Equivalent (L/100km)'], df_rules_CAFC_Multiplier)


# In[28]:


def Miles_Gallon(x):
    equation_results = np.zeros(16)

    equation_results[:4] = 235.215 / x[:4]
    equation_results[4] = 0
    equation_results[5] = 0
    equation_results[6] = 0
    equation_results[7:12] = 235.215 / x[7:12]
    equation_results[12] = 0
    equation_results[13] = 0
    equation_results[14] = 0
    equation_results[15] = 235.215 / x[15]
    
    return equation_results

df_model['Miles/Gallon'] = Miles_Gallon(df_model['FC Achieved Gasoline Equivalent (L/100km)'])


# In[29]:


def Battey_Fuel_Cell_Sytem(x):
    equation_results = np.zeros(16)

    equation_results[0] = 0
    equation_results[1] = 0
    equation_results[2] = 0
    equation_results[3] = 13.49 * x.iloc[0]
    equation_results[4] = 15.96 * x.iloc[0]
    equation_results[5] = 47.59 * x.iloc[0]
    equation_results[6] = 83 * x.iloc[0]
    equation_results[7] = 128
    equation_results[8] = 0
    equation_results[9] = 0
    equation_results[10] = 0
    equation_results[11] = 20 * x.iloc[0]
    equation_results[12] = 36 * x.iloc[0]
    equation_results[13] = 47 * x.iloc[0]
    equation_results[14] = 74 * x.iloc[0]
    equation_results[15] = 128

    return equation_results

df_model['Battey (KWh)/Fuel Cell Sytem (kW)'] = Battey_Fuel_Cell_Sytem(df_industry['Electrcity consumption efficiency improvement'])


# In[30]:


df_model['Internal Subsidy to Consumer(￥)'] = 0


# In[31]:


def FC_CS_Gasoline_Equivalent(x):
    equation_results = np.zeros(16)

    equation_results[:3] = x[:3]
    equation_results[3] = 5.50579117241379
    equation_results[4:7] = 0
    equation_results[7:11] = x[7:11]
    equation_results[11] = 7.794976
    equation_results[12:15] = 0
    equation_results[15] = x[15]
    
    return equation_results

df_model['FC CS Gasoline Equivalent (L/100km)'] = FC_CS_Gasoline_Equivalent(df_model['FC Achieved Gasoline Equivalent (L/100km)'])


# In[32]:


df_model.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x)


# "Rules" Supplement

# In[33]:


def CD_Mode_Elec_Target_1_for_BEV_PHEV_Fuel_Cell_System_Battery_Rated_Power_FCEV(x1, x2):
    equation_results = np.zeros_like(x1)

    equation_results[:3] = 0
    equation_results[3:7] = np.where(
        x1[3:7] <= 1000,
        0.0112 * x1[3:7] + 0.4,
        np.where(
            x1[3:7] <= 1600,
            0.0078 * x1[3:7] + 3.8,
            0.0048 * x1[3:7] + 8.6
        )
    )
    equation_results[7] = x2[7]
    equation_results[8:11] = 0
    equation_results[11:15] = np.where(
        x1[11:15] <= 1000,
        0.0112 * x1[11:15] + 0.4,
        np.where(
            x1[11:15] <= 1600,
            0.0078 * x1[11:15] + 3.8,
            0.0048 * x1[11:15] + 8.6
        )
    )
    equation_results[15] = x2[15]

    return equation_results

df_rules['CD Mode Elec Target 1 (kWh/100km) for BEV/PHEV-Fuel Cell System Battery Rated Power (kW) (FCEV)'] = CD_Mode_Elec_Target_1_for_BEV_PHEV_Fuel_Cell_System_Battery_Rated_Power_FCEV(df_model['Sales-weighted Weight (kg)'],
                                                                                                                                                                                               df_model['Battey (KWh)/Fuel Cell Sytem (kW)'])


# In[34]:


def BEV_PHEV_FCEV_Standard_Credit(x1, x2):
    equation_results = np.zeros_like(x1)

    equation_results[:3] = 0
    equation_results[3] = np.where(Year <= 2020, 2, 1.6)
    
    equation_results[4:7] = np.where(
        Year <= 2020,
        np.minimum(0.012 * x1[4:7] + 0.8, 5),
        np.where(x1[4:7] <= 150, 1, 0.0056 * x1[4:7] + 0.4)
    )
    
    equation_results[7] = np.where(
        Year <= 2020,
        np.minimum(x2[7] * 0.16, 5),
        np.minimum(x2[7] * 0.08, 6)
    )
    
    equation_results[8:11] = 0
    equation_results[11] = np.where(Year <= 2020, 2, 1.6)
    
    equation_results[12:15] = np.where(
        Year <= 2020,
        np.minimum(0.012 * x1[12:15] + 0.8, 5),
        np.where(x1[12:15] <= 150, 1, 0.0056 * x1[12:15] + 0.4)
    )
    
    equation_results[15] = np.where(
        Year <= 2020,
        np.minimum(x2[15] * 0.16, 5),
        np.minimum(x2[15] * 0.08, 6)
    )

    return equation_results

df_rules['BEV/PHEV/FCEV Standard Credit'] = BEV_PHEV_FCEV_Standard_Credit(df_model['Elec Range (km)'],
                                                                          df_rules['CD Mode Elec Target 1 (kWh/100km) for BEV/PHEV-Fuel Cell System Battery Rated Power (kW) (FCEV)'])


# In[35]:


def FC_Limit(x1, x2, x3, x4, x5, x6, x7):
    
    equation_results = np.zeros_like(x1)

    x3_interp = np.interp(x1, x2, x3)
    idx = np.searchsorted(x2, x1, side='right') - 1
    idx = np.clip(idx, 0, len(x2) - 1)

    equation_results[:16] = np.where(
        Year <= 2020,
        x3_interp[:16],
        np.where(
            x1[:16] < x4.iloc[0],
            x5.iloc[0],
            np.where(
                x1[:16] < x4.iloc[2],
                x6.iloc[1] * (x1[:16] + x7.iloc[1]) + x5.iloc[1],
                x5.iloc[2]
            )
        )
    )

    return equation_results


df_rules['FC Limit (L/100km)'] = FC_Limit(df_model['Sales-weighted Weight (kg)'], 
                                          df_rules_Phase_4_CAFC_WLTC_adjusted['CM'], 
                                          df_rules_Phase_4_CAFC_WLTC_adjusted['FC Limit'],
                                          df_rules_Phase_5_CAFC_Limit['CM'],
                                          df_rules_Phase_5_CAFC_Limit['c'],
                                          df_rules_Phase_5_CAFC_Limit['a'],
                                          df_rules_Phase_5_CAFC_Limit['b'])


# In[36]:


def CD_Mode_Elec_Target_1(x):
    equation_results = np.zeros_like(x)

    equation_results[:3] = 0
    equation_results[3:7] = np.where(
        x[3:7] < 1000,
        0.014 * x[3:7] + 0.5,
        np.where(
            x[3:7] <= 1600,
            0.012 * x[3:7] + 2.5,
            0.005 * x[3:7] + 13.7
        )
    )
    equation_results[7:11] = 0
    equation_results[11:15] = np.where(
        x[11:15] < 1000,
        0.014 * x[11:15] + 0.5,
        np.where(
            x[11:15] <= 1600,
            0.012 * x[11:15] + 2.5,
            0.005 * x[11:15] + 13.7
        )
    )
    equation_results[15] = 0

    return equation_results

df_rules['CD Mode Elec Target 1(kWh/100km)'] = CD_Mode_Elec_Target_1(df_model['Sales-weighted Weight (kg)'])


# In[37]:


def CD_Mode_Elec_Target_2(x):
    
    equation_results = np.zeros_like(x)
    
    equation_results[:3] = 0
    equation_results[3:7] = np.where(
        x[3:7] < 1000, 0.0098 * x[3:7] + 0.35,
        np.where(
            x[3:7] <= 1600, 0.0084 * x[3:7] + 1.75,
            0.0035 * x[3:7] + 9.59
        )
    )
    equation_results[7:11] = 0
    equation_results[11:15] = np.where(
        x[11:15] < 1000, 0.0098 * x[11:15] + 0.35,
        np.where(
            x[11:15] <= 1600, 0.0084 * x[11:15] + 1.75,
            0.0035 * x[11:15] + 9.59
        )
    )
    equation_results[15] = 0

    return equation_results

df_rules['CD Mode Elec Target 2(kWh/100km)'] = CD_Mode_Elec_Target_2(df_model['Sales-weighted Weight (kg)'])


# In[38]:


def NEV_Credit_Adjustment(x):
    equation_results = np.zeros_like(x)

    equation_results[:4] = 0

    equation_results[4:7] = np.where(
        x[4:7] < 150, 0.7,
        np.where(
            x[4:7] < 200, 0.8,
            np.where(
                x[4:7] < 300, 0.9,
                1
            )
        )
    )

    equation_results[7:12] = 0

    equation_results[12:15] = np.where(
        x[12:15] < 150, 0.7,
        np.where(
            x[12:15] < 200, 0.8,
            np.where(
                x[12:15] < 300, 0.9,
                1
            )
        )
    )

    equation_results[15] = 0

    return equation_results

df_rules['NEV Credit Adjustment'] = NEV_Credit_Adjustment(df_model['Elec Range (km)'])


# In[39]:


def Elec_Consp_Adjustment_Factor(x1, x2, x3, x4, x5, x6, x7):
    equation_results = np.zeros_like(x1)

    equation_results[:3] = 0
    equation_results[3] = np.where(
        Year <= 2020,
        np.where(
            x1[3] < 50, 0,
            np.where(
                x2[3] < (x3[3] * 0.7), 1, 0.5
            ) * np.where(
                x4[3] > x5[3], 0.5, 1
            )
        ),
        np.maximum(
            np.where(
                x2[3] < (x3[3] * 0.7), 1, 0.5
            ) * np.where(
                x4[3] > x6[3] * 1.35, 0.5, 1
            ),
            0.5
        )
    )
    equation_results[4:7] = np.where(
        Year <= 2020,
        np.where(
            x4[4:7] > x5[4:7], 0.5,
            np.where(
                x4[4:7] > x7[4:7], 1, 1.2
            )
        ),
        np.where(
            x4[4:7] <= x6[4:7],
            np.minimum(1.5, x6[4:7] / x4[4:7]),
            0.5
        )
    )
    equation_results[7:11] = 0
    equation_results[11] = np.where(
        Year <= 2020,
        np.where(
            x1[3] < 50, 0,
            np.where(
                x2[3] < (x3[3] * 0.7), 1, 0.5
            ) * np.where(
                x4[3] > x5[3], 0.5, 1
            )
        ),
        np.maximum(
            np.where(
                x2[3] < (x3[3] * 0.7), 1, 0.5
            ) * np.where(
                x4[3] > x6[3] * 1.35, 0.5, 1
            ),
            0.5
        )
    )
    equation_results[12:15] = np.where(
        Year <= 2020,
        np.where(
            x4[12:15] > x5[12:15], 0.5,
            np.where(
                x4[12:15] > x7[12:15], 1, 1.2
            )
        ),
        np.where(
            x4[12:15] <= x6[12:15],
            np.minimum(1.5, x6[12:15] / x4[12:15]),
            0.5
        )
    )
    equation_results[15] = 0

    return equation_results

df_rules['Elec Consp Adjustment Factor'] = Elec_Consp_Adjustment_Factor(df_model['Elec Range (km)'],
                                                                        df_model['FC CS Gasoline Equivalent (L/100km)'],
                                                                        df_rules['FC Limit (L/100km)'],
                                                                        df_model['EC CD (kWh/100km)'],
                                                                        df_rules['CD Mode Elec Target 1(kWh/100km)'],
                                                                        df_rules['CD Mode Elec Target 1 (kWh/100km) for BEV/PHEV-Fuel Cell System Battery Rated Power (kW) (FCEV)'],
                                                                        df_rules['CD Mode Elec Target 2(kWh/100km)'])


# In[40]:


def NEV_Credit(x1, x2, x3, x4):
    equation_results = np.zeros_like(x1)

    equation_results[:3] = x1[:3] * x2[:3]
    equation_results[3] = np.where(
        Year <= 2020,
        np.minimum(x1[3] * x2[3], 5),
        np.minimum(x1[3] * x2[3], 3.4)
    )
    equation_results[4:7] = np.where(
        Year <= 2020,
        np.minimum(x1[4:7] * x2[4:7], 5),
        np.minimum(x1[4:7] * x2[4:7] * x3[4:7] * x4[4:7], 3.4)
    )
    equation_results[7] = x1[7]
    equation_results[8:11] = x1[8:11] * x2[8:11]
    equation_results[11] = np.where(
        Year <= 2020,
        np.minimum(x1[11] * x2[11], 5),
        np.minimum(x1[11] * x2[11], 3.4)
    )
    equation_results[12:15] = np.where(
        Year <= 2020,
        np.minimum(x1[12:15] * x2[12:15], 5),
        np.minimum(x1[12:15] * x2[12:15] * x3[12:15] * x4[12:15], 3.4)
    )
    equation_results[15] = x1[15]

    return equation_results

df_rules['NEV Credit'] = NEV_Credit(df_rules['BEV/PHEV/FCEV Standard Credit'],
                                    df_rules['Elec Consp Adjustment Factor'],
                                    df_rules['Energy Density Adjustment'],
                                    df_rules['NEV Credit Adjustment'])


# In[41]:


df_rules.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x)


# FCEV-TCO

# In[42]:


#FCEV-TCO
df_FCEV_Technical_Information = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\FCEV-TCO.xlsx", usecols="B:F", skiprows=1, nrows=3)
df_FCEV_Unit_Cost = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\FCEV-TCO.xlsx", usecols="B:F", skiprows=7, nrows=3)
df_FCEV_Production_Cost = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\FCEV-TCO.xlsx", usecols="B:H", skiprows=13, nrows=3)


# In[43]:


def Fuel_Cell_System_Output(x):
    equation_results = np.zeros(2)

    equation_results[0] = x[7]
    equation_results[1] = x[15]
    
    return equation_results

df_FCEV_Technical_Information['Fuel Cell System output (kW)'] = Fuel_Cell_System_Output(df_model['Battey (KWh)/Fuel Cell Sytem (kW)'])


# In[44]:


def Hydrogen_Tank_Capacity(x1, x2):
    equation_results = np.zeros(2)

    equation_results[0] = x1[7] * x2[7] / 100 / 3.78541 / 1.019
    equation_results[1] = x1[15] * x2[15] / 100 / 3.78541 / 1.019
    
    return equation_results

df_FCEV_Technical_Information['Hydrogen Tank Capacity (kg)'] = Hydrogen_Tank_Capacity(df_model['FC Achieved Gasoline Equivalent (L/100km)'],
                                                                                      df_model['Elec Range (km)'])


# In[45]:


def Fuel_Cell_System_Cost(x1, x2):
    equation_results = np.zeros(2)

    equation_results[0] = x1.iloc[0] * x2.iloc[0]
    equation_results[1] = x1.iloc[0] * x2.iloc[0]
    
    return equation_results

df_FCEV_Unit_Cost['Fuel Cell System Cost ($/kW)'] = Fuel_Cell_System_Cost(df_industry['Fuel cell system costs in 2020 ($/kW)'],
                                                                          df_industry['Fuel cell system cost coefficient by year'])


# In[46]:


def Battery_Cost(x1, x2, x3):
    idx = np.searchsorted(x2, Year) 
    idx = np.clip(idx, 0, len(x2) - 1)  

    equation_results = np.zeros(2)  

    equation_results[0] = x1[idx] * x3.iloc[0]
    equation_results[1] = x1[idx] * x3.iloc[0]

    return equation_results

df_FCEV_Unit_Cost['Battery Cost ($/KWh)'] = Battery_Cost(df_projection['Battery Cost (benchmark = 2017)'],
                                                         df_projection['Year'],
                                                         df_industry['Battery pack costs in 2020 ($/kWh)'])


# In[47]:


def Motor_Cost(x1, x2):
    idx = np.searchsorted(x2, Year)  
    idx = np.clip(idx, 0, len(x2) - 1)  

    equation_results = np.zeros(2)  

    equation_results[0] = 8 * x1[idx]
    equation_results[1] = 8 * x1[idx]

    return equation_results

df_FCEV_Unit_Cost['Motor Cost ($/kW)'] = Motor_Cost(df_projection['Motor Cost ($/KW)'],
                                                    df_projection['Year'])


# In[48]:


def Hydrogen_Storage_Cost(x1, x2):
    idx = np.searchsorted(x2, Year)  
    idx = np.clip(idx, 0, len(x2) - 1)  

    equation_results = np.zeros(2)  

    equation_results[0] = x1[idx]
    equation_results[1] = x1[idx]

    return equation_results

df_FCEV_Unit_Cost['Hydrogen Storage Cost ($/kg)'] = Hydrogen_Storage_Cost(df_projection['Hydrogen Storage Cost ($/Kg)'],
                                                                          df_projection['Year'])


# In[49]:


def Fuel_Cell_System_Total_Cost(x1, x2):
    equation_results = np.zeros(2)

    equation_results[0] = x1.iloc[0] * x2.iloc[0]
    equation_results[1] = x1.iloc[0] * x2.iloc[0]
    
    return equation_results

df_FCEV_Production_Cost['Fuel Cell System Total Cost ($)'] = Fuel_Cell_System_Total_Cost(df_FCEV_Technical_Information['Fuel Cell System output (kW)'],
                                                                                         df_FCEV_Unit_Cost['Fuel Cell System Cost ($/kW)'])


# In[50]:


def Battery_Total_Cost(x1, x2):
    equation_results = np.zeros(2)

    equation_results[0] = x1.iloc[0] * x2.iloc[0]
    equation_results[1] = x1.iloc[1] * x2.iloc[1]
    
    return equation_results

df_FCEV_Production_Cost['Battery Total Cost ($)'] = Battery_Total_Cost(df_FCEV_Technical_Information['Battery Capacity(kWh)'],
                                                                       df_FCEV_Unit_Cost['Battery Cost ($/KWh)'])


# In[51]:


def Electric_Drive_Total_Cost(x1, x2):
    equation_results = np.zeros(2)

    equation_results[0] = x1.iloc[0] * x2.iloc[0]
    equation_results[1] = x1.iloc[1] * x2.iloc[1]
    
    return equation_results

df_FCEV_Production_Cost['Electric Drive Total Cost ($)'] = Electric_Drive_Total_Cost(df_FCEV_Technical_Information['Motor Power (KW)'],
                                                                                     df_FCEV_Unit_Cost['Motor Cost ($/kW)'])


# In[52]:


def Hydrogen_Storage_Total_Cost(x1, x2):
    equation_results = np.zeros(2)

    equation_results[0] = x1.iloc[0] * x2.iloc[0]
    equation_results[1] = x1.iloc[1] * x2.iloc[1]
    
    return equation_results

df_FCEV_Production_Cost['Hydrogen Storage Total Cost ($)'] = Hydrogen_Storage_Total_Cost(df_FCEV_Technical_Information['Hydrogen Tank Capacity (kg)'],
                                                                                         df_FCEV_Unit_Cost['Hydrogen Storage Cost ($/kg)'])


# In[53]:


def Other_Cost(x1, x2, x3):
    equation_results = np.zeros(2)

    equation_results[0] = x1.iloc[1] * x2.iloc[0]
    equation_results[1] = x1.iloc[4] * x3.iloc[0]
    
    return equation_results

df_FCEV_Production_Cost['Other Cost ($)'] = Other_Cost(df_model_vehicle_product_cost['Cost Intercept ($)'],
                                                       df_industry_vehicle_production['Sed BEV Base Cost Index'],
                                                       df_industry_vehicle_production['SUV BEV Base Cost Index'])


# In[54]:


def Total_Production_Cost(x1, x2, x3, x4, x5):
    equation_results = np.zeros(2)

    equation_results[0] = x1.iloc[0] + x2.iloc[0] + x3.iloc[0] + x4.iloc[0] + x5.iloc[0]
    equation_results[1] = x1.iloc[1] + x2.iloc[1] + x3.iloc[1] + x4.iloc[1] + x5.iloc[1]
    
    return equation_results

df_FCEV_Production_Cost['Total Production Cost ($)'] = Total_Production_Cost(df_FCEV_Production_Cost['Fuel Cell System Total Cost ($)'],
                                                                             df_FCEV_Production_Cost['Battery Total Cost ($)'],
                                                                             df_FCEV_Production_Cost['Electric Drive Total Cost ($)'],
                                                                             df_FCEV_Production_Cost['Hydrogen Storage Total Cost ($)'],
                                                                             df_FCEV_Production_Cost['Other Cost ($)'])


# “Model” Supplement 2

# In[55]:


def NEV_Credit(x1, x2):
    equation_results = np.zeros(16)

    equation_results[:3] = x1[:3]
    equation_results[3:8] = np.maximum(x1[3:8] * x2.iloc[0], 1)
    equation_results[8:11] = x1[8:11]
    equation_results[11:16] = np.maximum(x1[11:16] * x2.iloc[0], 1)
    
    return equation_results

df_model['NEV credit'] = NEV_Credit(df_rules['NEV Credit'], df_industry['PEV credit multiplier'])


# In[56]:


def Cost_Slope(x1,x2,x3):
    equation_results = np.zeros(6)

    equation_results[0] = 44 * x1.iloc[0]
    equation_results[1] = 89.2520357320322 * x2.iloc[0]
    equation_results[2] = 269.758679283696 * x3.iloc[0]
    equation_results[3] = 56.4889282574608 * x1.iloc[0]
    equation_results[4] = 111 * x2.iloc[0]
    equation_results[5] = 321.50484929574 * x3.iloc[0]
    
    return equation_results

df_model_vehicle_product_cost['Cost Slope ($/MPG)'] = Cost_Slope(df_industry_vehicle_production['ICEV Cost Slope Parameter_a'],
                                                                 df_industry_vehicle_production['ICEV Cost Slope Parameter_b'],
                                                                 df_industry_vehicle_production['ICEV Cost Slope Parameter_c'])


# In[57]:


def Cost_Intercept(x1,x2,x3,x4):
    equation_results = np.zeros(6)

    equation_results[0] = x1.iloc[0]
    equation_results[1] = ((x1.iloc[0] + x2.iloc[0] * x3.iloc[0]) - x2.iloc[0] * x3.iloc[1])
    equation_results[2] = (x1.iloc[0] + x2.iloc[0] * x3.iloc[0])- x2.iloc[0] * x3.iloc[1] + x3.iloc[1] * x2.iloc[1] - x3.iloc[2] * x2.iloc[1]
    equation_results[3] = x4.iloc[0]
    equation_results[4] = ((x4.iloc[0] + x2.iloc[3] * x3.iloc[3]) - x2.iloc[3] * x3.iloc[4])
    equation_results[5] = (x4.iloc[0] + x2.iloc[3] * x3.iloc[3]) - x2.iloc[3] * x3.iloc[4] + x3.iloc[4] * x2.iloc[4] - x3.iloc[5] * x2.iloc[4]
    
    return equation_results

df_model_vehicle_product_cost['Cost Intercept ($)'] = Cost_Intercept(df_industry_vehicle_production['Sed ProdCost A'],
                                                                     df_model_vehicle_product_cost['Breakpoint (MPG)'],
                                                                     df_model_vehicle_product_cost['Cost Slope ($/MPG)'],
                                                                     df_industry_vehicle_production['SUV ProdCost A'])


# In[58]:


def Production_Cost_Dollar(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
    equation_results = np.zeros(16)

    equation_results[:3] = np.where(
        x1[:3] > x2.iloc[1],
        x3.iloc[2] + x4.iloc[2] * x1[:3],
        np.where(
            x1[:3] > x2.iloc[0],
            x3.iloc[1] + x4.iloc[1] * x1[:3],
            x3.iloc[0] + x4.iloc[0] * x1[:3]
        )
    )
    equation_results[3] = (
        (x5.iloc[0] * x6.iloc[0] * x7[3]) * 1.635276 +
        x4.iloc[1] * (231.215 / x8[3]) * 0.611518 +
        x3.iloc[1] * x9.iloc[0]
    )
    equation_results[4:7] = (
        (x5.iloc[0] * x6.iloc[0] * x7[4:7]) * 1.635276 +
        x3.iloc[1] * x10.iloc[0]
    )
    equation_results[7] = x11.iloc[0]
    equation_results[8:11] = np.where(
        x1[8:11] > x2.iloc[4],
        x3.iloc[5] + x4.iloc[5] * x1[8:11],
        np.where(
            x1[8:11] > x2.iloc[3],
            x3.iloc[4] + x4.iloc[4] * x1[8:11],
            x3.iloc[3] + x4.iloc[3] * x1[8:11]
        )
    )
    equation_results[11] = (
        (x5.iloc[0] * x6.iloc[0] * x7[11]) * 1.635276 +
        x4.iloc[4] * (231.215 / x8[11]) * 0.611518 +
        x3.iloc[4] * x12.iloc[0]
    )
    equation_results[12:15] = (
        (x5.iloc[0] * x6.iloc[0] * x7[12:15]) * 1.635276 +
        x3.iloc[4] * x13.iloc[0]
    )
    equation_results[15] = x11.iloc[1]

    return equation_results

df_model['Production Cost ($)'] = Production_Cost_Dollar(df_model['Miles/Gallon'],
                                                         df_model_vehicle_product_cost['Breakpoint (MPG)'],
                                                         df_model_vehicle_product_cost['Cost Intercept ($)'],
                                                         df_model_vehicle_product_cost['Cost Slope ($/MPG)'],
                                                         df_industry['Battery pack costs in 2020 ($/kWh)'],
                                                         df_industry['Battery cost coefficient by year'],
                                                         df_model['Battey (KWh)/Fuel Cell Sytem (kW)'],
                                                         df_model['FC CS Gasoline Equivalent (L/100km)'],
                                                         df_industry_vehicle_production['Sed PHEV Base Cost Index'],
                                                         df_industry_vehicle_production['Sed BEV Base Cost Index'],
                                                         df_FCEV_Production_Cost['Total Production Cost ($)'],
                                                         df_industry_vehicle_production['SUV PHEV Base Cost Index'],
                                                         df_industry_vehicle_production['SUV BEV Base Cost Index'])


# In[59]:


def Production_Cost_CNY(x1, x2):
    equation_results = np.zeros(16)

    equation_results[:16] = x1[:16] * x2.iloc[0]
    
    return equation_results

df_model['Production Cost (￥)'] = Production_Cost_CNY(df_model['Production Cost ($)'],df_consumer['￥/$ in 2020'])


# In[60]:


def Additional_Subsidy_to_Consumers(x1, x2, x3):
    equation_results = np.zeros(16)

    equation_results[:2] = x1.iloc[0] + x2.iloc[0] + x3.iloc[0]
    equation_results[2] = x1.iloc[0] + x2.iloc[0] + x3.iloc[1]
    equation_results[3:8] = x1.iloc[0] + x2.iloc[1]
    equation_results[8:11] = x1.iloc[1] + x2.iloc[0] + x3.iloc[0]
    equation_results[11:16] = x1.iloc[1] + x2.iloc[1]
    
    return equation_results

df_model['Additional Subsidy to Consumers (￥)'] = Additional_Subsidy_to_Consumers(df_industry_subsidy_change['Subsidy Change_1'],
                                                                                   df_industry_subsidy_change['Subsidy Change_2'],
                                                                                   df_industry_subsidy_change['Subsidy Change_3'])


# In[61]:


def Central_Gov_Sub_to_Consumers(x1, x2):
    
    equation_results = np.zeros(16)

    equation_results[:3] = 0
    equation_results[3] = np.select(
        [Year == 2017, Year == 2018, Year == 2019, Year == 2020, Year == 2021, Year == 2022],
        [2.4, 2.2, 1, 0.85, 0.5, 0.25],
        default=0
    ) * 10000
    equation_results[4] = np.select(
        [Year == 2017, Year == 2018],
        [3.6, 1.5],
        default=0
    ) * 10000
    equation_results[5:7] = np.select(
        [Year == 2017, Year == 2018, Year == 2019, Year == 2020, Year == 2021, Year == 2022],
        [4.4, 4.5, 1.8, 1.62, 1, 0.5],
        default=0
    ) * 10000
    equation_results[7] = np.select(
        [Year == 2016, Year == 2017, Year == 2018, Year == 2019, Year == 2020, Year == 2021, Year == 2022, Year == 2023],
        [200000, 200000, x1[7] * 6000, x1[7] * 4800, np.minimum((x2[7] - 50) * 0.03 + 1, 1.9) * 1.3 * 100000, 
         np.minimum((x2[7] - 50) * 0.03 + 1, 1.9) * 1.2 * 100000, np.minimum((x2[7] - 50) * 0.03 + 1, 1.9) * 1.1 * 100000, 
         np.minimum((x2[7] - 50) * 0.03 + 1, 1.9) * 0.9 * 100000],
        default=0
    )
    equation_results[8:12] = 0
    equation_results[12] = np.select(
        [Year == 2017, Year == 2018, Year == 2019, Year == 2020, Year == 2021, Year == 2022],
        [2.4, 2.2, 1, 0.85, 0.5, 0.25],
        default=0
    ) * 10000
    equation_results[12:14] = np.select(
        [Year == 2017, Year == 2018, Year == 2019, Year == 2020, Year == 2021],
        [4.4, 3.4, 1.8, 1, 0.5],
        default=0
    ) * 10000
    equation_results[14] = np.select(
        [Year == 2017, Year == 2018, Year == 2019, Year == 2020, Year == 2021, Year == 2022],
        [4.4, 4.5, 1.8, 1.62, 1, 0.5],
        default=0
    ) * 10000
    equation_results[15] = np.select(
        [Year == 2016, Year == 2017, Year == 2018, Year == 2019, Year == 2020, Year == 2021, Year == 2022, Year == 2023],
        [200000, 200000, x1[15] * 6000, x1[15] * 4800, np.minimum((x2[15] - 50) * 0.03 + 1, 1.9) * 1.3 * 100000, 
         np.minimum((x2[15] - 50) * 0.03 + 1, 1.9) * 1.2 * 100000, np.minimum((x2[15] - 50) * 0.03 + 1, 1.9) * 1.1 * 100000, 
         np.minimum((x2[15] - 50) * 0.03 + 1, 1.9) * 0.9 * 100000],
        default=0
    )

    return equation_results

df_model['Central Gov Sub to Consumers (￥)'] = Central_Gov_Sub_to_Consumers(df_model['Battey (KWh)/Fuel Cell Sytem (kW)'],
                                                                             df_rules['CD Mode Elec Target 1 (kWh/100km) for BEV/PHEV-Fuel Cell System Battery Rated Power (kW) (FCEV)'])


# In[62]:


def Avg_Local_Gov_Sub_to_Consumers(x):
    equation_results = np.zeros(16)

    equation_results[:3] = 0
    equation_results[3:8] = np.where(Year <= 2019, x[3:8] / 2, 0)
    equation_results[8:11] = 0
    equation_results[11:16] = np.where(Year <= 2019, x[11:16] / 2, 0)
    
    return equation_results

df_model['Avg Local Gov Sub to Consumers (￥)'] = Avg_Local_Gov_Sub_to_Consumers(df_model['Central Gov Sub to Consumers (￥)'])


# In[63]:


df_model.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x)


# Infrastructure

# In[64]:


df_infrastructure_Personal_Vehicle_Market_BEV = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Infrastructure.xlsx", sheet_name="Personal Vehicle Market_BEV")
df_infrastructure_Fleet_Vehicle_Market_BEV = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Infrastructure.xlsx", sheet_name="Fleet Vehicle Market_BEV")

df_infrastructure_Personal_Vehicle_Market_PHEV = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Infrastructure.xlsx", sheet_name="Personal Vehicle Market_PHEV")
df_infrastructure_Fleet_Vehicle_Market_PHEV = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Infrastructure.xlsx", sheet_name="Fleet Vehicle Market_PHEV")

df_infrastructure_Personal_Vehicle_Market_FCEV = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Infrastructure.xlsx", sheet_name="Personal Vehicle Market_FCEV")
df_infrastructure_Fleet_Vehicle_Market_FCEV = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Infrastructure.xlsx", sheet_name="Fleet Vehicle Market_FCEV")

df_infrastructure_Consumer_Charging_Behavior = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Infrastructure.xlsx", sheet_name="Consumer Charging Behavior")
df_infrastructure_Charging_Infrastructure = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Infrastructure.xlsx", sheet_name="Charging Infrastructure", usecols="A:P", skiprows=1, nrows=2)
df_infrastructure_Hydrogen_Station_Infrastructure = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Infrastructure.xlsx", sheet_name="Hydrogen Station Infrastructure")
df_infrastructure_Gasoline_Station_Infrastructure = pd.read_excel(r"D:\Users\11379\Desktop\博士\博0\NEOCC model\Infrastructure.xlsx", sheet_name="Gasoline Station Infrastructure")


# In[65]:


def Probability_of_Owning_A_Home_Charging(x1, x2):
    
    idx = np.searchsorted(x2, Year)
    idx = np.clip(idx, 0, len(x2) - 1)
    
    equation_results = x1.iloc[idx] * (0.7 + 0.0096 * (Year - 2019))

    return equation_results

df_infrastructure_Charging_Infrastructure['Probability of owning a home-charging'] = Probability_of_Owning_A_Home_Charging(df_projection['Home Charg Availability'], 
                                                                                                                           df_projection['Year'])


# In[66]:


def Public_Charging_Availability(x1, x2):
    
    idx = np.searchsorted(x2, Year)
    idx = np.clip(idx, 0, len(x2) - 1)
    
    equation_results = x1.iloc[idx]

    return equation_results

df_infrastructure_Charging_Infrastructure['Public charging availability'] = Public_Charging_Availability(df_projection['Public Charg availability'], 
                                                                                                         df_projection['Year'])


# In[67]:


def Round_Trip_Time_Home_to_Public_Charge_Mins(x):
    
    equation_results = 3.391 * x.iloc[0] ** (-0.898244)

    return equation_results

df_infrastructure_Charging_Infrastructure['Round trip time (home to public charge, mins)'] = Round_Trip_Time_Home_to_Public_Charge_Mins(df_infrastructure_Charging_Infrastructure['Public charging availability'])


# In[68]:


def Fast_Charging_Possibility(x1, x2):
    
    idx = np.searchsorted(x2, Year)
    idx = np.clip(idx, 0, len(x2) - 1)
    
    equation_results = x1.iloc[idx]

    return equation_results

df_infrastructure_Charging_Infrastructure['Fast charging possibility'] = Fast_Charging_Possibility(df_projection['Fast-charg Availability in Public'], 
                                                                                                   df_projection['Year'])


# In[69]:


def Charging_Power(x1, x2):
    
    idx = np.searchsorted(x2, Year)
    idx = np.clip(idx, 0, len(x2) - 1)
    
    equation_results = x1.iloc[idx]

    return equation_results

df_infrastructure_Charging_Infrastructure['Charging power (kW)'] = Charging_Power(df_projection['Fast-charging power (kW)'], 
                                                                                  df_projection['Year'])


# In[70]:


def Slow_Charging_Possiblity(x):
    
    equation_results = 1 - x.loc[0]

    return equation_results

df_infrastructure_Charging_Infrastructure['Slow charging possiblity'] = Slow_Charging_Possiblity(df_infrastructure_Charging_Infrastructure['Fast charging possibility'])


# In[71]:


def Slow_Charging_Power():
    
    if Year < 2025:
        equation_results = 7 + 3 * (Year - 2019) / (2025 - 2019)
    else:
        equation_results = 10

    return equation_results

df_infrastructure_Charging_Infrastructure['Charging power (kW).1'] = Slow_Charging_Power()


# In[72]:


def Public_H2_Refueling_Availability(x1, x2):
    
    idx = np.searchsorted(x2, Year)
    idx = np.clip(idx, 0, len(x2) - 1)
    
    equation_results = x1.iloc[idx]

    return equation_results

df_infrastructure_Hydrogen_Station_Infrastructure['Public H2 refueling availability'] = Public_H2_Refueling_Availability(df_projection['Public H2 station availability'], 
                                                                                                                         df_projection['Year'])


# In[73]:


def Round_Trip_Time_Home_to_Public_Stations_Mins(x):
    
    equation_results = 3.391 * x.iloc[0] ** (-0.898244)

    return equation_results

df_infrastructure_Hydrogen_Station_Infrastructure['Round trip time (home to public stations, mins)'] = Round_Trip_Time_Home_to_Public_Stations_Mins(df_infrastructure_Hydrogen_Station_Infrastructure['Public H2 refueling availability'])


# In[74]:


def Hydrogen_Price(x1, x2):
    
    idx = np.searchsorted(x2, Year)
    idx = np.clip(idx, 0, len(x2) - 1)
    
    equation_results = x1.iloc[idx] / 3.71483

    return equation_results

df_infrastructure_Hydrogen_Station_Infrastructure['Hydrogen price  (￥/Gasoline Equivalent Liter)'] = Hydrogen_Price(df_projection['Hydrogen price (CNY/kg)'], 
                                                                                                                     df_projection['Year'])


# In[75]:


def Refueling_Annoyance_Multiplier(x):
    
    equation_results = x

    return equation_results

df_infrastructure_Hydrogen_Station_Infrastructure['Refueling annoyance multiplier'] = Refueling_Annoyance_Multiplier(df_infrastructure_Charging_Infrastructure['Travel annoyance multiplier'])


# In[76]:


def Gasoline_Price(x1, x2):
    
    idx = np.searchsorted(x2, Year)
    idx = np.clip(idx, 0, len(x2) - 1)
    
    equation_results = x1.iloc[idx]

    return equation_results

df_infrastructure_Gasoline_Station_Infrastructure['Gasoline price (￥/L)'] = Gasoline_Price(df_projection['Fuel price(CNY/Liter)'], 
                                                                                            df_projection['Year'])


# In[77]:


def Costs_for_A_full_Range_Home_Charging(x1, x2, x3):
    equation_results = np.zeros(6)
    
    equation_results[:3] = (x1.loc[0] * x2.loc[4:6] * x3.loc[4:6] / 100).values
    equation_results[3:6] = (x1.loc[0] * x2.loc[12:14] * x3.loc[12:14] / 100).values

    return equation_results

df_infrastructure_Personal_Vehicle_Market_BEV['Costs for a full range @ home-charging (￥)'] = Costs_for_A_full_Range_Home_Charging(df_infrastructure_Charging_Infrastructure['Home electricity price (￥/kWh)'], 
                                                                                                                                    df_model['Elec Range (km)'], 
                                                                                                                                    df_model['EC CD (kWh/100km)'])


# In[78]:


def Costs_for_A_Full_Range_Public_Charging(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12):
    equation_results = np.zeros(6)
    
    equation_results[:3] = x1.iloc[0] * (x2.iloc[0] * (x3.iloc[0] / 60 + x4.loc[4:6] * x5.loc[4:6] / 100 / x6.iloc[0]) * x7.iloc[0] + (x8.iloc[0] + x9.iloc[0]) * x4.loc[4:6] * x5.loc[4:6] / 100) + x10.iloc[0] * (x2.iloc[0] * x3.iloc[0] * 2 * x11.iloc[0] / 60 + (x8.iloc[0] + x12.iloc[0]) * x4.loc[4:6] * x5.loc[4:6] / 100)
    equation_results[3:6] = x1.iloc[0] * (x2.iloc[0] * (x3.iloc[0] / 60 + x4.loc[12:14] * x5.loc[12:14] / 100 / x6.iloc[0]) * x7.iloc[0] + (x8.iloc[0] + x9.iloc[0]) * x4.loc[12:14] * x5.loc[12:14] / 100) + x10.iloc[0] * (x2.iloc[0] * x3.iloc[0] * 2 * x11.iloc[0] / 60 + (x8.iloc[0] + x12.iloc[0]) * x4.loc[12:14] * x5.loc[12:14] / 100)

    return equation_results

df_infrastructure_Personal_Vehicle_Market_BEV['Costs for a full range @ public-charging (￥)'] = Costs_for_A_Full_Range_Public_Charging(df_infrastructure_Charging_Infrastructure['Fast charging possibility'],
                                                                                                                                        df_consumer['Time value (￥/hour)'],
                                                                                                                                        df_infrastructure_Charging_Infrastructure['Round trip time (home to public charge, mins)'],
                                                                                                                                        df_model['Elec Range (km)'],
                                                                                                                                        df_model['EC CD (kWh/100km)'],
                                                                                                                                        df_infrastructure_Charging_Infrastructure['Charging power (kW)'],
                                                                                                                                        df_infrastructure_Charging_Infrastructure['Travel annoyance multiplier'],
                                                                                                                                        df_infrastructure_Charging_Infrastructure['Public charging electric price  (￥/kWh)'],
                                                                                                                                        df_infrastructure_Charging_Infrastructure['Fast charging service (￥/kWh)'],
                                                                                                                                        df_infrastructure_Charging_Infrastructure['Slow charging possiblity'],
                                                                                                                                        df_infrastructure_Charging_Infrastructure['Travel annoyance multiplier'],
                                                                                                                                        df_infrastructure_Charging_Infrastructure['Slow charging service (￥/kWh)'])


# In[79]:


def Costs_for_A_Full_Range_Workplace_Charging(x1, x2, x3):
    equation_results = np.zeros(6)
    
    equation_results[:3] = x1.iloc[0] * x2.loc[4:6] * x3.loc[4:6] / 100
    equation_results[3:6] =  x1.iloc[0] * x2.loc[12:14] * x3.loc[12:14] / 100
    
    return equation_results

df_infrastructure_Personal_Vehicle_Market_BEV['Costs for a full range @ workplace-charging (￥)'] = Costs_for_A_Full_Range_Workplace_Charging(df_infrastructure_Charging_Infrastructure['Infrastructure break-even cost (￥/kWh)'],
                                                                                                                                              df_model['Elec Range (km)'],
                                                                                                                                              df_model['EC CD (kWh/100km)'])


# In[80]:


df_infrastructure_Personal_Vehicle_Market_BEV.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x)


# In[81]:


def Electricity_Costs_for_A_Full_Range(x1, x2, x3, x4, x5, x6, x7):
    equation_results = np.zeros(6)
    
    equation_results[:3] = x1.iloc[0] * (x2.iloc[0] + x3.iloc[0]) * x4[4:7] * x5[4:7] / 100 + x6.iloc[0] * (x2.iloc[0] + x7.iloc[0]) * x4[4:7] * x5[4:7] / 100
    equation_results[3:6] = x1.iloc[0] * (x2.iloc[0] + x3.iloc[0]) * x4[12:15] * x5[12:15] / 100 + x6.iloc[0] * (x2.iloc[0] + x7.iloc[0]) * x4[12:15] * x5[12:15] / 100
    
    return equation_results

df_infrastructure_Fleet_Vehicle_Market_BEV['Electricity costs for a full range (￥)'] = Electricity_Costs_for_A_Full_Range(df_infrastructure_Charging_Infrastructure['Fast charging possibility'],
                                                                                                                           df_infrastructure_Charging_Infrastructure['Public charging electric price  (￥/kWh)'],
                                                                                                                           df_infrastructure_Charging_Infrastructure['Fast charging service (￥/kWh)'],
                                                                                                                           df_model['Elec Range (km)'],
                                                                                                                           df_model['EC CD (kWh/100km)'],
                                                                                                                           df_infrastructure_Charging_Infrastructure['Slow charging possiblity'],
                                                                                                                           df_infrastructure_Charging_Infrastructure['Slow charging service (￥/kWh)'])


# In[82]:


def Charg_Annoyance_for_A_Full_Range(x1, x2, x3, x4, x5, x6, x7):
    equation_results = np.zeros(6)
    
    equation_results[:3] = x1.iloc[0] * (x2[4:7] * x3[4:7] / 100 / x4.iloc[0] * x5.iloc[1]) + x6.iloc[0] * (x2[4:7] * x3[4:7] / 100 / x7.iloc[0] * x5.iloc[1])
    equation_results[3:6] = x1.iloc[0] * (x2[12:15] * x3[12:15] / 100 / x4.iloc[0] * x5.iloc[1]) + x6.iloc[0] * (x2[12:15] * x3[12:15] / 100 / x7.iloc[0] * x5.iloc[1])
    
    return equation_results

df_infrastructure_Fleet_Vehicle_Market_BEV['Charg annoyance for a full range (￥)'] = Charg_Annoyance_for_A_Full_Range(df_infrastructure_Charging_Infrastructure['Fast charging possibility'],
                                                                                                                       df_model['Elec Range (km)'],
                                                                                                                       df_model['EC CD (kWh/100km)'],
                                                                                                                       df_infrastructure_Charging_Infrastructure['Charging power (kW)'],
                                                                                                                       df_consumer['Time value (￥/hour)'],
                                                                                                                       df_infrastructure_Charging_Infrastructure['Slow charging possiblity'],
                                                                                                                       df_infrastructure_Charging_Infrastructure['Charging power (kW).1'])


# In[83]:


def Number_of_Charging_for_BEV_Full_Range(x1, x2, x3):
    equation_results = np.zeros(6)
    
    equation_results[:3] = np.ceil(x1.iloc[1] / (x2.iloc[4:7] * x3.iloc[1]))
    equation_results[3:6] = np.ceil(x1.iloc[1] / (x2.iloc[12:15] * x3.iloc[1]))
    
    return equation_results

df_infrastructure_Fleet_Vehicle_Market_BEV['Number of charging for BEV full range'] = Number_of_Charging_for_BEV_Full_Range(df_consumer['Lifetime kilometers'],
                                                                                                                            df_model['Elec Range (km)'],
                                                                                                                            df_consumer['Range utility'])


# In[84]:


def Lifetime_Costs(x1, x2, x3):
    equation_results = np.zeros(6)
    
    equation_results[:6] = (x1[:6] + x2[:6]) * x3[:6]
    
    return equation_results

df_infrastructure_Fleet_Vehicle_Market_BEV['Lifetime costs'] = Lifetime_Costs(df_infrastructure_Fleet_Vehicle_Market_BEV['Electricity costs for a full range (￥)'],
                                                                              df_infrastructure_Fleet_Vehicle_Market_BEV['Charg annoyance for a full range (￥)'],
                                                                              df_infrastructure_Fleet_Vehicle_Market_BEV['Number of charging for BEV full range'])


# In[85]:


df_infrastructure_Fleet_Vehicle_Market_BEV.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x)


# In[86]:


def Daily_Fuel_VKT(x):
    equation_results = np.zeros(2)
    
    equation_results[:2] = x.iloc[0]
    
    return equation_results

df_infrastructure_Personal_Vehicle_Market_PHEV['Daily Fuel VKT'] = Daily_Fuel_VKT(df_consumer['Driving - mean (km)'])


# In[87]:


def Lifetime_Fuel_Use(x1, x2, x3, x4):
    equation_results = np.zeros(2)
    
    equation_results[0] = x1.iloc[0] / x2.iloc[0] * x3.iloc[0] / 100 * x4[3]
    equation_results[1] = x1.iloc[1] / x2.iloc[0] * x3.iloc[0] / 100 * x4[11]
    
    return equation_results

df_infrastructure_Personal_Vehicle_Market_PHEV['Lifetime fuel use (L)'] = Lifetime_Fuel_Use(df_infrastructure_Personal_Vehicle_Market_PHEV['Daily Fuel VKT'],
                                                                                            df_consumer['Driving - mean (km)'],
                                                                                            df_consumer['Lifetime kilometers'],
                                                                                            df_model['FC CS Gasoline Equivalent (L/100km)'])


# In[88]:


def Fuel_Costs(x1, x2):
    equation_results = np.zeros(2)
    
    equation_results[0] = x1.iloc[0] * x2.iloc[0]
    equation_results[1] = x1.iloc[1] * x2.iloc[0]
    
    return equation_results

df_infrastructure_Personal_Vehicle_Market_PHEV['Fuel costs (￥)'] = Fuel_Costs(df_infrastructure_Personal_Vehicle_Market_PHEV['Lifetime fuel use (L)'],
                                                                               df_infrastructure_Gasoline_Station_Infrastructure['Gasoline price (￥/L)'])


# In[89]:


def Refuelling_Cost(x1, x2, x3):
    equation_results = np.zeros(2)
    
    equation_results[0] = x1.iloc[0] / x2.iloc[0] * x3.iloc[0]
    equation_results[1] = x1.iloc[1] / x2.iloc[0] * x3.iloc[0]
    
    return equation_results

df_infrastructure_Personal_Vehicle_Market_PHEV['Refuelling cost (￥)'] = Refuelling_Cost(df_infrastructure_Personal_Vehicle_Market_PHEV['Lifetime fuel use (L)'],
                                                                                         df_infrastructure_Gasoline_Station_Infrastructure['Gasoline Refill Amount (L)'],
                                                                                         df_consumer['Time value (￥/hour)'])


# In[90]:


def Total_Energy_Costs(x1, x2):
    equation_results = np.zeros(2)
    
    equation_results[0] = x1.iloc[0] + x2.iloc[0] 
    equation_results[1] = x1.iloc[1] + x2.iloc[1] 
    
    return equation_results

df_infrastructure_Personal_Vehicle_Market_PHEV['Total energy costs (￥)'] = Total_Energy_Costs(df_infrastructure_Personal_Vehicle_Market_PHEV['Fuel costs (￥)'],
                                                                                               df_infrastructure_Personal_Vehicle_Market_PHEV['Refuelling cost (￥)'])


# In[91]:


df_infrastructure_Personal_Vehicle_Market_PHEV.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x)


# In[92]:


def Daily_Fuel_VKT(x):
    equation_results = np.zeros(2)
    
    equation_results[:2] = x.iloc[1]
    
    return equation_results

df_infrastructure_Fleet_Vehicle_Market_PHEV['Daily Fuel VKT'] = Daily_Fuel_VKT(df_consumer['Driving - mean (km)'])


# In[93]:


def Lifetime_Fuel_Use(x1, x2, x3, x4):
    equation_results = np.zeros(2)
    
    equation_results[0] = x1.iloc[0] / x2.iloc[1] * x3.iloc[1] / 100 * x4[3]
    equation_results[1] = x1.iloc[1] / x2.iloc[1] * x3.iloc[1] / 100 * x4[11]
    
    return equation_results

df_infrastructure_Fleet_Vehicle_Market_PHEV['Lifetime fuel use (L)'] = Lifetime_Fuel_Use(df_infrastructure_Fleet_Vehicle_Market_PHEV['Daily Fuel VKT'],
                                                                                         df_consumer['Driving - mean (km)'],
                                                                                         df_consumer['Lifetime kilometers'],
                                                                                         df_model['FC CS Gasoline Equivalent (L/100km)'])


# In[94]:


def Fuel_Costs(x1, x2):
    equation_results = np.zeros(2)
    
    equation_results[0] = x1.iloc[0] * x2.iloc[0]
    equation_results[1] = x1.iloc[1] * x2.iloc[0]
    
    return equation_results

df_infrastructure_Fleet_Vehicle_Market_PHEV['Fuel costs (￥)'] = Fuel_Costs(df_infrastructure_Fleet_Vehicle_Market_PHEV['Lifetime fuel use (L)'],
                                                                            df_infrastructure_Gasoline_Station_Infrastructure['Gasoline price (￥/L)'])


# In[95]:


def Refuelling_Cost(x1, x2, x3):
    equation_results = np.zeros(2)
    
    equation_results[0] = x1.iloc[0] / x2.iloc[0] * x3.iloc[1]
    equation_results[1] = x1.iloc[1] / x2.iloc[0] * x3.iloc[1]
    
    return equation_results

df_infrastructure_Fleet_Vehicle_Market_PHEV['Refuelling cost (￥)'] = Refuelling_Cost(df_infrastructure_Fleet_Vehicle_Market_PHEV['Lifetime fuel use (L)'],
                                                                                      df_infrastructure_Gasoline_Station_Infrastructure['Gasoline Refill Amount (L)'],
                                                                                      df_consumer['Time value (￥/hour)'])


# In[96]:


def Total_Energy_Costs(x1, x2):
    equation_results = np.zeros(2)
    
    equation_results[0] = x1.iloc[0] + x2.iloc[0] 
    equation_results[1] = x1.iloc[1] + x2.iloc[1] 
    
    return equation_results

df_infrastructure_Fleet_Vehicle_Market_PHEV['Total energy costs (￥)'] = Total_Energy_Costs(df_infrastructure_Fleet_Vehicle_Market_PHEV['Fuel costs (￥)'],
                                                                                            df_infrastructure_Fleet_Vehicle_Market_PHEV['Refuelling cost (￥)'])


# In[97]:


df_infrastructure_Fleet_Vehicle_Market_PHEV.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x)


# In[98]:


def Tank_Size(x1, x2):
    equation_results = np.zeros(2)
    
    equation_results[0] = x1[7] * x2[7] / 100 / 3.78541 / 1.019
    equation_results[1] = x1[15] * x2[15] / 100 / 3.78541 / 1.019
    
    return equation_results

df_infrastructure_Personal_Vehicle_Market_FCEV['Tank size (kg)'] = Tank_Size(df_model['Elec Range (km)'],
                                                                             df_model['FC CS Gasoline Equivalent (L/100km)'])


# In[99]:


def Refilling_Inconvenience_Costs_Public(x1, x2, x3, x4, x5, x6, x7):
    equation_results = np.zeros(2)
    
    equation_results[0] = (x1.iloc[0] * x2.iloc[0] + x3.iloc[0] * (1 - x4.iloc[0]) / x5.iloc[0] + x6.iloc[0]) * x7.iloc[0] / 60
    equation_results[1] = (x1.iloc[0] * x2.iloc[0] + x3.iloc[1] * (1 - x4.iloc[0]) / x5.iloc[0] + x6.iloc[0]) * x7.iloc[0] / 60
    
    return equation_results

df_infrastructure_Personal_Vehicle_Market_FCEV['Refilling Inconvenience Costs @ Public (￥/Trip)'] = Refilling_Inconvenience_Costs_Public(df_infrastructure_Hydrogen_Station_Infrastructure['Refueling annoyance multiplier'],
                                                                                                                                          df_infrastructure_Hydrogen_Station_Infrastructure['Round trip time (home to public stations, mins)'],
                                                                                                                                          df_infrastructure_Personal_Vehicle_Market_FCEV['Tank size (kg)'],
                                                                                                                                          df_infrastructure_Hydrogen_Station_Infrastructure['Tank state at beginning of refill (%)'],
                                                                                                                                          df_infrastructure_Hydrogen_Station_Infrastructure['Filling speed (kg/min)'],
                                                                                                                                          df_infrastructure_Hydrogen_Station_Infrastructure['Refueling time, i.e. time on operating the dispenser (mins)'],
                                                                                                                                          df_consumer['Time value (￥/hour)'])


# In[100]:


df_infrastructure_Personal_Vehicle_Market_FCEV.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x)


# Personal Vehicle Market

# In[101]:


def UF_Personal_Vehicle(x, adjust):
    scale = df_consumer['Range Extension'].iloc[0] * adjust / df_consumer['Gamma-Scale-thetaBeta'].iloc[0]
    shape = df_consumer['Gamma-Shape-kAlpha'].iloc[0] + 1
    return gamma.cdf(x * scale, a=shape)

adjustments = [1, 1, 1, df_consumer['Range utility'].iloc[0], df_consumer['Range utility'].iloc[0], df_consumer['Range utility'].iloc[0], 
               df_consumer['Range utility'].iloc[0], 1, 1, 1, 1, df_consumer['Range utility'].iloc[0], df_consumer['Range utility'].iloc[0], 
               df_consumer['Range utility'].iloc[0], df_consumer['Range utility'].iloc[0], 1]

df_model_personal_vehicle_market['UF'] = [UF_Personal_Vehicle(df_model['Elec Range (km)'].iloc[i], adjustments[i]) for i in range(len(df_model['Elec Range (km)']))]


# In[102]:


def PF_Personal_Vehicle(x, adjust):
    scale = df_consumer['Range Extension'].iloc[0] * adjust / df_consumer['Gamma-Scale-thetaBeta'].iloc[0]
    shape = df_consumer['Gamma-Shape-kAlpha'].iloc[0]
    return gamma.cdf(x * scale, a=shape)

adjustments = [1, 1, 1, df_consumer['Range utility'].iloc[0], df_consumer['Range utility'].iloc[0], df_consumer['Range utility'].iloc[0], 
               df_consumer['Range utility'].iloc[0], 1, 1, 1, 1, df_consumer['Range utility'].iloc[0], df_consumer['Range utility'].iloc[0], 
               df_consumer['Range utility'].iloc[0], df_consumer['Range utility'].iloc[0], 1]

df_model_personal_vehicle_market['PF'] = [PF_Personal_Vehicle(df_model['Elec Range (km)'].iloc[i], adjustments[i]) for i in range(len(df_model['Elec Range (km)']))]


# In[103]:


def Daily_Fuel_VKT_Personal_Vehicle(x1, x2, x3):
    equation_results = np.zeros_like(x1)

    equation_results[:3] = df_consumer['Driving - mean (km)'].iloc[0] * (1 - x1[:3]) - x3[:3] * df_consumer['Range Extension'].iloc[0] * (1 - x2[:3])
    equation_results[3:7] = df_consumer['Driving - mean (km)'].iloc[0] * (1 - x1[3:7]) - x3[3:7] * df_consumer['Range Extension'].iloc[0] * df_consumer['Range utility'].iloc[0] * (1 - x2[3:7])
    equation_results[7] = df_consumer['Driving - mean (km)'].iloc[0] * x1[7] + x3[7] * df_consumer['Range Extension'].iloc[0] * (1 - x2[7])
    equation_results[8:10] = df_consumer['Driving - mean (km)'].iloc[0] * (1 - x1[8:10]) - x3[8:10] * df_consumer['Range Extension'].iloc[0] * (1 - x2[8:10])
    equation_results[10:15] = df_consumer['Driving - mean (km)'].iloc[0] * (1 - x1[10:15]) - x3[10:15] * df_consumer['Range Extension'].iloc[0] * df_consumer['Range utility'].iloc[0] * (1 - x2[10:15])
    equation_results[15] = df_consumer['Driving - mean (km)'].iloc[0] * x1[15] + x3[15] * df_consumer['Range Extension'].iloc[0] * (1 - x2[15])
    return equation_results

df_model_personal_vehicle_market['Daily Fuel VKT'] = Daily_Fuel_VKT_Personal_Vehicle(df_model_personal_vehicle_market['UF'], df_model_personal_vehicle_market['PF'], df_model['Elec Range (km)'])


# In[104]:


def Daily_Elec_VKT_Personal_Vehicle(x1, x2, x3):
    equation_results = np.zeros_like(x1)

    equation_results[:2] = df_consumer['Driving - mean (km)'].iloc[0] * x1[:2] + x3[:2] * df_consumer['Range Extension'].iloc[0] * (1 - x2[:2])
    equation_results[2:7] = df_consumer['Driving - mean (km)'].iloc[0] * x1[2:7] + x3[2:7] * df_consumer['Range Extension'].iloc[0] * df_consumer['Range utility'].iloc[0] * (1 - x2[2:7])
    equation_results[7] = 0
    equation_results[8:10] = df_consumer['Driving - mean (km)'].iloc[0] * x1[8:10] + x3[8:10] * df_consumer['Range Extension'].iloc[0] * (1 - x2[8:10])
    equation_results[10:15] = df_consumer['Driving - mean (km)'].iloc[0] * x1[10:15] + x3[10:15] * df_consumer['Range Extension'].iloc[0] * df_consumer['Range utility'].iloc[0] * (1 - x2[10:15])
    equation_results[15] = 0
    return equation_results

df_model_personal_vehicle_market['Daily Elec VKT'] = Daily_Elec_VKT_Personal_Vehicle(df_model_personal_vehicle_market['UF'], df_model_personal_vehicle_market['PF'], df_model['Elec Range (km)'])


# In[105]:


def Lifetime_Fuel_Use_Personal_Vehicle(x1, x2, x3, x4):
    equation_results = np.zeros_like(x1)
    
    equation_results[:16] = x1[:16] / x2.iloc[0] * x3.iloc[0] / 100 * x4[:16]

    return equation_results

df_model_personal_vehicle_market['Lifetime Fuel Use (L)'] = Lifetime_Fuel_Use_Personal_Vehicle(df_model_personal_vehicle_market['Daily Fuel VKT'], 
                                                                                               df_consumer['Driving - mean (km)'], 
                                                                                               df_consumer['Lifetime kilometers'],
                                                                                               df_model['FC CS Gasoline Equivalent (L/100km)'])


# In[106]:


def Lifetime_Elec_Use_Personal_Vehicle(x1, x2, x3, x4):
    equation_results = np.zeros_like(x1)
    
    equation_results[:16] = x1[:16] / x2.iloc[0] * x3.iloc[0] / 100 * x4[:16]

    return equation_results

df_model_personal_vehicle_market['Lifetime Elec Use (kWh)'] = Lifetime_Elec_Use_Personal_Vehicle(df_model_personal_vehicle_market['Daily Elec VKT'], 
                                                                                                 df_consumer['Driving - mean (km)'], 
                                                                                                 df_consumer['Lifetime kilometers'],
                                                                                                 df_model['EC CD (kWh/100km)'])


# In[107]:


def Price_to_consumers_Personal_Vehicle(x1, x2, x3, x4, x5, x6, x7, x8):
    equation_results = np.zeros_like(x1)
    
    equation_results[:3] = x1[:3] * x2.iloc[0] - x3[:3] - x4[:3] - x5[:3] - x6[:3]
    equation_results[3:7] = x1[3:7] * x7.iloc[0] - np.minimum(x3[3:7] + x4[3:7] + x5[3:7], x1[3:7] * x7.iloc[0] * 0.7) - x6[3:7]
    equation_results[7:11] = x1[7:11] * x8.iloc[0] - x3[7:11] - x4[7:11] - x5[7:11] - x6[7:11]
    equation_results[11:15] = x1[11:15] * x7.iloc[0] - np.minimum(x3[11:15] + x4[11:15] + x5[11:15], x1[11:15] * x7.iloc[0] * 0.7) - x6[11:15]
    equation_results[15] = x1[15] * x7.iloc[0] - x3[15] - x4[15] - x5[15] - x6[15]

    return equation_results

df_model_personal_vehicle_market['Price to consumers (￥)'] = Price_to_consumers_Personal_Vehicle(df_model['Production Cost (￥)'],
                                                                                                  df_industry_vehicle_production['Sedan Markup'], 
                                                                                                  df_model['Additional Subsidy to Consumers (￥)'],
                                                                                                  df_model['Central Gov Sub to Consumers (￥)'],
                                                                                                  df_model['Avg Local Gov Sub to Consumers (￥)'],
                                                                                                  df_model['Internal Subsidy to Consumer(￥)'],
                                                                                                  df_industry_vehicle_production['PEV/FCEV Markup'],
                                                                                                  df_industry_vehicle_production['SUV Markup'])


# In[108]:


def Energy_Cost_Non_BEV(x1, x2, x3):
    equation_results = np.zeros_like(x1)
    
    equation_results[:4] = x1[:4] * x2.iloc[0]
    equation_results[4:6] = 0
    equation_results[7] = x1[7] * x3.iloc[0]
    equation_results[8:12] = x1[8:12] * x2.iloc[0]
    equation_results[12:15] = 0
    equation_results[15] = x1[15] * x3.iloc[0]

    return equation_results

df_model_personal_vehicle_market['Energy Cost (non-BEV)'] = Energy_Cost_Non_BEV(df_model_personal_vehicle_market['Lifetime Fuel Use (L)'],
                                                                                df_infrastructure_Gasoline_Station_Infrastructure['Gasoline price (￥/L)'],
                                                                                df_infrastructure_Hydrogen_Station_Infrastructure['Hydrogen price  (￥/Gasoline Equivalent Liter)'])


# "Infrastructure" Supplement

# In[109]:


def Number_of_Charging_for_BEV_Full_Range(x1, x2, x3, x4):
    equation_results = np.zeros(6)
    
    equation_results[:3] = np.ceil(x1[4:7] / (x2[4:7] * x3.iloc[0] * x4[4:7] / 100))
    equation_results[3:7] = np.ceil(x1[12:15] / (x2[12:15] * x3.iloc[0] * x4[12:15] / 100))
    
    return equation_results

df_infrastructure_Personal_Vehicle_Market_BEV['Number of charging for BEV full range'] = Number_of_Charging_for_BEV_Full_Range(df_model_personal_vehicle_market['Lifetime Elec Use (kWh)'],
                                                                                                                               df_model['Elec Range (km)'],
                                                                                                                               df_consumer['Range utility'],
                                                                                                                               df_model['EC CD (kWh/100km)'])


# In[110]:


def Lifetime_Costs_for_Driver_w_o_Home_parking(x1, x2, x3, x4, x5, x6, x7):
    equation_results = np.zeros(6)
    
    equation_results[:3] = x1[:3] * (x2.iloc[1] * x3[:3] + x4.iloc[1] * x5[:3] + x6.iloc[1] * x7[:3])
    equation_results[3:7] = x1[3:7] * (x2.iloc[1] * x3[3:7] + x4.iloc[1] * x5[3:7] + x6.iloc[1] * x7[3:7])
    
    return equation_results

df_infrastructure_Personal_Vehicle_Market_BEV['Lifetime costs for driver w/o home-parking'] = Lifetime_Costs_for_Driver_w_o_Home_parking(df_infrastructure_Personal_Vehicle_Market_BEV['Number of charging for BEV full range'],
                                                                                                                                         df_infrastructure_Consumer_Charging_Behavior['Residential parking'],
                                                                                                                                         df_infrastructure_Personal_Vehicle_Market_BEV['Costs for a full range @ home-charging (￥)'],
                                                                                                                                         df_infrastructure_Consumer_Charging_Behavior['Public-charging'],
                                                                                                                                         df_infrastructure_Personal_Vehicle_Market_BEV['Costs for a full range @ public-charging (￥)'],
                                                                                                                                         df_infrastructure_Consumer_Charging_Behavior['Workplaces'],
                                                                                                                                         df_infrastructure_Personal_Vehicle_Market_BEV['Costs for a full range @ workplace-charging (￥)'])


# In[111]:


def Lifetime_Costs_for_Driver_w_Home_parking(x1, x2, x3, x4, x5, x6, x7, x8):
    equation_results = np.zeros(6)
    
    equation_results[:3] = x1[:3] * (x2.iloc[0] * x3[:3] + x4.iloc[0] * x5[:3] + x6.iloc[0] * x7[:3]) + x8.iloc[0]
    equation_results[3:7] = x1[3:7] * (x2.iloc[0] * x3[3:7] + x4.iloc[0] * x5[3:7] + x6.iloc[0] * x7[3:7]) + x8.iloc[0]
    
    return equation_results

df_infrastructure_Personal_Vehicle_Market_BEV['Lifetime costs for driver with home-parking'] = Lifetime_Costs_for_Driver_w_Home_parking(df_infrastructure_Personal_Vehicle_Market_BEV['Number of charging for BEV full range'],
                                                                                                                                         df_infrastructure_Consumer_Charging_Behavior['Residential parking'],
                                                                                                                                         df_infrastructure_Personal_Vehicle_Market_BEV['Costs for a full range @ home-charging (￥)'],
                                                                                                                                         df_infrastructure_Consumer_Charging_Behavior['Public-charging'],
                                                                                                                                         df_infrastructure_Personal_Vehicle_Market_BEV['Costs for a full range @ public-charging (￥)'],
                                                                                                                                         df_infrastructure_Consumer_Charging_Behavior['Workplaces'],
                                                                                                                                         df_infrastructure_Personal_Vehicle_Market_BEV['Costs for a full range @ workplace-charging (￥)'],
                                                                                                                                         df_infrastructure_Charging_Infrastructure['Home charging cost (￥)'])


# In[112]:


def Electricity_Costs(x1, x2, x3):
    equation_results = np.zeros(2)
    
    equation_results[0] = x1[3] * x2.iloc[0] + x3.iloc[0]
    equation_results[1] = x1[11] * x2.iloc[0] + x3.iloc[0]
    
    return equation_results

df_infrastructure_Personal_Vehicle_Market_PHEV['Electricity costs (￥)'] = Electricity_Costs(df_model_personal_vehicle_market['Lifetime Elec Use (kWh)'],
                                                                                             df_infrastructure_Charging_Infrastructure['Home electricity price (￥/kWh)'],
                                                                                             df_infrastructure_Charging_Infrastructure['Home charging cost (￥)'])


# In[113]:


def Lifetime_Hydrogen_Use(x):
    equation_results = np.zeros(2)
    
    equation_results[0] = x[7] / 3.78541 / 1.019
    equation_results[1] = x[15] / 3.78541 / 1.019
    
    return equation_results

df_infrastructure_Personal_Vehicle_Market_FCEV['Lifetime Hydrogen Use (kg)'] = Lifetime_Hydrogen_Use(df_model_personal_vehicle_market['Lifetime Fuel Use (L)'])


# In[114]:


def Number_of_Charging_for_BEV_Full_Range(x1, x2):
    equation_results = np.zeros(2)
    
    equation_results[0] = np.ceil(x1.iloc[0] / x2.iloc[0])
    equation_results[1] = np.ceil(x1.iloc[1] / x2.iloc[1])

    return equation_results

df_infrastructure_Personal_Vehicle_Market_FCEV['Number of charging for BEV full range'] = Number_of_Charging_for_BEV_Full_Range(df_infrastructure_Personal_Vehicle_Market_FCEV['Lifetime Hydrogen Use (kg)'],
                                                                                                                                df_infrastructure_Personal_Vehicle_Market_FCEV['Tank size (kg)'])


# In[115]:


def Refilling_Inconvenience_Costs_Public(x1, x2):
    equation_results = np.zeros(2)
    
    equation_results[0] = x1.iloc[0] * x2.iloc[0]
    equation_results[1] = x1.iloc[1] * x2.iloc[1]
    
    return equation_results

df_infrastructure_Personal_Vehicle_Market_FCEV['Refilling Inconvenience Costs @ Public (￥/Lifetime)'] = Refilling_Inconvenience_Costs_Public(df_infrastructure_Personal_Vehicle_Market_FCEV['Refilling Inconvenience Costs @ Public (￥/Trip)'],
                                                                                                                                              df_infrastructure_Personal_Vehicle_Market_FCEV['Number of charging for BEV full range'])


# In[116]:


def Refuelling_Cost_Non_BEV(x1, x2, x3, x4):
    equation_results = np.zeros_like(x1)
    
    equation_results[:7] = x1[:7] / x2.iloc[0] * x3.iloc[0]
    equation_results[7] = x4.iloc[0]
    equation_results[8:15] = x1[8:15] / x2.iloc[0] * x3.iloc[0]
    equation_results[15] = x4.iloc[1]

    return equation_results

df_model_personal_vehicle_market['Refuelling Cost (non-BEV)'] = Refuelling_Cost_Non_BEV(df_model_personal_vehicle_market['Lifetime Fuel Use (L)'],
                                                                                        df_infrastructure_Gasoline_Station_Infrastructure['Gasoline Refill Amount (L)'],
                                                                                        df_infrastructure_Gasoline_Station_Infrastructure['Gasoline Refill Hassle (￥/each)'],
                                                                                        df_infrastructure_Personal_Vehicle_Market_FCEV['Refilling Inconvenience Costs @ Public (￥/Lifetime)'])


# In[117]:


def Total_Energy_Costs_1(x1, x2, x3):
    equation_results = np.zeros(2)
    
    equation_results[0] = x1[0] + x2[3] + x3[3]
    equation_results[1] = x1[1] + x2[11] + x3[11]
    
    return equation_results

df_infrastructure_Personal_Vehicle_Market_PHEV['Total energy costs (￥).1'] = Total_Energy_Costs_1(df_infrastructure_Personal_Vehicle_Market_PHEV['Electricity costs (￥)'],
                                                                                                   df_model_personal_vehicle_market['Energy Cost (non-BEV)'],
                                                                                                   df_model_personal_vehicle_market['Refuelling Cost (non-BEV)'])


# "Personal Vehicle Market" Supplement

# In[118]:


def Charg_Cost_PEV(x1, x2, x3, x4, x5):
    equation_results = np.zeros(16)
    
    equation_results[:3] = 0
    equation_results[3] = x1.iloc[0] * (1 - x2.iloc[0]) + x3.iloc[0] * x2.iloc[0]
    equation_results[4:7] = x4[:3] * (1 - x2.iloc[0]) + x5[:3] * x2.iloc[0]
    equation_results[7:11] = 0
    equation_results[11] = x1.iloc[1] * (1 - x2.iloc[0]) + x3.iloc[1] * x2.iloc[0]
    equation_results[12:15] = x4[3:6] * (1 - x2.iloc[0]) + x5[3:6] * x2.iloc[0]
    equation_results[15] = 0

    return equation_results

df_model_personal_vehicle_market['Charg Cost (PEV)'] = Charg_Cost_PEV(df_infrastructure_Personal_Vehicle_Market_PHEV['Total energy costs (￥)'],
                                                                      df_infrastructure_Charging_Infrastructure['Probability of owning a home-charging'],
                                                                      df_infrastructure_Personal_Vehicle_Market_PHEV['Total energy costs (￥).1'],
                                                                      df_infrastructure_Personal_Vehicle_Market_BEV['Lifetime costs for driver w/o home-parking'],
                                                                      df_infrastructure_Personal_Vehicle_Market_BEV['Lifetime costs for driver with home-parking'])


# In[119]:


def Range_Cost_BEV_Personal_Vehicle(x1, x2, x3, x4, x5):
    equation_results = np.zeros_like(x1)
    
    equation_results[:7] = np.where(x1[:7] == 0, (1 - x2[:7]) * x3.iloc[0] / x4.iloc[0] * x5.iloc[0], 0)
    equation_results[7] = (1 - x2[7]) * x3.iloc[0] / x4.iloc[0] * x5.iloc[0]
    equation_results[8:15] = np.where(x1[8:15] == 0, (1 - x2[8:15]) * x3.iloc[0] / x4.iloc[0] * x5.iloc[0], 0)
    equation_results[15] = (1 - x2[15]) * x3.iloc[0] / x4.iloc[0] * x5.iloc[0]
    
    return equation_results

df_model_personal_vehicle_market['Range Cost (BEV)'] = Range_Cost_BEV_Personal_Vehicle(df_model['FC CS Gasoline Equivalent (L/100km)'],
                                                                                       df_model_personal_vehicle_market['PF'], 
                                                                                       df_consumer['Daily Range limitation Cost for PEV/FCEV (￥)'],
                                                                                       df_consumer['Driving - mean (km)'],
                                                                                       df_consumer['Lifetime kilometers'])


# In[120]:


def Purchase_Privilage_Personal_Vehicle(x1, x2):
    idx = np.searchsorted(x2, Year)  
    idx = np.clip(idx, 0, len(x2) - 1)

    equation_results = np.zeros(16)
    
    equation_results[0] = 0
    equation_results[1] = 0
    equation_results[2] = 0
    equation_results[3] = x1[idx] / 2
    equation_results[4:8] = x1[idx]
    equation_results[8:10] = 0
    equation_results[10] = x1[idx] / 2
    equation_results[11:16] = x1[idx]

    return equation_results

df_model_personal_vehicle_market['Purchase Privilage'] = Purchase_Privilage_Personal_Vehicle(df_projection['Purchase Privilage'],
                                                                                             df_projection['Year'])


# In[121]:


def Constant_Personal_Vehicle(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15):
    idx = np.searchsorted(x2, Year)  
    idx = np.clip(idx, 0, len(x2) - 1)
    
    equation_results = np.zeros(16)
    
    equation_results[:2] = x1[idx]
    equation_results[2] = x3[idx]
    equation_results[3] = x4[idx]
    equation_results[4] = x5[idx]
    equation_results[5] = x6[idx]
    equation_results[6] = x7[idx]
    equation_results[7] = x8[idx]
    equation_results[8:10] = x9[idx]
    equation_results[10] = x10[idx]
    equation_results[11] = x11[idx]
    equation_results[12] = x12[idx]
    equation_results[13] = x13[idx]
    equation_results[14] = x14[idx]
    equation_results[15] = x15[idx]

    return equation_results

df_model_personal_vehicle_market['Constant'] = Constant_Personal_Vehicle(df_constant_personal_vehicles['ICEV'],
                                                                         df_constant_personal_vehicles['Year'],
                                                                         df_constant_personal_vehicles['ICEV-Low'],
                                                                         df_constant_personal_vehicles['PHEV'],
                                                                         df_constant_personal_vehicles['BEV-Sht'],
                                                                         df_constant_personal_vehicles['BEV-Mid'],
                                                                         df_constant_personal_vehicles['BEV-Lng'],
                                                                         df_constant_personal_vehicles['FCEV'],
                                                                         df_constant_personal_vehicles['ICEV.1'],
                                                                         df_constant_personal_vehicles['ICEV-Low.1'],
                                                                         df_constant_personal_vehicles['PHEV.1'],
                                                                         df_constant_personal_vehicles['BEV-Sht.1'],
                                                                         df_constant_personal_vehicles['BEV-Mid.1'],
                                                                         df_constant_personal_vehicles['BEV-Lng.1'],
                                                                         df_constant_personal_vehicles['FCEV.1'])


# In[122]:


def Ownership_Total_Cost_Personal_Vehicle(x1, x2, x3, x4, x5, x6, x7, x8):
    
    equation_results = np.zeros(16)
    
    equation_results[:3] = np.where(Year <= 2020,
                                x1[:3] + x2[:3] + x3[:3] + x4[:3] + x5[:3] + x6[:3] + x7[:3],
                                x1[:3] + x2[:3] + x3[:3] + x4[:3] + x5[:3] + x6[:3] + x8[:3])
    equation_results[3] = np.where(Year <= 2020,
                               x1[3] + x5[3] + x6[3] + x7[3],
                               x1[3] + x5[3] + x6[3] + x8[3])
    equation_results[4:11] = np.where(Year <= 2020,
                                  x1[4:11] + x2[4:11] + x3[4:11] + x4[4:11] + x5[4:11] + x6[4:11] + x7[4:11],
                                  x1[4:11] + x2[4:11] + x3[4:11] + x4[4:11] + x5[4:11] + x6[4:11] + x8[4:11])
    equation_results[11] = np.where(Year <= 2020,
                                x1[11] + x5[11] + x6[11] + x7[11],
                                x1[11] + x5[11] + x6[11] + x8[11])
    equation_results[12:16] = np.where(Year <= 2020,
                                   x1[12:16] + x2[12:16] + x3[12:16] + x4[12:16] + x5[12:16] + x6[12:16] + x7[12:16],
                                   x1[12:16] + x2[12:16] + x3[12:16] + x4[12:16] + x5[12:16] + x6[12:16] + x8[12:16])

    return equation_results

df_model_personal_vehicle_market['Ownership Total Cost'] = Ownership_Total_Cost_Personal_Vehicle(df_model_personal_vehicle_market['Price to consumers (￥)'],
                                                                                                 df_model_personal_vehicle_market['Energy Cost (non-BEV)'],
                                                                                                 df_model_personal_vehicle_market['Refuelling Cost (non-BEV)'],
                                                                                                 df_model_personal_vehicle_market['Range Cost (BEV)'],
                                                                                                 df_model_personal_vehicle_market['Charg Cost (PEV)'],
                                                                                                 df_model_personal_vehicle_market['Purchase Privilage'],
                                                                                                 df_model_personal_vehicle_market['Calibrated Other Cost'],
                                                                                                 df_model_personal_vehicle_market['Constant'])


# In[123]:


EXP = np.exp
LN = np.log

df_DCM_ExpUSum_personal_vehicle = pd.DataFrame({
    'Sedan-ExpUSum_IC': EXP(df_DCM_personal_vehicle['4_Technology_IC'].iloc[0]
                            * float(df_model_personal_vehicle_market['Ownership Total Cost'].iloc[0])) 
    + EXP(df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * float(df_model_personal_vehicle_market['Ownership Total Cost'].iloc[1])) 
    + EXP(df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * float(df_model_personal_vehicle_market['Ownership Total Cost'].iloc[2])),
    
    'Sedan-ExpUSum_PH': EXP(df_DCM_personal_vehicle['4_Technology_PH'].iloc[0]
                            * float(df_model_personal_vehicle_market['Ownership Total Cost'].iloc[3])),
    
    'Sedan-ExpUSum_EV': EXP(df_DCM_personal_vehicle['4_Technology_EV'].iloc[0] 
                            * float(df_model_personal_vehicle_market['Ownership Total Cost'].iloc[4])) 
    + EXP(df_DCM_personal_vehicle['4_Technology_EV'].iloc[0] * float(df_model_personal_vehicle_market['Ownership Total Cost'].iloc[5])) 
    + EXP(df_DCM_personal_vehicle['4_Technology_EV'].iloc[0] * float(df_model_personal_vehicle_market['Ownership Total Cost'].iloc[6])),
    
    'Sedan-ExpUSum_FC': EXP(df_DCM_personal_vehicle['4_Technology_FC'].iloc[0] 
                            * float(df_model_personal_vehicle_market['Ownership Total Cost'].iloc[7])),
    
    'SUV-ExpUSum_IC': EXP(df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] 
                          * float(df_model_personal_vehicle_market['Ownership Total Cost'].iloc[8])) 
    + EXP(df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * float(df_model_personal_vehicle_market['Ownership Total Cost'].iloc[9])) 
    + EXP(df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * float(df_model_personal_vehicle_market['Ownership Total Cost'].iloc[10])),
    
    'SUV-ExpUSum_PH': EXP(df_DCM_personal_vehicle['4_Technology_PH'].iloc[0] 
                          * float(df_model_personal_vehicle_market['Ownership Total Cost'].iloc[11])),
    
    'SUV-ExpUSum_EV': EXP(df_DCM_personal_vehicle['4_Technology_EV'].iloc[0] 
                          * float(df_model_personal_vehicle_market['Ownership Total Cost'].iloc[12])) 
    + EXP(df_DCM_personal_vehicle['4_Technology_EV'].iloc[0] * float(df_model_personal_vehicle_market['Ownership Total Cost'].iloc[13])) 
    + EXP(df_DCM_personal_vehicle['4_Technology_EV'].iloc[0] * float(df_model_personal_vehicle_market['Ownership Total Cost'].iloc[14])),
    
    'SUV-ExpUSum_FC': EXP(df_DCM_personal_vehicle['4_Technology_FC'].iloc[0] 
                          * float(df_model_personal_vehicle_market['Ownership Total Cost'].iloc[15])),
}, index=[0])


# In[124]:


EXP = np.exp
LN = np.log

df_DCM_ExpUSum_personal_vehicle['ExpUSum_Sedan'] = (
    EXP(LN(df_DCM_ExpUSum_personal_vehicle['Sedan-ExpUSum_IC'].iloc[0]) / df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_DCM_personal_vehicle['3_Powertrain_Conv'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_personal_vehicle['Sedan-ExpUSum_PH'].iloc[0]) / df_DCM_personal_vehicle['4_Technology_PH'].iloc[0] * df_DCM_personal_vehicle['3_Powertrain_Conv'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_personal_vehicle['Sedan-ExpUSum_EV'].iloc[0]) / df_DCM_personal_vehicle['4_Technology_EV'].iloc[0] * df_DCM_personal_vehicle['3_Powertrain_Electricity'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_personal_vehicle['Sedan-ExpUSum_FC'].iloc[0]) / df_DCM_personal_vehicle['4_Technology_FC'].iloc[0] * df_DCM_personal_vehicle['3_Powertrain_Hydrogen'].iloc[0])
)

df_DCM_ExpUSum_personal_vehicle['ExpUSum_SUV_Crossover'] = (
    EXP(LN(df_DCM_ExpUSum_personal_vehicle['SUV-ExpUSum_IC'].iloc[0]) / df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_DCM_personal_vehicle['3_Powertrain_Conv'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_personal_vehicle['SUV-ExpUSum_PH'].iloc[0]) / df_DCM_personal_vehicle['4_Technology_PH'].iloc[0] * df_DCM_personal_vehicle['3_Powertrain_Conv'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_personal_vehicle['SUV-ExpUSum_EV'].iloc[0]) / df_DCM_personal_vehicle['4_Technology_EV'].iloc[0] * df_DCM_personal_vehicle['3_Powertrain_Electricity'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_personal_vehicle['SUV-ExpUSum_FC'].iloc[0]) / df_DCM_personal_vehicle['4_Technology_FC'].iloc[0] * df_DCM_personal_vehicle['3_Powertrain_Hydrogen'].iloc[0])
)

df_DCM_ExpUSum_personal_vehicle['ExpUSum_Sedan-BuyAll'] = (
    EXP(LN(df_DCM_ExpUSum_personal_vehicle['Sedan-ExpUSum_IC'].iloc[0]) / df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_DCM_personal_vehicle['3_Powertrain_Conv'].iloc[0]) +
    EXP(LN(EXP(df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_model_personal_vehicle_market['Ownership Total Cost'].iloc[3])) / df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_DCM_personal_vehicle['3_Powertrain_Conv'].iloc[0]) +
    EXP(LN(
        EXP(df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_model_personal_vehicle_market['Ownership Total Cost'].iloc[4]) +
        EXP(df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_model_personal_vehicle_market['Ownership Total Cost'].iloc[5]) +
        EXP(df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_model_personal_vehicle_market['Ownership Total Cost'].iloc[6])
    ) / df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_DCM_personal_vehicle['3_Powertrain_Conv'].iloc[0]) +
    EXP(LN(EXP(df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_model_personal_vehicle_market['Ownership Total Cost'].iloc[7])) / df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_DCM_personal_vehicle['3_Powertrain_Conv'].iloc[0])
)

df_DCM_ExpUSum_personal_vehicle['ExpUSum_SUV_Crossover-BuyAll'] = (
    EXP(LN(df_DCM_ExpUSum_personal_vehicle['SUV-ExpUSum_IC'].iloc[0]) / df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_DCM_personal_vehicle['3_Powertrain_Conv'].iloc[0]) + 
    EXP(LN(EXP(df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_model_personal_vehicle_market['Ownership Total Cost'].iloc[11])) / df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_DCM_personal_vehicle['3_Powertrain_Conv'].iloc[0]) +
    EXP(LN(
        EXP(df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_model_personal_vehicle_market['Ownership Total Cost'].iloc[12]) +
        EXP(df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_model_personal_vehicle_market['Ownership Total Cost'].iloc[13]) +
        EXP(df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_model_personal_vehicle_market['Ownership Total Cost'].iloc[14])
    ) / df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_DCM_personal_vehicle['3_Powertrain_Conv'].iloc[0]) +
    EXP(LN(EXP(df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_model_personal_vehicle_market['Ownership Total Cost'].iloc[15])) / df_DCM_personal_vehicle['4_Technology_IC'].iloc[0] * df_DCM_personal_vehicle['3_Powertrain_Conv'].iloc[0])
)

df_DCM_ExpUSum_personal_vehicle['All-ExpUSum_Personal vehicle'] = (
    EXP(LN(df_DCM_ExpUSum_personal_vehicle['ExpUSum_Sedan'].iloc[0]) / df_DCM_personal_vehicle['3_Powertrain_Conv'].iloc[0] * df_DCM_personal_vehicle['2_Class_Sedan'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_personal_vehicle['ExpUSum_Sedan'].iloc[0]) / df_DCM_personal_vehicle['3_Powertrain_Electricity'].iloc[0] * df_DCM_personal_vehicle['2_Class_Sedan'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_personal_vehicle['ExpUSum_Sedan'].iloc[0]) / df_DCM_personal_vehicle['3_Powertrain_Hydrogen'].iloc[0] * df_DCM_personal_vehicle['2_Class_Sedan'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_personal_vehicle['ExpUSum_SUV_Crossover'].iloc[0]) / df_DCM_personal_vehicle['3_Powertrain_Conv'].iloc[0] * df_DCM_personal_vehicle['2_Class_SUV/Crossover'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_personal_vehicle['ExpUSum_SUV_Crossover'].iloc[0]) / df_DCM_personal_vehicle['3_Powertrain_Electricity'].iloc[0] * df_DCM_personal_vehicle['2_Class_SUV/Crossover'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_personal_vehicle['ExpUSum_SUV_Crossover'].iloc[0]) / df_DCM_personal_vehicle['3_Powertrain_Hydrogen'].iloc[0] * df_DCM_personal_vehicle['2_Class_SUV/Crossover'].iloc[0])
)

df_DCM_ExpUSum_personal_vehicle['All-ExpUSum_Personal vehicle (Main Stream)'] = (
    EXP(LN(df_DCM_ExpUSum_personal_vehicle['ExpUSum_Sedan-BuyAll'].iloc[0]) / df_DCM_personal_vehicle['3_Powertrain_Conv'].iloc[0] * df_DCM_personal_vehicle['2_Class_Sedan'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_personal_vehicle['ExpUSum_SUV_Crossover-BuyAll'].iloc[0]) / df_DCM_personal_vehicle['3_Powertrain_Conv'].iloc[0] * df_DCM_personal_vehicle['2_Class_SUV/Crossover'].iloc[0])
)

df_DCM_ExpUSum_personal_vehicle['Buy-ExpUSum'] = (
    EXP(LN(df_DCM_ExpUSum_personal_vehicle['All-ExpUSum_Personal vehicle (Main Stream)'].iloc[0]) / df_DCM_personal_vehicle['1_Class'].iloc[0] * df_DCM_personal_vehicle['Buy/NoBuy'].iloc[0]) + 1
)

df_DCM_ExpUSum_personal_vehicle['PBuy'] = 1 - 1 / df_DCM_ExpUSum_personal_vehicle['Buy-ExpUSum'].iloc[0]


# In[125]:


df_DCM_ExpUSum_personal_vehicle


# In[126]:


EXP = np.exp
LN = np.log

def Pcosteff_Personal_Vehicle(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
    equation_results = np.zeros_like(x2)
    
    equation_results[0:3] = EXP(x1.iloc[0] * x2[0:3]) / x3.iloc[0]
    equation_results[3] = EXP(x4.iloc[0] * x2[3]) / x5.iloc[0]
    equation_results[4:7] = EXP(x6.iloc[0] * x2[4:7]) / x7.iloc[0]
    equation_results[7] = EXP(x8.iloc[0] * x2[7]) / x9.iloc[0]
    equation_results[8:11] = EXP(x1.iloc[0] * x2[8:11]) / x10.iloc[0]
    equation_results[11] = EXP(x4.iloc[0] * x2[11]) / x11.iloc[0]
    equation_results[12:15] = EXP(x6.iloc[0] * x2[12:15]) / x12.iloc[0]
    equation_results[15] = EXP(x8.iloc[0] * x2[15]) / x13.iloc[0]

    return equation_results

df_model_personal_vehicle_market['Pcosteff'] = Pcosteff_Personal_Vehicle(df_DCM_personal_vehicle['4_Technology_IC'],
                                                                         df_model_personal_vehicle_market['Ownership Total Cost'],
                                                                         df_DCM_ExpUSum_personal_vehicle['Sedan-ExpUSum_IC'],
                                                                         df_DCM_personal_vehicle['4_Technology_PH'],
                                                                         df_DCM_ExpUSum_personal_vehicle['Sedan-ExpUSum_PH'],
                                                                         df_DCM_personal_vehicle['4_Technology_EV'],
                                                                         df_DCM_ExpUSum_personal_vehicle['Sedan-ExpUSum_EV'],
                                                                         df_DCM_personal_vehicle['4_Technology_FC'],
                                                                         df_DCM_ExpUSum_personal_vehicle['Sedan-ExpUSum_FC'],
                                                                         df_DCM_ExpUSum_personal_vehicle['SUV-ExpUSum_IC'],
                                                                         df_DCM_ExpUSum_personal_vehicle['SUV-ExpUSum_PH'],
                                                                         df_DCM_ExpUSum_personal_vehicle['SUV-ExpUSum_EV'],
                                                                         df_DCM_ExpUSum_personal_vehicle['SUV-ExpUSum_FC'])


# In[127]:


EXP = np.exp
LN = np.log

def Pptrain_Personal_Vehicle(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17):
    
    equation_results = np.zeros(16)
    
    equation_results[0:3] = EXP(LN(x1.iloc[0]) / x2.iloc[0] * x3.iloc[0]) / x4.iloc[0]
    equation_results[3] = EXP(LN(x5.iloc[0]) / x6.iloc[0] * x3.iloc[0]) / x4.iloc[0]
    equation_results[4:7] = EXP(LN(x7.iloc[0])/ x8.iloc[0] * x9.iloc[0]) / x4.iloc[0]
    equation_results[7] = EXP(LN(x10.iloc[0]) / x11.iloc[0] * x12.iloc[0]) / x4.iloc[0]
    equation_results[8:11] = EXP(LN(x13.iloc[0]) / x2.iloc[0] * x3.iloc[0]) / x14.iloc[0]
    equation_results[11] = EXP(LN(x15.iloc[0]) / x6.iloc[0] * x3.iloc[0]) / x14.iloc[0]
    equation_results[12:15] = EXP(LN(x16.iloc[0]) / x8.iloc[0] * x9.iloc[0]) / x14.iloc[0]
    equation_results[15] = EXP(LN(x17.iloc[0]) / x11.iloc[0] * x12.iloc[0]) / x14.iloc[0]

    return equation_results

df_model_personal_vehicle_market['Pptrain'] = Pptrain_Personal_Vehicle(df_DCM_ExpUSum_personal_vehicle['Sedan-ExpUSum_IC'],
                                                                       df_DCM_personal_vehicle['4_Technology_IC'],
                                                                       df_DCM_personal_vehicle['3_Powertrain_Conv'],
                                                                       df_DCM_ExpUSum_personal_vehicle['ExpUSum_Sedan'],
                                                                       df_DCM_ExpUSum_personal_vehicle['Sedan-ExpUSum_PH'],
                                                                       df_DCM_personal_vehicle['4_Technology_PH'],
                                                                       df_DCM_ExpUSum_personal_vehicle['Sedan-ExpUSum_EV'],
                                                                       df_DCM_personal_vehicle['4_Technology_EV'],
                                                                       df_DCM_personal_vehicle['3_Powertrain_Electricity'],
                                                                       df_DCM_ExpUSum_personal_vehicle['Sedan-ExpUSum_FC'],
                                                                       df_DCM_personal_vehicle['4_Technology_FC'],
                                                                       df_DCM_personal_vehicle['3_Powertrain_Hydrogen'],
                                                                       df_DCM_ExpUSum_personal_vehicle['SUV-ExpUSum_IC'],
                                                                       df_DCM_ExpUSum_personal_vehicle['ExpUSum_SUV_Crossover'],
                                                                       df_DCM_ExpUSum_personal_vehicle['SUV-ExpUSum_PH'],
                                                                       df_DCM_ExpUSum_personal_vehicle['SUV-ExpUSum_EV'],
                                                                       df_DCM_ExpUSum_personal_vehicle['SUV-ExpUSum_FC'])


# In[128]:


EXP = np.exp
LN = np.log

def Pclass_Personal_Vehicle(x1, x2, x3, x4, x5, x6, x7, x8):
    
    equation_results = np.zeros(16)
    
    equation_results[0:8] = (EXP(LN(x1.iloc[0]) / x2.iloc[0] * x3.iloc[0]) + EXP(LN(x1.iloc[0]) / x4.iloc[0] * x3.iloc[0]) + EXP(LN(x1.iloc[0]) / x5.iloc[0] * x3.iloc[0])) / x6.iloc[0]
    equation_results[8:16] = (EXP(LN(x7.iloc[0]) / x2.iloc[0] * x8.iloc[0]) + EXP(LN(x7.iloc[0]) / x4.iloc[0] * x8.iloc[0]) + EXP(LN(x7.iloc[0]) / x5.iloc[0] * x8.iloc[0])) / x6.iloc[0]

    return equation_results

df_model_personal_vehicle_market['Pclass'] = Pclass_Personal_Vehicle(df_DCM_ExpUSum_personal_vehicle['ExpUSum_Sedan'],
                                                                     df_DCM_personal_vehicle['3_Powertrain_Conv'],
                                                                     df_DCM_personal_vehicle['2_Class_Sedan'],
                                                                     df_DCM_personal_vehicle['3_Powertrain_Electricity'],
                                                                     df_DCM_personal_vehicle['3_Powertrain_Hydrogen'],
                                                                     df_DCM_ExpUSum_personal_vehicle['All-ExpUSum_Personal vehicle'],
                                                                     df_DCM_ExpUSum_personal_vehicle['ExpUSum_SUV_Crossover'],
                                                                     df_DCM_personal_vehicle['2_Class_SUV/Crossover'])


# In[129]:


def Market_Share_Personal_Vehicle(x1, x2, x3):
    
    equation_results = np.zeros(16)
    
    equation_results[0:16] = x1[0:16] * x2[0:16] * x3[0:16]

    return equation_results

df_model_personal_vehicle_market['Market share'] = Market_Share_Personal_Vehicle(df_model_personal_vehicle_market['Pcosteff'],
                                                                                 df_model_personal_vehicle_market['Pptrain'],
                                                                                 df_model_personal_vehicle_market['Pclass'])


# In[130]:


df_model_output = pd.DataFrame({
    'Veh Total Sales (1,000)': round(df_consumer['Household numbers'].iloc[0] * df_DCM_ExpUSum_personal_vehicle['PBuy'] / df_industry_vehicle_production['Personal purchase share in sales market'].iloc[0] / 1000, 3),
    'CAFC Target Multiplier': df_projection.loc[df_projection['Year'] == Year, 'CAFC Target Multiplier'].values[0],
    'NEV Target %': df_projection.loc[df_projection['Year'] == Year, 'NEV Target'].values[0],
    'CAFC/NEV credit conv': 1,
    
    'NEV Credits price (￥)': 0
}, index=[0]) 


# In[131]:


def Sales_Personal_Vehicle(x1, x2, x3):
    
    equation_results = np.zeros(16)
    
    equation_results[0:16] = x1.iloc[0] * x2.iloc[0] * x3[0:16]

    return equation_results

df_model_personal_vehicle_market['Sales (1000)'] = Sales_Personal_Vehicle(df_model_output['Veh Total Sales (1,000)'],
                                                                          df_industry_vehicle_production['Personal purchase share in sales market'],
                                                                          df_model_personal_vehicle_market['Market share'])


# In[132]:


df_model_personal_vehicle_market.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x)


# Fleet Vehicle Market

# In[133]:


def UF_Fleet_Vehicle(x, adjust):
    scale = df_consumer['Range Extension'].iloc[1] * adjust / df_consumer['Gamma-Scale-thetaBeta'].iloc[1]
    shape = df_consumer['Gamma-Shape-kAlpha'].iloc[1] + 1
    return gamma.cdf(x * scale, a=shape)

adjustments = [1, 1, 1, df_consumer['Range utility'].iloc[1], df_consumer['Range utility'].iloc[1], df_consumer['Range utility'].iloc[1], 
               df_consumer['Range utility'].iloc[1], 1, 1, 1, 1, df_consumer['Range utility'].iloc[1], df_consumer['Range utility'].iloc[1], 
               df_consumer['Range utility'].iloc[1], df_consumer['Range utility'].iloc[1], 1]

df_model_fleet_vehicle_market['UF'] = [UF_Fleet_Vehicle(df_model['Elec Range (km)'].iloc[i], adjustments[i]) for i in range(len(df_model['Elec Range (km)']))]


# In[134]:


def PF_Fleet_Vehicle(x, adjust):
    scale = df_consumer['Range Extension'].iloc[1] * adjust / df_consumer['Gamma-Scale-thetaBeta'].iloc[1]
    shape = df_consumer['Gamma-Shape-kAlpha'].iloc[1]
    return gamma.cdf(x * scale, a=shape)

adjustments = [1, 1, 1, df_consumer['Range utility'].iloc[1], df_consumer['Range utility'].iloc[1], df_consumer['Range utility'].iloc[1], 
               df_consumer['Range utility'].iloc[1], 1, 1, 1, 1, df_consumer['Range utility'].iloc[1], df_consumer['Range utility'].iloc[1], 
               df_consumer['Range utility'].iloc[1], df_consumer['Range utility'].iloc[1], 1]

df_model_fleet_vehicle_market['PF'] = [PF_Fleet_Vehicle(df_model['Elec Range (km)'].iloc[i], adjustments[i]) for i in range(len(df_model['Elec Range (km)']))]


# In[135]:


def Daily_Fuel_VKT_Fleet_Vehicle(x1, x2, x3):
    equation_results = np.zeros_like(x1)

    equation_results[:2] = df_consumer['Driving - mean (km)'].iloc[1] * (1 - x1[:2]) - x3[:2] * df_consumer['Range Extension'].iloc[1] * (1 - x2[:2])
    equation_results[2:7] = df_consumer['Driving - mean (km)'].iloc[1] * (1 - x1[2:7]) - x3[2:7] * df_consumer['Range Extension'].iloc[1] * df_consumer['Range utility'].iloc[0] * (1 - x2[2:7])
    equation_results[7] = df_consumer['Driving - mean (km)'].iloc[1] * x1[7] + x3[7] * df_consumer['Range Extension'].iloc[1] * (1 - x2[7])
    equation_results[8:10] = df_consumer['Driving - mean (km)'].iloc[1] * (1 - x1[8:10]) - x3[8:10] * df_consumer['Range Extension'].iloc[1] * (1 - x2[8:10])
    equation_results[10:15] = df_consumer['Driving - mean (km)'].iloc[1] * (1 - x1[10:15]) - x3[10:15] * df_consumer['Range Extension'].iloc[1] * df_consumer['Range utility'].iloc[0] * (1 - x2[10:15])
    equation_results[15] = df_consumer['Driving - mean (km)'].iloc[1] * x1[15] + x3[15] * df_consumer['Range Extension'].iloc[1] * (1 - x2[15])
    return equation_results

df_model_fleet_vehicle_market['Daily Fuel VKT'] = Daily_Fuel_VKT_Fleet_Vehicle(df_model_fleet_vehicle_market['UF'], df_model_fleet_vehicle_market['PF'], df_model['Elec Range (km)'])


# In[136]:


def Daily_Elec_VKT_Fleet_Vehicle(x1, x2, x3):
    equation_results = np.zeros_like(x1)

    equation_results[:2] = df_consumer['Driving - mean (km)'].iloc[1] * x1[:2] + x3[:2] * df_consumer['Range Extension'].iloc[1] * (1 - x2[:2])
    equation_results[2:7] = df_consumer['Driving - mean (km)'].iloc[1] * x1[2:7] + x3[2:7] * df_consumer['Range Extension'].iloc[1] * df_consumer['Range utility'].iloc[1] * (1 - x2[2:7])
    equation_results[7] = 0
    equation_results[8:10] = df_consumer['Driving - mean (km)'].iloc[1] * x1[8:10] + x3[8:10] * df_consumer['Range Extension'].iloc[1] * (1 - x2[8:10])
    equation_results[10:15] = df_consumer['Driving - mean (km)'].iloc[1] * x1[10:15] + x3[10:15] * df_consumer['Range Extension'].iloc[1] * df_consumer['Range utility'].iloc[1] * (1 - x2[10:15])
    equation_results[15] = 0
    return equation_results

df_model_fleet_vehicle_market['Daily Elec VKT'] = Daily_Elec_VKT_Fleet_Vehicle(df_model_fleet_vehicle_market['UF'], df_model_fleet_vehicle_market['PF'], df_model['Elec Range (km)'])


# In[137]:


def Lifetime_Fuel_Use_Fleet_Vehicle(x1, x2, x3, x4):
    equation_results = np.zeros_like(x1)
    
    equation_results[:16] = x1[:16] / x2.iloc[1] * x3.iloc[1] / 100 * x4[:16]

    return equation_results

df_model_fleet_vehicle_market['Lifetime Fuel Use (L)'] = Lifetime_Fuel_Use_Fleet_Vehicle(df_model_fleet_vehicle_market['Daily Fuel VKT'], 
                                                                                         df_consumer['Driving - mean (km)'], 
                                                                                         df_consumer['Lifetime kilometers'],
                                                                                         df_model['FC CS Gasoline Equivalent (L/100km)'])


# In[138]:


def Lifetime_Elec_Use_Fleet_Vehicle(x1, x2, x3, x4):
    equation_results = np.zeros_like(x1)
    
    equation_results[:16] = x1[:16] / x2.iloc[1] * x3.iloc[1] / 100 * x4[:16]

    return equation_results

df_model_fleet_vehicle_market['Lifetime Elec Use (kWh)'] = Lifetime_Elec_Use_Fleet_Vehicle(df_model_fleet_vehicle_market['Daily Elec VKT'], 
                                                                                           df_consumer['Driving - mean (km)'], 
                                                                                           df_consumer['Lifetime kilometers'],
                                                                                           df_model['EC CD (kWh/100km)'])


# In[139]:


def Price_to_consumers_Fleet_Vehicle(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    equation_results = np.zeros_like(x1)
    
    equation_results[:3] = x1[:3] * x2.iloc[0] * x9.iloc[0] - x3[:3] - x4[:3] - x5[:3] - x6[:3]
    equation_results[3:7] = x1[3:7] * x7.iloc[0] * x9.iloc[0] - np.minimum(x3[3:7] + x4[3:7] + x5[3:7], x1[3:7] * x7.iloc[0] * x9.iloc[0] * 0.7) - x6[3:7]
    equation_results[7:11] = x1[7:11] * x8.iloc[0] * x9.iloc[0] - x3[7:11] - x4[7:11] - x5[7:11] - x6[7:11]
    equation_results[11:15] = x1[11:15] * x7.iloc[0] * x9.iloc[0] - np.minimum(x3[11:15] + x4[11:15] + x5[11:15], x1[11:15] * x7.iloc[0] * x9.iloc[0] * 0.7) - x6[11:15]
    equation_results[15] = x1[15] * x7.iloc[0] * x9.iloc[0] - x3[15] - x4[15] - x5[15] - x6[15]

    return equation_results

df_model_fleet_vehicle_market['Price to consumers (￥)'] = Price_to_consumers_Fleet_Vehicle(df_model['Production Cost (￥)'],
                                                                                            df_industry_vehicle_production['Sedan Markup'],
                                                                                            df_model['Additional Subsidy to Consumers (￥)'],
                                                                                            df_model['Central Gov Sub to Consumers (￥)'],
                                                                                            df_model['Avg Local Gov Sub to Consumers (￥)'],
                                                                                            df_model['Internal Subsidy to Consumer(￥)'],
                                                                                            df_industry_vehicle_production['PEV/FCEV Markup'],
                                                                                            df_industry_vehicle_production['SUV Markup'],
                                                                                            df_industry_vehicle_production['Fleet Vehicle Purchase Discount'])


# In[140]:


def Energy_Cost_Non_BEV(x1, x2, x3):
    equation_results = np.zeros_like(x1)
    
    equation_results[:4] = x1[:4] * x2.iloc[0]
    equation_results[4:6] = 0
    equation_results[7] = x1[7] * x3.iloc[0]
    equation_results[8:12] = x1[8:12] * x2.iloc[0]
    equation_results[12:15] = 0
    equation_results[15] = x1[15] * x3.iloc[0]

    return equation_results

df_model_fleet_vehicle_market['Energy Cost (non-BEV)'] = Energy_Cost_Non_BEV(df_model_fleet_vehicle_market['Lifetime Fuel Use (L)'],
                                                                             df_infrastructure_Gasoline_Station_Infrastructure['Gasoline price (￥/L)'],
                                                                             df_infrastructure_Hydrogen_Station_Infrastructure['Hydrogen price  (￥/Gasoline Equivalent Liter)'])


# In[141]:


def Electricity_Costs(x1, x2, x3, x4, x5, x6):
    equation_results = np.zeros(2)
    
    equation_results[0] = x1[3] * ((x2.iloc[0] + x3.iloc[0]) * x4.iloc[0] + (x2.iloc[0] + x5.iloc[0]) * x6.iloc[0])
    equation_results[1] = x1[11] * ((x2.iloc[0] + x3.iloc[0]) * x4.iloc[0] + (x2.iloc[0] + x5.iloc[0]) * x6.iloc[0])
    
    return equation_results

df_infrastructure_Fleet_Vehicle_Market_PHEV['Electricity costs (￥)'] = Electricity_Costs(df_model_fleet_vehicle_market['Lifetime Elec Use (kWh)'],
                                                                                          df_infrastructure_Charging_Infrastructure['Public charging electric price  (￥/kWh)'],
                                                                                          df_infrastructure_Charging_Infrastructure['Fast charging service (￥/kWh)'],
                                                                                          df_infrastructure_Charging_Infrastructure['Fast charging possibility'],
                                                                                          df_infrastructure_Charging_Infrastructure['Slow charging service (￥/kWh)'],
                                                                                          df_infrastructure_Charging_Infrastructure['Slow charging possiblity'])


# In[142]:


def Tank_Size(x1, x2):
    equation_results = np.zeros(2)
    
    equation_results[0] = x1[7] * x2[7] / 100 / 3.78541 / 1.019
    equation_results[1] = x1[15] * x2[15] / 100 / 3.78541 / 1.019
    
    return equation_results

df_infrastructure_Fleet_Vehicle_Market_FCEV['Tank size (kg)'] = Tank_Size(df_model['Elec Range (km)'],
                                                                             df_model['FC CS Gasoline Equivalent (L/100km)'])


# In[143]:


def Refilling_Inconvenience_Costs_Public(x1, x2, x3, x4, x5, x6, x7):
    equation_results = np.zeros(2)
    
    equation_results[0] = (x1.iloc[0] * x2.iloc[0] + x3.iloc[0] * (1 - x4.iloc[0]) / x5.iloc[0] + x6.iloc[0]) * x7.iloc[0] / 60
    equation_results[1] = (x1.iloc[0] * x2.iloc[0] + x3.iloc[1] * (1 - x4.iloc[0]) / x5.iloc[0] + x6.iloc[0]) * x7.iloc[0] / 60
    
    return equation_results

df_infrastructure_Fleet_Vehicle_Market_FCEV['Refilling Inconvenience Costs @ Public (￥/Trip)'] = Refilling_Inconvenience_Costs_Public(df_infrastructure_Hydrogen_Station_Infrastructure['Refueling annoyance multiplier'],
                                                                                                                                          df_infrastructure_Hydrogen_Station_Infrastructure['Round trip time (home to public stations, mins)'],
                                                                                                                                          df_infrastructure_Personal_Vehicle_Market_FCEV['Tank size (kg)'],
                                                                                                                                          df_infrastructure_Hydrogen_Station_Infrastructure['Tank state at beginning of refill (%)'],
                                                                                                                                          df_infrastructure_Hydrogen_Station_Infrastructure['Filling speed (kg/min)'],
                                                                                                                                          df_infrastructure_Hydrogen_Station_Infrastructure['Refueling time, i.e. time on operating the dispenser (mins)'],
                                                                                                                                          df_consumer['Time value (￥/hour)'])


# In[144]:


def Lifetime_Hydrogen_Use(x):
    equation_results = np.zeros(2)
    
    equation_results[0] = x[7] / 3.78541 / 1.019
    equation_results[1] = x[15] / 3.78541 / 1.019
    
    return equation_results

df_infrastructure_Fleet_Vehicle_Market_FCEV['Lifetime Hydrogen Use (kg)'] = Lifetime_Hydrogen_Use(df_model_fleet_vehicle_market['Lifetime Fuel Use (L)'])


# In[145]:


def Number_of_Charging_for_BEV_Full_Range(x1, x2):
    equation_results = np.zeros(2)
    
    equation_results[0] = np.ceil(x1.iloc[0] / x2.iloc[0])
    equation_results[1] = np.ceil(x1.iloc[1] / x2.iloc[1])
    
    return equation_results

df_infrastructure_Fleet_Vehicle_Market_FCEV['Number of charging for BEV full range'] = Number_of_Charging_for_BEV_Full_Range(df_infrastructure_Fleet_Vehicle_Market_FCEV['Lifetime Hydrogen Use (kg)'],
                                                                                                                             df_infrastructure_Fleet_Vehicle_Market_FCEV['Tank size (kg)'])


# In[146]:


def Refilling_Inconvenience_Costs_Public(x1, x2):
    equation_results = np.zeros(2)
    
    equation_results[0] = x1.iloc[0] * x2.iloc[0]
    equation_results[1] = x1.iloc[1] * x2.iloc[1]
    
    return equation_results

df_infrastructure_Fleet_Vehicle_Market_FCEV['Refilling Inconvenience Costs @ Public (￥/Lifetime)'] = Refilling_Inconvenience_Costs_Public(df_infrastructure_Fleet_Vehicle_Market_FCEV['Refilling Inconvenience Costs @ Public (￥/Trip)'],
                                                                                                                                           df_infrastructure_Fleet_Vehicle_Market_FCEV['Number of charging for BEV full range'])


# In[147]:


def Refuelling_Cost_Non_BEV(x1, x2, x3, x4):
    equation_results = np.zeros_like(x1)
    
    equation_results[:7] = x1[:7] / x2.iloc[0] * x3.iloc[0]
    equation_results[7] = x4.iloc[0]
    equation_results[8:15] = x1[8:15] / x2.iloc[0] * x3.iloc[0]
    equation_results[15] = x4.iloc[1]

    return equation_results

df_model_fleet_vehicle_market['Refuelling Cost (non-BEV)'] = Refuelling_Cost_Non_BEV(df_model_fleet_vehicle_market['Lifetime Fuel Use (L)'],
                                                                                     df_infrastructure_Gasoline_Station_Infrastructure['Gasoline Refill Amount (L)'],
                                                                                     df_infrastructure_Gasoline_Station_Infrastructure['Gasoline Refill Hassle (￥/each)'],
                                                                                     df_infrastructure_Fleet_Vehicle_Market_FCEV['Refilling Inconvenience Costs @ Public (￥/Lifetime)'])


# In[148]:


def Total_Charg_Costs(x1, x2, x3):
    equation_results = np.zeros(2)
    
    equation_results[0] = x1.iloc[0] + x2[3] + x3[3]
    equation_results[1] = x1.iloc[1] + x2[11] + x3[11]
    
    return equation_results

df_infrastructure_Fleet_Vehicle_Market_PHEV['Total charg costs (￥)'] = Total_Charg_Costs(df_infrastructure_Fleet_Vehicle_Market_PHEV['Electricity costs (￥)'],
                                                                                          df_model_fleet_vehicle_market['Energy Cost (non-BEV)'],
                                                                                          df_model_fleet_vehicle_market['Refuelling Cost (non-BEV)'])


# In[149]:


def Range_Cost_BEV_Fleet_Vehicle(x1, x2, x3, x4, x5, x6):
    equation_results = np.zeros_like(x1)
    
    equation_results[:7] = np.where(x1[:7] == 0,
                                    (x2[:7] / x3.iloc[1] * x4.iloc[1]) * (x5.iloc[1] / x6.iloc[1]),
                                    0)
    equation_results[7] = ((x6.iloc[1] - x2[7]) / x3.iloc[1] * x4.iloc[1]) * (x5.iloc[1] / x6.iloc[1])
    equation_results[8:15] = np.where(x1[8:15] == 0,
                                      (x2[8:15] / x3.iloc[1] * x4.iloc[1]) * (x5.iloc[1] / x6.iloc[1]),
                                      0)
    equation_results[15] = ((x6.iloc[1] - x2[15]) / x3.iloc[1] * x4.iloc[1]) * (x5.iloc[1] / x6.iloc[1])
    
    return equation_results

df_model_fleet_vehicle_market['Range Cost (BEV)'] = Range_Cost_BEV_Fleet_Vehicle(df_model['FC CS Gasoline Equivalent (L/100km)'],
                                                                                 df_model_fleet_vehicle_market['Daily Fuel VKT'], 
                                                                                 df_consumer['Travel distance per ride (km)'], 
                                                                                 df_consumer['Fare per ride (￥)'],
                                                                                 df_consumer['Lifetime kilometers'],
                                                                                 df_consumer['Driving - mean (km)'])


# In[150]:


def Charg_Cost_PEV(x1, x2, x3, x4):
    equation_results = np.zeros(16)
    
    equation_results[:3] = 0
    equation_results[3] = x1.iloc[0] * x2.iloc[0] + x3.iloc[0] * (1 - x2.iloc[0])
    equation_results[4:7] = x4[0:3]
    equation_results[7:11] = 0
    equation_results[11] = x1.iloc[1] * x2.iloc[0] + x3.iloc[1] * (1 - x2.iloc[0])
    equation_results[12:15] = x4[3:7]
    equation_results[15] = 0

    return equation_results

df_model_fleet_vehicle_market['Charg Cost (PEV)'] = Charg_Cost_PEV(df_infrastructure_Fleet_Vehicle_Market_PHEV['Total charg costs (￥)'],
                                                                   df_infrastructure_Charging_Infrastructure['Public charging availability'],
                                                                   df_infrastructure_Fleet_Vehicle_Market_PHEV['Total energy costs (￥)'],
                                                                   df_infrastructure_Fleet_Vehicle_Market_BEV['Lifetime costs'])


# In[151]:


def Purchase_Privilage_Fleet_Vehicle(x1, x2):
    equation_results = np.zeros(16)
    
    equation_results[0:16] = 0

    return equation_results

df_model_fleet_vehicle_market['Purchase Privilage'] = Purchase_Privilage_Fleet_Vehicle(df_projection['Purchase Privilage'],
                                                                                       df_projection['Year'])


# In[152]:


def Constant_Fleet_Vehicle(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15):
    idx = np.searchsorted(x2, Year)  
    idx = np.clip(idx, 0, len(x2) - 1)
    
    equation_results = np.zeros(16)
    
    equation_results[:2] = x1[idx]
    equation_results[2] = x3[idx]
    equation_results[3] = x4[idx]
    equation_results[4] = x5[idx]
    equation_results[5] = x6[idx]
    equation_results[6] = x7[idx]
    equation_results[7] = x8[idx]
    equation_results[8:10] = x9[idx]
    equation_results[10] = x10[idx]
    equation_results[11] = x11[idx]
    equation_results[12] = x12[idx]
    equation_results[13] = x13[idx]
    equation_results[14] = x14[idx]
    equation_results[15] = x15[idx]

    return equation_results

df_model_fleet_vehicle_market['Constant'] = Constant_Fleet_Vehicle(df_constant_fleet_vehicles['ICEV.2'],
                                                                   df_constant_fleet_vehicles['Year.1'],
                                                                   df_constant_fleet_vehicles['ICEV-Low.2'],
                                                                   df_constant_fleet_vehicles['PHEV.2'],
                                                                   df_constant_fleet_vehicles['BEV-Sht.2'],
                                                                   df_constant_fleet_vehicles['BEV-Mid.2'],
                                                                   df_constant_fleet_vehicles['BEV-Lng.2'],
                                                                   df_constant_fleet_vehicles['FCEV.2'],
                                                                   df_constant_fleet_vehicles['ICEV.3'],
                                                                   df_constant_fleet_vehicles['ICEV-Low.3'],
                                                                   df_constant_fleet_vehicles['PHEV.3'],
                                                                   df_constant_fleet_vehicles['BEV-Sht.3'],
                                                                   df_constant_fleet_vehicles['BEV-Mid.3'],
                                                                   df_constant_fleet_vehicles['BEV-Lng.3'],
                                                                   df_constant_fleet_vehicles['FCEV.3'])


# In[153]:


def Ownership_Total_Cost_Fleet_Vehicle(x1, x2, x3, x4, x5, x6, x7, x8):
    
    equation_results = np.zeros(16)
    
    equation_results[:3] = np.where(Year <= 2020,
                                x1[:3] + x2[:3] + x3[:3] + x4[:3] + x5[:3] + x6[:3] + x7[:3],
                                x1[:3] + x2[:3] + x3[:3] + x4[:3] + x5[:3] + x6[:3] + x8[:3])
    equation_results[3] = np.where(Year <= 2020,
                               x1[3] + x5[3] + x6[3] + x7[3],
                               x1[3] + x5[3] + x6[3] + x8[3])
    equation_results[4:11] = np.where(Year <= 2020,
                                  x1[4:11] + x2[4:11] + x3[4:11] + x4[4:11] + x5[4:11] + x6[4:11] + x7[4:11],
                                  x1[4:11] + x2[4:11] + x3[4:11] + x4[4:11] + x5[4:11] + x6[4:11] + x8[4:11])
    equation_results[11] = np.where(Year <= 2020,
                                x1[11] + x5[11] + x6[11] + x7[11],
                                x1[11] + x5[11] + x6[11] + x8[11])
    equation_results[12:16] = np.where(Year <= 2020,
                                   x1[12:16] + x2[12:16] + x3[12:16] + x4[12:16] + x5[12:16] + x6[12:16] + x7[12:16],
                                   x1[12:16] + x2[12:16] + x3[12:16] + x4[12:16] + x5[12:16] + x6[12:16] + x8[12:16])

    return equation_results

df_model_fleet_vehicle_market['Ownership Total Cost'] = Ownership_Total_Cost_Fleet_Vehicle(df_model_fleet_vehicle_market['Price to consumers (￥)'],
                                                                                           df_model_fleet_vehicle_market['Energy Cost (non-BEV)'],
                                                                                           df_model_fleet_vehicle_market['Refuelling Cost (non-BEV)'],
                                                                                           df_model_fleet_vehicle_market['Range Cost (BEV)'],
                                                                                           df_model_fleet_vehicle_market['Charg Cost (PEV)'],
                                                                                           df_model_fleet_vehicle_market['Purchase Privilage'],
                                                                                           df_model_fleet_vehicle_market['Calibrated Other Cost'],
                                                                                           df_model_fleet_vehicle_market['Constant'])


# In[154]:


EXP = np.exp
LN = np.log

df_DCM_ExpUSum_fleet_vehicle = pd.DataFrame({
    'Sedan-ExpUSum_IC': EXP(df_DCM_fleet_vehicle['4_Technology_IC'].iloc[0]
                            * float(df_model_fleet_vehicle_market['Ownership Total Cost'].iloc[0])) 
    + EXP(df_DCM_fleet_vehicle['4_Technology_IC'].iloc[0] * float(df_model_fleet_vehicle_market['Ownership Total Cost'].iloc[1])) 
    + EXP(df_DCM_fleet_vehicle['4_Technology_IC'].iloc[0] * float(df_model_fleet_vehicle_market['Ownership Total Cost'].iloc[2])),
    
    'Sedan-ExpUSum_PH': EXP(df_DCM_fleet_vehicle['4_Technology_PH'].iloc[0]
                            * float(df_model_fleet_vehicle_market['Ownership Total Cost'].iloc[3])),
    
    'Sedan-ExpUSum_EV': EXP(df_DCM_fleet_vehicle['4_Technology_EV'].iloc[0] 
                            * float(df_model_fleet_vehicle_market['Ownership Total Cost'].iloc[4])) 
    + EXP(df_DCM_fleet_vehicle['4_Technology_EV'].iloc[0] * float(df_model_fleet_vehicle_market['Ownership Total Cost'].iloc[5])) 
    + EXP(df_DCM_fleet_vehicle['4_Technology_EV'].iloc[0] * float(df_model_fleet_vehicle_market['Ownership Total Cost'].iloc[6])),
    
    'Sedan-ExpUSum_FC': EXP(df_DCM_fleet_vehicle['4_Technology_FC'].iloc[0] 
                            * float(df_model_fleet_vehicle_market['Ownership Total Cost'].iloc[7])),
    
    'SUV-ExpUSum_IC': EXP(df_DCM_fleet_vehicle['4_Technology_IC'].iloc[0] 
                          * float(df_model_fleet_vehicle_market['Ownership Total Cost'].iloc[8])) 
    + EXP(df_DCM_fleet_vehicle['4_Technology_IC'].iloc[0] * float(df_model_fleet_vehicle_market['Ownership Total Cost'].iloc[9])) 
    + EXP(df_DCM_fleet_vehicle['4_Technology_IC'].iloc[0] * float(df_model_fleet_vehicle_market['Ownership Total Cost'].iloc[10])),
    
    'SUV-ExpUSum_PH': EXP(df_DCM_fleet_vehicle['4_Technology_PH'].iloc[0] 
                          * float(df_model_fleet_vehicle_market['Ownership Total Cost'].iloc[11])),
    
    'SUV-ExpUSum_EV': EXP(df_DCM_fleet_vehicle['4_Technology_EV'].iloc[0] 
                          * float(df_model_fleet_vehicle_market['Ownership Total Cost'].iloc[12])) 
    + EXP(df_DCM_fleet_vehicle['4_Technology_EV'].iloc[0] * float(df_model_fleet_vehicle_market['Ownership Total Cost'].iloc[13])) 
    + EXP(df_DCM_fleet_vehicle['4_Technology_EV'].iloc[0] * float(df_model_fleet_vehicle_market['Ownership Total Cost'].iloc[14])),
    
    'SUV-ExpUSum_FC': EXP(df_DCM_fleet_vehicle['4_Technology_FC'].iloc[0] 
                          * float(df_model_fleet_vehicle_market['Ownership Total Cost'].iloc[15])),
}, index=[0])


# In[155]:


df_DCM_ExpUSum_fleet_vehicle


# In[156]:


EXP = np.exp
LN = np.log

df_DCM_ExpUSum_fleet_vehicle['ExpUSum_Sedan'] = (
    EXP(LN(df_DCM_ExpUSum_fleet_vehicle['Sedan-ExpUSum_IC'].iloc[0]) / df_DCM_fleet_vehicle['4_Technology_IC'].iloc[0] * df_DCM_fleet_vehicle['3_Powertrain_Conv'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_fleet_vehicle['Sedan-ExpUSum_PH'].iloc[0]) / df_DCM_fleet_vehicle['4_Technology_PH'].iloc[0] * df_DCM_fleet_vehicle['3_Powertrain_Conv'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_fleet_vehicle['Sedan-ExpUSum_EV'].iloc[0]) / df_DCM_fleet_vehicle['4_Technology_EV'].iloc[0] * df_DCM_fleet_vehicle['3_Powertrain_Electricity'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_fleet_vehicle['Sedan-ExpUSum_FC'].iloc[0]) / df_DCM_fleet_vehicle['4_Technology_FC'].iloc[0] * df_DCM_fleet_vehicle['3_Powertrain_Hydrogen'].iloc[0])
)

df_DCM_ExpUSum_fleet_vehicle['ExpUSum_SUV_Crossover'] = (
    EXP(LN(df_DCM_ExpUSum_fleet_vehicle['SUV-ExpUSum_IC'].iloc[0]) / df_DCM_fleet_vehicle['4_Technology_IC'].iloc[0] * df_DCM_fleet_vehicle['3_Powertrain_Conv'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_fleet_vehicle['SUV-ExpUSum_PH'].iloc[0]) / df_DCM_fleet_vehicle['4_Technology_PH'].iloc[0] * df_DCM_fleet_vehicle['3_Powertrain_Conv'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_fleet_vehicle['SUV-ExpUSum_EV'].iloc[0]) / df_DCM_fleet_vehicle['4_Technology_EV'].iloc[0] * df_DCM_fleet_vehicle['3_Powertrain_Electricity'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_fleet_vehicle['SUV-ExpUSum_FC'].iloc[0]) / df_DCM_fleet_vehicle['4_Technology_FC'].iloc[0] * df_DCM_fleet_vehicle['3_Powertrain_Hydrogen'].iloc[0])
)

df_DCM_ExpUSum_fleet_vehicle['All-ExpUSum_Fleet vehicle'] = (
    EXP(LN(df_DCM_ExpUSum_fleet_vehicle['ExpUSum_Sedan'].iloc[0]) / df_DCM_fleet_vehicle['3_Powertrain_Conv'].iloc[0] * df_DCM_fleet_vehicle['2_Class_Sedan'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_fleet_vehicle['ExpUSum_Sedan'].iloc[0]) / df_DCM_fleet_vehicle['3_Powertrain_Electricity'].iloc[0] * df_DCM_fleet_vehicle['2_Class_Sedan'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_fleet_vehicle['ExpUSum_Sedan'].iloc[0]) / df_DCM_fleet_vehicle['3_Powertrain_Hydrogen'].iloc[0] * df_DCM_fleet_vehicle['2_Class_Sedan'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_fleet_vehicle['ExpUSum_SUV_Crossover'].iloc[0]) / df_DCM_fleet_vehicle['3_Powertrain_Conv'].iloc[0] * df_DCM_fleet_vehicle['2_Class_SUV/Crossover'].iloc[0]) + 
    EXP(LN(df_DCM_ExpUSum_fleet_vehicle['ExpUSum_SUV_Crossover'].iloc[0]) / df_DCM_fleet_vehicle['3_Powertrain_Electricity'].iloc[0] * df_DCM_fleet_vehicle['2_Class_SUV/Crossover'].iloc[0]) +
    EXP(LN(df_DCM_ExpUSum_fleet_vehicle['ExpUSum_SUV_Crossover'].iloc[0]) / df_DCM_fleet_vehicle['3_Powertrain_Hydrogen'].iloc[0] * df_DCM_fleet_vehicle['2_Class_SUV/Crossover'].iloc[0])
)


# In[157]:


EXP = np.exp
LN = np.log

def Pcosteff_Fleet_Vehicle(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
    equation_results = np.zeros_like(x2)
    
    equation_results[0:3] = EXP(x1.iloc[0] * x2[0:3]) / x3.iloc[0]
    equation_results[3] = EXP(x4.iloc[0] * x2[3]) / x5.iloc[0]
    equation_results[4:7] = EXP(x6.iloc[0] * x2[4:7]) / x7.iloc[0]
    equation_results[7] = EXP(x8.iloc[0] * x2[7]) / x9.iloc[0]
    equation_results[8:11] = EXP(x1.iloc[0] * x2[8:11]) / x10.iloc[0]
    equation_results[11] = EXP(x4.iloc[0] * x2[11]) / x11.iloc[0]
    equation_results[12:15] = EXP(x6.iloc[0] * x2[12:15]) / x12.iloc[0]
    equation_results[15] = EXP(x8.iloc[0] * x2[15]) / x13.iloc[0]

    return equation_results

df_model_fleet_vehicle_market['Pcosteff'] = Pcosteff_Fleet_Vehicle(df_DCM_fleet_vehicle['4_Technology_IC'],
                                                                         df_model_fleet_vehicle_market['Ownership Total Cost'],
                                                                         df_DCM_ExpUSum_fleet_vehicle['Sedan-ExpUSum_IC'],
                                                                         df_DCM_fleet_vehicle['4_Technology_PH'],
                                                                         df_DCM_ExpUSum_fleet_vehicle['Sedan-ExpUSum_PH'],
                                                                         df_DCM_fleet_vehicle['4_Technology_EV'],
                                                                         df_DCM_ExpUSum_fleet_vehicle['Sedan-ExpUSum_EV'],
                                                                         df_DCM_fleet_vehicle['4_Technology_FC'],
                                                                         df_DCM_ExpUSum_fleet_vehicle['Sedan-ExpUSum_FC'],
                                                                         df_DCM_ExpUSum_fleet_vehicle['SUV-ExpUSum_IC'],
                                                                         df_DCM_ExpUSum_fleet_vehicle['SUV-ExpUSum_PH'],
                                                                         df_DCM_ExpUSum_fleet_vehicle['SUV-ExpUSum_EV'],
                                                                         df_DCM_ExpUSum_fleet_vehicle['SUV-ExpUSum_FC'])


# In[158]:


EXP = np.exp
LN = np.log

def Pptrain_Fleet_Vehicle(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17):
    
    equation_results = np.zeros(16)
    
    equation_results[0:3] = EXP(LN(x1.iloc[0]) / x2.iloc[0] * x3.iloc[0]) / x4.iloc[0]
    equation_results[3] = EXP(LN(x5.iloc[0]) / x6.iloc[0] * x3.iloc[0]) / x4.iloc[0]
    equation_results[4:7] = EXP(LN(x7.iloc[0])/ x8.iloc[0] * x9.iloc[0]) / x4.iloc[0]
    equation_results[7] = EXP(LN(x10.iloc[0]) / x11.iloc[0] * x12.iloc[0]) / x4.iloc[0]
    equation_results[8:11] = EXP(LN(x13.iloc[0]) / x2.iloc[0] * x3.iloc[0]) / x14.iloc[0]
    equation_results[11] = EXP(LN(x15.iloc[0]) / x6.iloc[0] * x3.iloc[0]) / x14.iloc[0]
    equation_results[12:15] = EXP(LN(x16.iloc[0]) / x8.iloc[0] * x9.iloc[0]) / x14.iloc[0]
    equation_results[15] = EXP(LN(x17.iloc[0]) / x11.iloc[0] * x12.iloc[0]) / x14.iloc[0]

    return equation_results

df_model_fleet_vehicle_market['Pptrain'] = Pptrain_Fleet_Vehicle(df_DCM_ExpUSum_fleet_vehicle['Sedan-ExpUSum_IC'],
                                                                 df_DCM_fleet_vehicle['4_Technology_IC'],
                                                                 df_DCM_fleet_vehicle['3_Powertrain_Conv'],
                                                                 df_DCM_ExpUSum_fleet_vehicle['ExpUSum_Sedan'],
                                                                 df_DCM_ExpUSum_fleet_vehicle['Sedan-ExpUSum_PH'],
                                                                 df_DCM_fleet_vehicle['4_Technology_PH'],
                                                                 df_DCM_ExpUSum_fleet_vehicle['Sedan-ExpUSum_EV'],
                                                                 df_DCM_fleet_vehicle['4_Technology_EV'],
                                                                 df_DCM_fleet_vehicle['3_Powertrain_Electricity'],
                                                                 df_DCM_ExpUSum_fleet_vehicle['Sedan-ExpUSum_FC'],
                                                                 df_DCM_fleet_vehicle['4_Technology_FC'],
                                                                 df_DCM_fleet_vehicle['3_Powertrain_Hydrogen'],
                                                                 df_DCM_ExpUSum_fleet_vehicle['SUV-ExpUSum_IC'],
                                                                 df_DCM_ExpUSum_fleet_vehicle['ExpUSum_SUV_Crossover'],
                                                                 df_DCM_ExpUSum_fleet_vehicle['SUV-ExpUSum_PH'],
                                                                 df_DCM_ExpUSum_fleet_vehicle['SUV-ExpUSum_EV'],
                                                                 df_DCM_ExpUSum_fleet_vehicle['SUV-ExpUSum_FC'])


# In[159]:


EXP = np.exp
LN = np.log

def Pclass_Fleet_Vehicle(x1, x2, x3, x4, x5, x6, x7, x8):
    
    equation_results = np.zeros(16)
    
    equation_results[0:8] = (EXP(LN(x1.iloc[0]) / x2.iloc[0] * x3.iloc[0]) + EXP(LN(x1.iloc[0]) / x4.iloc[0] * x3.iloc[0]) + EXP(LN(x1.iloc[0]) / x5.iloc[0] * x3.iloc[0])) / x6.iloc[0]
    equation_results[8:16] = (EXP(LN(x7.iloc[0]) / x2.iloc[0] * x8.iloc[0]) + EXP(LN(x7.iloc[0]) / x4.iloc[0] * x8.iloc[0]) + EXP(LN(x7.iloc[0]) / x5.iloc[0] * x8.iloc[0])) / x6.iloc[0]

    return equation_results

df_model_fleet_vehicle_market['Pclass'] = Pclass_Fleet_Vehicle(df_DCM_ExpUSum_fleet_vehicle['ExpUSum_Sedan'],
                                                               df_DCM_fleet_vehicle['3_Powertrain_Conv'],
                                                               df_DCM_fleet_vehicle['2_Class_Sedan'],
                                                               df_DCM_fleet_vehicle['3_Powertrain_Electricity'],
                                                               df_DCM_fleet_vehicle['3_Powertrain_Hydrogen'],
                                                               df_DCM_ExpUSum_fleet_vehicle['All-ExpUSum_Fleet vehicle'],
                                                               df_DCM_ExpUSum_fleet_vehicle['ExpUSum_SUV_Crossover'],
                                                               df_DCM_fleet_vehicle['2_Class_SUV/Crossover'])


# In[160]:


def Market_Share_Fleet_Vehicle(x1, x2, x3):
    
    equation_results = np.zeros(16)
    
    equation_results[0:16] = x1[0:16] * x2[0:16] * x3[0:16]

    return equation_results

df_model_fleet_vehicle_market['Market share'] = Market_Share_Fleet_Vehicle(df_model_fleet_vehicle_market['Pcosteff'],
                                                                           df_model_fleet_vehicle_market['Pptrain'],
                                                                           df_model_fleet_vehicle_market['Pclass'])


# In[161]:


def Sales_Fleet_Vehicle(x1, x2, x3):
    
    equation_results = np.zeros(16)
    
    equation_results[0:16] = x1.iloc[0] * (1 - x2.iloc[0]) * x3[0:16]

    return equation_results

df_model_fleet_vehicle_market['Sales (1000)'] = Sales_Fleet_Vehicle(df_model_output['Veh Total Sales (1,000)'],
                                                                    df_industry_vehicle_production['Personal purchase share in sales market'],
                                                                    df_model_fleet_vehicle_market['Market share'])


# In[162]:


df_model_fleet_vehicle_market.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x)


# In[163]:


def Total_Sales(x1, x2):
    
    equation_results = np.zeros(16)
    
    equation_results[0:16] = x1[0:16] + x2[0:16]

    return equation_results

df_model['Total Sales (1000)'] = Total_Sales(df_model_personal_vehicle_market['Sales (1000)'],
                                             df_model_fleet_vehicle_market['Sales (1000)'])


# In[164]:


def CAFC_Target(x1, x2, x3, x4):
    
    result = np.sum(x1.iloc[0:16] * x2.iloc[0:16]) / x3.iloc[0] * x4.iloc[0]
    
    return result

df_model_output['CAFC Target (L/100km)'] = CAFC_Target(df_model['Total Sales (1000)'],
                                                       df_model['FC Target (L/100km)'],
                                                       df_model_output['Veh Total Sales (1,000)'],
                                                       df_model_output['CAFC Target Multiplier'])


# In[165]:


def CAFC_Achieved(x1, x2, x3):
    
    result = (x1.iloc[0:16] * x2.iloc[0:16]).sum() / (x1.iloc[0:16] * x3.iloc[0:16]).sum()
    
    return result

df_model_output['CAFC Achieved (L/100km)'] = CAFC_Achieved(df_model['Total Sales (1000)'],
                                                           df_model['FC Achieved Gasoline Equivalent (L/100km)'],
                                                           df_model['CAFC Multiplier'])


# In[166]:


def Industry_CAFC_Credits(x1, x2, x3, x4, x5):

    result = (((x1.iloc[0:16] * x2.iloc[0:16]).sum() / x1.iloc[0:16].sum() * x3.iloc[0]) - ((x1.iloc[0:16] * x4.iloc[0:16]).sum() / (x1.iloc[0:16] * x5.iloc[0:16]).sum())) * x1.iloc[0:16].sum() * 1000
    
    return result

df_model_output['Industry CAFC credits'] = Industry_CAFC_Credits(df_model['Total Sales (1000)'],
                                                                 df_model['FC Target (L/100km)'],
                                                                 df_model_output['CAFC Target Multiplier'],
                                                                 df_model['FC Achieved Gasoline Equivalent (L/100km)'],
                                                                 df_model['CAFC Multiplier'])


# In[167]:


def Efficient_ICEV_Multiplier(x1, x2, x3):
    
    equation_results = np.zeros_like(x1)

    equation_results[:3] = np.where((x1[:3] < x2[:3] * x3.iloc[0]), ICEV_Low_Benefit_coefficient, 1)
    equation_results[3:8] = 0
    equation_results[8:11] = np.where((x1[8:11] < x2[8:11] * x3.iloc[0]), ICEV_Low_Benefit_coefficient, 1)
    equation_results[11:16] = 0

    return equation_results

df_rules['Efficient ICEV multiplier'] = Efficient_ICEV_Multiplier(df_model['FC Achieved Gasoline Equivalent (L/100km)'],
                                                                  df_model['FC Target (L/100km)'],
                                                                  df_model_output['CAFC Target Multiplier'])


# In[168]:


def NEV_Target_1000(x1, x2, x3):

    result = (x1.iloc[0:16] * x2.iloc[0:16]).sum() * x3.iloc[0]
    
    return result

df_model_output['NEV Target (1000)'] = NEV_Target_1000(df_model['Total Sales (1000)'],
                                                       df_rules['Efficient ICEV multiplier'],
                                                       df_model_output['NEV Target %'])


# In[169]:


def NEV_Achieved_1000(x1, x2):

    result = (x1.iloc[0:16] * x2.iloc[0:16]).sum() * (0.95 + 0.05 * (Year - 2020) / (2050 - 2020))
    
    return result

df_model_output['NEV Achieved (1000)'] = NEV_Achieved_1000(df_model['Total Sales (1000)'],
                                                           df_model['NEV credit'])


# In[170]:


def NEV_Credits(x1, x2):

    result = round((x1 - x2) * 1000, 0)
    
    return result

df_model_output['NEV Credits'] = NEV_Credits(df_model_output['NEV Achieved (1000)'],
                                             df_model_output['NEV Target (1000)'])


# In[171]:


def NEV_Credits_Surplus(x):
    return np.where(x > 0, x, 0)

df_model_output['NEV Credits surplus'] = NEV_Credits_Surplus(df_model_output['NEV Credits'])


# In[172]:


def CAFC_Credits_NEV(x1, x2, x3):
    return x1 + x2 * x3

df_model_output['CAFC credits w/ NEV'] = CAFC_Credits_NEV(df_model_output['Industry CAFC credits'],
                                                          df_model_output['NEV Credits surplus'],
                                                          df_model_output['CAFC/NEV credit conv'])


# In[173]:


def Net_Internal_Subsidy_from_OEM(x1, x2):
    return -(x1 * x2).iloc[0:16].sum() * 1000

df_model_output['Net Internal Subsidy from OEM (￥)'] = Net_Internal_Subsidy_from_OEM(df_model['Total Sales (1000)'],
                                                                                      df_model['Internal Subsidy to Consumer(￥)'])


# In[174]:


def Profit(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):

    result = (np.sum(x1[:16] * x2[:16]) +
              np.sum(x3[:16] * x4[:16]) +
              np.sum(x5[:16] * x6[:16]) +
              np.sum(x5[:16] * x7[:16]) +
              np.sum(x5[:16] * x8[:16]) -
              np.sum(x5[:16] * x9[:16]) -
              np.sum(x5[:16] * x10[:16])) * 1000
    
    return result

df_model_output['Profit (￥)'] = Profit(df_model_personal_vehicle_market['Sales (1000)'],
                                        df_model_personal_vehicle_market['Price to consumers (￥)'],
                                        df_model_fleet_vehicle_market['Sales (1000)'],
                                        df_model_fleet_vehicle_market['Price to consumers (￥)'],
                                        df_model['Total Sales (1000)'],
                                        df_model['Additional Subsidy to Consumers (￥)'],
                                        df_model['Central Gov Sub to Consumers (￥)'],
                                        df_model['Avg Local Gov Sub to Consumers (￥)'],
                                        df_model['Internal Subsidy to Consumer(￥)'],
                                        df_model['Production Cost (￥)'])


# In[175]:


def Profit_veh(x1, x2):
    return x1 / x2 / 1000

df_model_output['Profit (￥/veh)'] = Profit_veh(df_model_output['Profit (￥)'],
                                                df_model_output['Veh Total Sales (1,000)'])


# In[176]:


df_model_output.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x)


# In[177]:


df_model.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x)


# In[178]:


df_model_personal_vehicle_market.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x)


# In[179]:


df_model_fleet_vehicle_market.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x)


# No Rules
Genetic Algorithm Optimization
# In[180]:


# parameter
POPULATION_SIZE = 50 # Define the number of individuals in the population.
GENERATIONS = 1000 # Specify the number of generations over which the algorithm will run.
MUTATION_RATE = 0.2 # Set the mutation rate as a probability that an individual will mutate.

# Initialize the population by creating a list of individuals, where each individual is represented as an array of 16 random integers within the specified "Internal Subsidy" range.
def initialize_population(size, subsidy_range):
    return [np.random.randint(*subsidy_range, 16) for _ in range(size)]

# Simulate profit calculation.
def Profit(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    result = ((np.sum(x1[:16] * x2[:16]) +
              np.sum(x3[:16] * x4[:16]) +
              np.sum(x5[:16] * x6[:16]) +
              np.sum(x5[:16] * x7[:16]) +
              np.sum(x5[:16] * x8[:16]) -
              np.sum(x5[:16] * x9[:16]) -
              np.sum(x5[:16] * x10[:16])) * 1000) / x11 / 1000
    
    return result
    
def calculate_model_output(df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market):
    model_output = pd.concat([df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market], axis=1)
    # Call the new Profit function to calculate profit
    model_output['Profit (￥/veh)'] = Profit(
        df_model_personal_vehicle_market['Sales (1000)'],
        df_model_personal_vehicle_market['Price to consumers (￥)'],
        df_model_fleet_vehicle_market['Sales (1000)'],
        df_model_fleet_vehicle_market['Price to consumers (￥)'],
        df_model['Total Sales (1000)'],
        df_model['Additional Subsidy to Consumers (￥)'],
        df_model['Central Gov Sub to Consumers (￥)'],
        df_model['Avg Local Gov Sub to Consumers (￥)'],
        df_model['Internal Subsidy to Consumer(￥)'],
        df_model['Production Cost (￥)'],
        df_model_output['Veh Total Sales (1,000)'])
    return model_output

# Apply the subsidy values to the 'Internal Subsidy to Consumer(￥)' column of the model, calculates the model output, and returns the total profit.
def calculate_profit(subsidy_values):
    df_model_copy = df_model.copy()
    df_model_copy.loc[:15, 'Internal Subsidy to Consumer(￥)'] = subsidy_values
    model_output = calculate_model_output(df_model_copy, df_model_personal_vehicle_market, df_model_fleet_vehicle_market)
    total_profit = model_output['Profit (￥/veh)'].sum()
    return total_profit, model_output

# Sort the population based on the profit in descending order and selects the top half of the population.
def select_population(population, profits):
    sorted_indices = [i for i, _ in sorted(enumerate(profits), key=lambda x: x[1], reverse=True)]
    return [population[i] for i in sorted_indices[:POPULATION_SIZE // 2]]

# Perform a single-point crossover between two parents to generate two new children.
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# Mutate an individual by randomly changing one of its subsidy values if the mutation condition (random chance less than the mutation rate) is met.
def mutate(individual, subsidy_range):
    if np.random.rand() < MUTATION_RATE:
        mutation_point = np.random.randint(len(individual))
        individual[mutation_point] = np.random.randint(*subsidy_range)
    return individual

# Initialize population
subsidy_range = (0, 10000)  # Assume "Internal Subsidy" range is between 0 to 10000
population = initialize_population(POPULATION_SIZE, subsidy_range)

# Initialize variables to track the best individual and its profit across all generations.
best_profit = -float('inf')
best_individual = None
best_output = None

# Run the genetic algorithm over a specified number of generations.
for generation in range(GENERATIONS):
    profits = []
    outputs = []
    
    # Calculate the profit for each individual in the current population.
    for individual in population:
        profit, output = calculate_profit(individual)
        profits.append(profit)
        outputs.append(output)
    
    # Optimal individual
    max_profit = max(profits)
    if max_profit > best_profit:
        best_profit = max_profit
        best_individual = population[profits.index(max_profit)]
        best_output = outputs[profits.index(max_profit)]
    
    # Select the top-performing individuals for reproduction.
    selected_population = select_population(population, profits)
    
    # Generate a new population through crossover and mutation from selected individuals.
    new_population = []
    while len(new_population) < POPULATION_SIZE:
        idx1, idx2 = np.random.randint(0, len(selected_population), 2)
        parent1, parent2 = selected_population[idx1], selected_population[idx2]
        child1, child2 = crossover(parent1, parent2)
        new_population.append(mutate(child1, subsidy_range))
        new_population.append(mutate(child2, subsidy_range))

    # Update the population for the next generation.
    population = new_population

print("Optimal Internal Subsidy combination:", best_individual)
print("Maximum profit:", best_profit)
print("Optimal output model_output:")
print(best_output.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))


# In[181]:


# Adjust the subsidy combination so that one of the values is 0
min_subsidy_value = min(best_individual)
adjusted_individual = best_individual - min_subsidy_value

# Calculate the adjusted profit and model output
adjusted_profit, adjusted_output = calculate_profit(adjusted_individual)

# Print the results
print("Optimal Internal Subsidy combination (adjusted):", adjusted_individual)
print("Maximum profit (adjusted):", adjusted_profit)
print("Optimal output model_output (adjusted):")
print(adjusted_output.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))

Evolopy
# In[182]:


get_ipython().system('git clone https://github.com/7ossam81/EvoloPy.git')


# In[183]:


# Change working directory
os.chdir("C:/Users/11379/EvoloPy/")


# In[184]:


from optimizer import run


# In[185]:


optimizer = ["GA", "PSO", "GWO"]
objectivefunc = ["F3", "F4"]
NumOfRuns = 30
params = {"PopulationSize": 30, "Iterations": 100}
export_flags = {
    "Export_avg": True,
    "Export_details": True,
    "Export_convergence": True,
    "Export_boxplot": True,
}


# In[186]:


# Define the objective function
def calculate_profit(subsidy_values):
    df_model_copy = df_model.copy()
    df_model_copy.loc[:15, 'Internal Subsidy to Consumer(￥)'] = subsidy_values
    model_output = calculate_model_output(df_model_copy, df_model_personal_vehicle_market, df_model_fleet_vehicle_market)
    total_profit = model_output['Profit (￥/veh)'].sum()
    return -total_profit  # Since Evolopy minimizes the objective function, we return negative profit

def calculate_model_output(df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market):
    model_output = pd.concat([df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market], axis=1)
    # Call the new Profit function to calculate profit
    model_output['Profit (￥/veh)'] = Profit(
        df_model_personal_vehicle_market['Sales (1000)'],
        df_model_personal_vehicle_market['Price to consumers (￥)'],
        df_model_fleet_vehicle_market['Sales (1000)'],
        df_model_fleet_vehicle_market['Price to consumers (￥)'],
        df_model['Total Sales (1000)'],
        df_model['Additional Subsidy to Consumers (￥)'],
        df_model['Central Gov Sub to Consumers (￥)'],
        df_model['Avg Local Gov Sub to Consumers (￥)'],
        df_model['Internal Subsidy to Consumer(￥)'],
        df_model['Production Cost (￥)'],
        df_model_output['Veh Total Sales (1,000)'])
    return model_output

def Profit(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    result = ((np.sum(x1[:16] * x2[:16]) +
              np.sum(x3[:16] * x4[:16]) +
              np.sum(x5[:16] * x6[:16]) +
              np.sum(x5[:16] * x7[:16]) +
              np.sum(x5[:16] * x8[:16]) -
              np.sum(x5[:16] * x9[:16]) -
              np.sum(x5[:16] * x10[:16])) * 1000) / x11 / 1000
    return result

run(optimizer, objectivefunc, NumOfRuns, params, export_flags)


# In[187]:


from IPython.display import Image
import datetime
import ipywidgets as widgets


# In[188]:


# Function to handle folder selection and display the CSV content
def display_csv_content(foldername):
    # Construct the full path to the CSV file
    filename = os.path.join(foldername, 'experiment.csv')
    
    # Read and display the first 6 rows of the CSV file
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        display(df.head(6).applymap(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))
    else:
        print("File not found:", filename)

foldernames = [filename for filename in os.listdir() if filename.startswith(str(datetime.datetime.now().year)) and os.path.isdir(filename)]
default_folder = foldername if len(foldernames) == 0 else foldernames[0]
drop_folder = widgets.Dropdown(options=foldernames, description='Select folder:', value=default_folder)

#Get the selected folder
foldername = drop_folder.value

# Show the widget and bind the display function to changes in the dropdown
widgets.interact(display_csv_content, foldername=drop_folder)


# In[189]:


def display_csv_and_convergence_curve(foldername):
    # Display the CSV content
    csv_filename = os.path.join(foldername, 'experiment.csv')
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        display(df.head(6).applymap(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))
    else:
        print("CSV file not found:", csv_filename)
    
    # Function to handle convergence plot selection and display
    def show_convergence_curve(plot_filename):
        image_path = os.path.join(foldername, plot_filename)
        if os.path.exists(image_path):
            display(Image(filename=image_path))
        else:
            print("Image file not found:", image_path)
    
    # Select convergence curve to show
    filenames = [filename for filename in os.listdir(foldername) if filename.startswith('convergence')]
    drop_plot_convergence = widgets.Dropdown(options=filenames, description='Select plot:')
    
    # Interact for plot selection and display
    widgets.interact(show_convergence_curve, plot_filename=drop_plot_convergence)

# Create a dropdown for folder selection
foldernames = [filename for filename in os.listdir() if filename.startswith(str(datetime.datetime.now().year)) and os.path.isdir(filename)]
drop_folder = widgets.Dropdown(options=foldernames, description='Select folder:')

# Show the widget and bind the display function to changes in the dropdown
widgets.interact(display_csv_and_convergence_curve, foldername=drop_folder)


# In[190]:


# Function to display the CSV content and boxplots
def display_boxplot_image(foldername):
    # Display the CSV content
    csv_filename = os.path.join(foldername, 'experiment.csv')
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        display(df.head(6).applymap(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))
    else:
        print("CSV file not found:", csv_filename)
    
    # Function to handle boxplot selection and display
    def show_boxplot_image(plot_filename):
        image_path = os.path.join(foldername, plot_filename)
        if os.path.exists(image_path):
            display(Image(filename=image_path))
        else:
            print("Image file not found:", image_path)
    
    # Select boxplot to show
    filenames = [filename for filename in os.listdir(foldername) if filename.startswith('boxplot')]
    if filenames:
        drop_plot_box = widgets.Dropdown(options=filenames, description='Select plot:')
        widgets.interact(show_boxplot_image, plot_filename=drop_plot_box)
    else:
        print("No boxplot images found in the selected folder.")

# Create a dropdown for folder selection
foldernames = [filename for filename in os.listdir() if filename.startswith(str(datetime.datetime.now().year)) and os.path.isdir(filename)]
if foldernames:
    drop_folder = widgets.Dropdown(options=foldernames, description='Select folder:')
    widgets.interact(display_boxplot_image, foldername=drop_folder)
else:
    print("No folders from the current year found.")


# In[191]:


def newrun(optimizer, objectivefunc, NumOfRuns, params, export_flags, df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market):
    results = {}
    global_best_solution = None
    global_best_value = np.inf

    for opt in optimizer:
        best_solution = None
        best_value = np.inf
        for _ in range(NumOfRuns):
            solution = np.random.uniform(0, 10000, 16)  # Generate 16 subsidy values
            value = calculate_profit(solution)
            if value < best_value:
                best_value = value
                best_solution = solution
        best_solution -= np.min(best_solution)  # Normalize the best solution
        results[opt] = (best_solution, -best_value)
        print(f"Best solution for {opt}: {best_solution}, Profit: {-best_value}")

        if best_value < global_best_value:
            global_best_value = best_value
            global_best_solution = best_solution

    global_best_solution -= np.min(global_best_solution)  # Ensure at least one subsidy value is zero
    df_model.loc[:15, 'Internal Subsidy to Consumer(￥)'] = global_best_solution
    optimal_model_output = calculate_model_output(df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market)

    # Optionally print optimal output
    print("Optimal output model_output:")
    print(optimal_model_output.applymap(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))

    # Return the results and the optimal model output
    return results, optimal_model_output

results, optimal_output = newrun(optimizer, objectivefunc, NumOfRuns, params, export_flags, df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market)


# Dual-credit Policy
Genetic Algorithm Optimization
# In[192]:


# Add constraints to the profit calculation
def calculate_profit_with_constraints(subsidy_values):
    df_model_copy = df_model.copy()
    df_model_copy.loc[:15, 'Internal Subsidy to Consumer(￥)'] = subsidy_values
    model_output = calculate_model_output(df_model_copy, df_model_personal_vehicle_market, df_model_fleet_vehicle_market)
    
    # Apply constraints
    if (df_model_output['NEV Credits'] < 0).any() and (df_model_output['CAFC credits w/ NEV'] < 0).any():
        total_profit = -float('inf')  # Apply a large negative penalty if constraints are not met
    else:
        total_profit = model_output['Profit (￥/veh)'].sum()
    
    return total_profit, model_output

# Run the genetic algorithm
for generation in range(GENERATIONS):
    profits = []
    outputs = []
    
    for individual in population:
        profit, output = calculate_profit_with_constraints(individual)
        profits.append(profit)
        outputs.append(output)
    
    max_profit = max(profits)
    if max_profit > best_profit:
        best_profit = max_profit
        best_individual = population[profits.index(max_profit)]
        best_output = outputs[profits.index(max_profit)]
    
    selected_population = select_population(population, profits)
    
    new_population = []
    while len(new_population) < POPULATION_SIZE:
        idx1, idx2 = np.random.randint(0, len(selected_population), 2)
        parent1, parent2 = selected_population[idx1], selected_population[idx2]
        child1, child2 = crossover(parent1, parent2)
        new_population.append(mutate(child1, subsidy_range))
        new_population.append(mutate(child2, subsidy_range))

    population = new_population

print("Optimal Internal Subsidy combination:", best_individual)
print("Maximum profit:", best_profit)
print("Optimal output model_output:")
print(best_output.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))


# In[193]:


# Adjust the subsidy combination so that one of the values is 0
min_subsidy_value = min(best_individual)
adjusted_individual = best_individual - min_subsidy_value

# Ensure calculate_profit returns both profit and model output as a tuple
def calculate_profit(subsidy_values):
    df_model_copy = df_model.copy()
    df_model_copy.loc[:15, 'Internal Subsidy to Consumer(￥)'] = subsidy_values
    model_output = calculate_model_output(df_model_copy, df_model_personal_vehicle_market, df_model_fleet_vehicle_market)
    total_profit = model_output['Profit (￥/veh)'].sum()
    return total_profit, model_output  # Return as a tuple

# Calculate the adjusted profit and model output
adjusted_profit, adjusted_output = calculate_profit(adjusted_individual)

# Print the results
print("Optimal Internal Subsidy combination (adjusted):", adjusted_individual)
print("Maximum profit (adjusted):", adjusted_profit)
print("Optimal output model_output (adjusted):")
print(adjusted_output.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))

Evolopy
# In[194]:


# Define the objective function
def calculate_profit(subsidy_values):
    df_model_copy = df_model.copy()
    df_model_copy.loc[:15, 'Internal Subsidy to Consumer(￥)'] = subsidy_values
    model_output = calculate_model_output(df_model_copy, df_model_personal_vehicle_market, df_model_fleet_vehicle_market)

    # Check constraints
    if (df_model_output['NEV Credits'] >= 0).all() and (df_model_output['CAFC credits w/ NEV'] >= 0).all():
        total_profit = model_output['Profit (￥/veh)'].sum()
        return -total_profit  # Since Evolopy minimizes the objective function, we return negative profit
    else:
        return -1e10  # Return a very large negative value if constraints are not met

def calculate_model_output(df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market):
    model_output = pd.concat([df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market], axis=1)
    # Call the new Profit function to calculate profit
    model_output['Profit (￥/veh)'] = Profit(
        df_model_personal_vehicle_market['Sales (1000)'],
        df_model_personal_vehicle_market['Price to consumers (￥)'],
        df_model_fleet_vehicle_market['Sales (1000)'],
        df_model_fleet_vehicle_market['Price to consumers (￥)'],
        df_model['Total Sales (1000)'],
        df_model['Additional Subsidy to Consumers (￥)'],
        df_model['Central Gov Sub to Consumers (￥)'],
        df_model['Avg Local Gov Sub to Consumers (￥)'],
        df_model['Internal Subsidy to Consumer(￥)'],
        df_model['Production Cost (￥)'],
        df_model_output['Veh Total Sales (1,000)'])
    return model_output

def Profit(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    result = ((np.sum(x1[:16] * x2[:16]) +
              np.sum(x3[:16] * x4[:16]) +
              np.sum(x5[:16] * x6[:16]) +
              np.sum(x5[:16] * x7[:16]) +
              np.sum(x5[:16] * x8[:16]) -
              np.sum(x5[:16] * x9[:16]) -
              np.sum(x5[:16] * x10[:16])) * 1000) / x11 / 1000
    return result

run(optimizer, objectivefunc, NumOfRuns, params, export_flags)


# In[195]:


# Function to handle folder selection and display the CSV content
def display_csv_content(foldername):
    # Construct the full path to the CSV file
    filename = os.path.join(foldername, 'experiment.csv')
    
    # Read and display the first 6 rows of the CSV file
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        display(df.head(6).applymap(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))
    else:
        print("File not found:", filename)

# Create a dropdown widget for folder selection
foldernames = [filename for filename in os.listdir() if filename.startswith(str(datetime.datetime.now().year)) and os.path.isdir(filename)]
drop_folder = widgets.Dropdown(options=foldernames, description='Select folder:')
drop_folder

# Show the widget and bind the display function to changes in the dropdown
widgets.interact(display_csv_content, foldername=drop_folder)


# In[196]:


def display_csv_and_convergence_curve(foldername):
    # Display the CSV content
    csv_filename = os.path.join(foldername, 'experiment.csv')
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        display(df.head(6).applymap(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))
    else:
        print("CSV file not found:", csv_filename)
    
    # Function to handle convergence plot selection and display
    def show_convergence_curve(plot_filename):
        image_path = os.path.join(foldername, plot_filename)
        if os.path.exists(image_path):
            display(Image(filename=image_path))
        else:
            print("Image file not found:", image_path)
    
    # Select convergence curve to show
    filenames = [filename for filename in os.listdir(foldername) if filename.startswith('convergence')]
    drop_plot_convergence = widgets.Dropdown(options=filenames, description='Select plot:')
    
    # Interact for plot selection and display
    widgets.interact(show_convergence_curve, plot_filename=drop_plot_convergence)

# Create a dropdown for folder selection
foldernames = [filename for filename in os.listdir() if filename.startswith(str(datetime.datetime.now().year)) and os.path.isdir(filename)]
drop_folder = widgets.Dropdown(options=foldernames, description='Select folder:')

# Show the widget and bind the display function to changes in the dropdown
widgets.interact(display_csv_and_convergence_curve, foldername=drop_folder)


# In[197]:


# Function to display the CSV content and boxplots
def display_boxplot_image(foldername):
    # Display the CSV content
    csv_filename = os.path.join(foldername, 'experiment.csv')
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        display(df.head(6).applymap(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))
    else:
        print("CSV file not found:", csv_filename)
    
    # Function to handle boxplot selection and display
    def show_boxplot_image(plot_filename):
        image_path = os.path.join(foldername, plot_filename)
        if os.path.exists(image_path):
            display(Image(filename=image_path))
        else:
            print("Image file not found:", image_path)
    
    # Select boxplot to show
    filenames = [filename for filename in os.listdir(foldername) if filename.startswith('boxplot')]
    if filenames:
        drop_plot_box = widgets.Dropdown(options=filenames, description='Select plot:')
        widgets.interact(show_boxplot_image, plot_filename=drop_plot_box)
    else:
        print("No boxplot images found in the selected folder.")

# Create a dropdown for folder selection
foldernames = [filename for filename in os.listdir() if filename.startswith(str(datetime.datetime.now().year)) and os.path.isdir(filename)]
if foldernames:
    drop_folder = widgets.Dropdown(options=foldernames, description='Select folder:')
    widgets.interact(display_boxplot_image, foldername=drop_folder)
else:
    print("No folders from the current year found.")


# In[198]:


def newrun(optimizer, objectivefunc, NumOfRuns, params, export_flags, df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market):
    results = {}
    global_best_solution = None
    global_best_value = np.inf

    for opt in optimizer:
        best_solution = None
        best_value = np.inf
        for _ in range(NumOfRuns):
            solution = np.random.uniform(0, 10000, 16)  # Generate 16 subsidy values
            value = calculate_profit(solution)
            if value < best_value:
                best_value = value
                best_solution = solution
        # Normalize the best solution by subtracting the smallest subsidy value
        best_solution -= np.min(best_solution)
        results[opt] = (best_solution, -best_value)
        print(f"Best solution for {opt}: {best_solution}, Profit: {-best_value}")

        # Track the overall best across optimizers
        if best_value < global_best_value:
            global_best_value = best_value
            global_best_solution = best_solution

    # Update the model with the best overall solution normalized
    global_best_solution -= np.min(global_best_solution)  # Ensure at least one subsidy value is zero
    df_model.loc[:15, 'Internal Subsidy to Consumer(￥)'] = global_best_solution

    # Recalculate the optimal model output using the best subsidy values
    optimal_model_output = calculate_model_output(df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market)

    # Print the formatted optimal model output
    print("Optimal output model_output:")
    print(optimal_model_output.applymap(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))

# Set up and run the optimization including direct handling
newrun(optimizer, objectivefunc, NumOfRuns, params, export_flags, df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market)


# CAFC Policy Only
Genetic Algorithm Optimization
# In[199]:


# Add constraints to the profit calculation
def calculate_profit_with_constraints(subsidy_values):
    df_model_copy = df_model.copy()
    df_model_copy.loc[:15, 'Internal Subsidy to Consumer(￥)'] = subsidy_values
    model_output = calculate_model_output(df_model_copy, df_model_personal_vehicle_market, df_model_fleet_vehicle_market)
    
    # Apply constraints
    if (df_model_output['Industry CAFC credits'] < 0).any():
        total_profit = -float('inf')  # Apply a large negative penalty if constraints are not met
    else:
        total_profit = model_output['Profit (￥/veh)'].sum()
    
    return total_profit, model_output

# Run the genetic algorithm
for generation in range(GENERATIONS):
    profits = []
    outputs = []
    
    for individual in population:
        profit, output = calculate_profit_with_constraints(individual)
        profits.append(profit)
        outputs.append(output)
    
    max_profit = max(profits)
    if max_profit > best_profit:
        best_profit = max_profit
        best_individual = population[profits.index(max_profit)]
        best_output = outputs[profits.index(max_profit)]
    
    selected_population = select_population(population, profits)
    
    new_population = []
    while len(new_population) < POPULATION_SIZE:
        idx1, idx2 = np.random.randint(0, len(selected_population), 2)
        parent1, parent2 = selected_population[idx1], selected_population[idx2]
        child1, child2 = crossover(parent1, parent2)
        new_population.append(mutate(child1, subsidy_range))
        new_population.append(mutate(child2, subsidy_range))

    population = new_population

print("Optimal Internal Subsidy combination:", best_individual)
print("Maximum profit:", best_profit)
print("Optimal output model_output:")
print(best_output.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))


# In[200]:


# Adjust the subsidy combination so that one of the values is 0
min_subsidy_value = min(best_individual)
adjusted_individual = best_individual - min_subsidy_value

# Ensure calculate_profit returns both profit and model output as a tuple
def calculate_profit(subsidy_values):
    df_model_copy = df_model.copy()
    df_model_copy.loc[:15, 'Internal Subsidy to Consumer(￥)'] = subsidy_values
    model_output = calculate_model_output(df_model_copy, df_model_personal_vehicle_market, df_model_fleet_vehicle_market)
    total_profit = model_output['Profit (￥/veh)'].sum()
    return total_profit, model_output  # Return as a tuple

# Calculate the adjusted profit and model output
adjusted_profit, adjusted_output = calculate_profit(adjusted_individual)

# Print the results
print("Optimal Internal Subsidy combination (adjusted):", adjusted_individual)
print("Maximum profit (adjusted):", adjusted_profit)
print("Optimal output model_output (adjusted):")
print(adjusted_output.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))

Evolopy
# In[201]:


# Define the objective function
def calculate_profit(subsidy_values):
    df_model_copy = df_model.copy()
    df_model_copy.loc[:15, 'Internal Subsidy to Consumer(￥)'] = subsidy_values
    model_output = calculate_model_output(df_model_copy, df_model_personal_vehicle_market, df_model_fleet_vehicle_market)
    
    # Check the constraint
    if (df_model_output['Industry CAFC credits'] < 0).any():
        return np.inf  # Return a very high value if the constraint is violated
    
    total_profit = model_output['Profit (￥/veh)'].sum()
    return -total_profit  # Since Evolopy minimizes the objective function, we return negative profit

def calculate_model_output(df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market):
    model_output = pd.concat([df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market], axis=1)
    # Call the new Profit function to calculate profit
    model_output['Profit (￥/veh)'] = Profit(
        df_model_personal_vehicle_market['Sales (1000)'],
        df_model_personal_vehicle_market['Price to consumers (￥)'],
        df_model_fleet_vehicle_market['Sales (1000)'],
        df_model_fleet_vehicle_market['Price to consumers (￥)'],
        df_model['Total Sales (1000)'],
        df_model['Additional Subsidy to Consumers (￥)'],
        df_model['Central Gov Sub to Consumers (￥)'],
        df_model['Avg Local Gov Sub to Consumers (￥)'],
        df_model['Internal Subsidy to Consumer(￥)'],
        df_model['Production Cost (￥)'],
        df_model_output['Veh Total Sales (1,000)'])
    return model_output

def Profit(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    result = ((np.sum(x1[:16] * x2[:16]) +
              np.sum(x3[:16] * x4[:16]) +
              np.sum(x5[:16] * x6[:16]) +
              np.sum(x5[:16] * x7[:16]) +
              np.sum(x5[:16] * x8[:16]) -
              np.sum(x5[:16] * x9[:16]) -
              np.sum(x5[:16] * x10[:16])) * 1000) / x11 / 1000
    return result

run(optimizer, objectivefunc, NumOfRuns, params, export_flags)


# In[202]:


# Function to handle folder selection and display the CSV content
def display_csv_content(foldername):
    # Construct the full path to the CSV file
    filename = os.path.join(foldername, 'experiment.csv')
    
    # Read and display the first 6 rows of the CSV file
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        display(df.head(6).applymap(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))
    else:
        print("File not found:", filename)

# Create a dropdown widget for folder selection
foldernames = [filename for filename in os.listdir() if filename.startswith(str(datetime.datetime.now().year)) and os.path.isdir(filename)]
drop_folder = widgets.Dropdown(options=foldernames, description='Select folder:')
drop_folder

# Show the widget and bind the display function to changes in the dropdown
widgets.interact(display_csv_content, foldername=drop_folder)


# In[203]:


def display_csv_and_convergence_curve(foldername):
    # Display the CSV content
    csv_filename = os.path.join(foldername, 'experiment.csv')
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        display(df.head(6).applymap(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))
    else:
        print("CSV file not found:", csv_filename)
    
    # Function to handle convergence plot selection and display
    def show_convergence_curve(plot_filename):
        image_path = os.path.join(foldername, plot_filename)
        if os.path.exists(image_path):
            display(Image(filename=image_path))
        else:
            print("Image file not found:", image_path)
    
    # Select convergence curve to show
    filenames = [filename for filename in os.listdir(foldername) if filename.startswith('convergence')]
    drop_plot_convergence = widgets.Dropdown(options=filenames, description='Select plot:')
    
    # Interact for plot selection and display
    widgets.interact(show_convergence_curve, plot_filename=drop_plot_convergence)

# Create a dropdown for folder selection
foldernames = [filename for filename in os.listdir() if filename.startswith(str(datetime.datetime.now().year)) and os.path.isdir(filename)]
drop_folder = widgets.Dropdown(options=foldernames, description='Select folder:')

# Show the widget and bind the display function to changes in the dropdown
widgets.interact(display_csv_and_convergence_curve, foldername=drop_folder)


# In[204]:


# Function to display the CSV content and boxplots
def display_boxplot_image(foldername):
    # Display the CSV content
    csv_filename = os.path.join(foldername, 'experiment.csv')
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        display(df.head(6).applymap(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))
    else:
        print("CSV file not found:", csv_filename)
    
    # Function to handle boxplot selection and display
    def show_boxplot_image(plot_filename):
        image_path = os.path.join(foldername, plot_filename)
        if os.path.exists(image_path):
            display(Image(filename=image_path))
        else:
            print("Image file not found:", image_path)
    
    # Select boxplot to show
    filenames = [filename for filename in os.listdir(foldername) if filename.startswith('boxplot')]
    if filenames:
        drop_plot_box = widgets.Dropdown(options=filenames, description='Select plot:')
        widgets.interact(show_boxplot_image, plot_filename=drop_plot_box)
    else:
        print("No boxplot images found in the selected folder.")

# Create a dropdown for folder selection
foldernames = [filename for filename in os.listdir() if filename.startswith(str(datetime.datetime.now().year)) and os.path.isdir(filename)]
if foldernames:
    drop_folder = widgets.Dropdown(options=foldernames, description='Select folder:')
    widgets.interact(display_boxplot_image, foldername=drop_folder)
else:
    print("No folders from the current year found.")


# In[205]:


def newrun(optimizer, objectivefunc, NumOfRuns, params, export_flags, df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market):
    results = {}
    global_best_solution = None
    global_best_value = np.inf

    for opt in optimizer:
        best_solution = None
        best_value = np.inf
        for _ in range(NumOfRuns):
            solution = np.random.uniform(0, 10000, 16)  # Generate 16 subsidy values
            value = calculate_profit(solution)
            if value < best_value:
                best_value = value
                best_solution = solution
        # Normalize the best solution by subtracting the smallest subsidy value
        best_solution -= np.min(best_solution)
        results[opt] = (best_solution, -best_value)
        print(f"Best solution for {opt}: {best_solution}, Profit: {-best_value}")

        # Track the overall best across optimizers
        if best_value < global_best_value:
            global_best_value = best_value
            global_best_solution = best_solution

    # Update the model with the best overall solution normalized
    global_best_solution -= np.min(global_best_solution)  # Ensure at least one subsidy value is zero
    df_model.loc[:15, 'Internal Subsidy to Consumer(￥)'] = global_best_solution

    # Recalculate the optimal model output using the best subsidy values
    optimal_model_output = calculate_model_output(df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market)

    # Print the formatted optimal model output
    print("Optimal output model_output:")
    print(optimal_model_output.applymap(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))

# Set up and run the optimization including direct handling
newrun(optimizer, objectivefunc, NumOfRuns, params, export_flags, df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market)


# NEV Policy Only
Genetic Algorithm Optimization
# In[206]:


# Add constraints to the profit calculation
def calculate_profit_with_constraints(subsidy_values):
    df_model_copy = df_model.copy()
    df_model_copy.loc[:15, 'Internal Subsidy to Consumer(￥)'] = subsidy_values
    model_output = calculate_model_output(df_model_copy, df_model_personal_vehicle_market, df_model_fleet_vehicle_market)
    
    # Apply constraints
    if (df_model_output['NEV Credits'] < 0).any():
        total_profit = -float('inf')  # Apply a large negative penalty if constraints are not met
    else:
        total_profit = model_output['Profit (￥/veh)'].sum()
    
    return total_profit, model_output

# Run the genetic algorithm
for generation in range(GENERATIONS):
    profits = []
    outputs = []
    
    for individual in population:
        profit, output = calculate_profit_with_constraints(individual)
        profits.append(profit)
        outputs.append(output)
    
    max_profit = max(profits)
    if max_profit > best_profit:
        best_profit = max_profit
        best_individual = population[profits.index(max_profit)]
        best_output = outputs[profits.index(max_profit)]
    
    selected_population = select_population(population, profits)
    
    new_population = []
    while len(new_population) < POPULATION_SIZE:
        idx1, idx2 = np.random.randint(0, len(selected_population), 2)
        parent1, parent2 = selected_population[idx1], selected_population[idx2]
        child1, child2 = crossover(parent1, parent2)
        new_population.append(mutate(child1, subsidy_range))
        new_population.append(mutate(child2, subsidy_range))

    population = new_population

print("Optimal Internal Subsidy combination:", best_individual)
print("Maximum profit:", best_profit)
print("Optimal output model_output:")
print(best_output.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))


# In[207]:


# Adjust the subsidy combination so that one of the values is 0
min_subsidy_value = min(best_individual)
adjusted_individual = best_individual - min_subsidy_value

# Ensure calculate_profit returns both profit and model output as a tuple
def calculate_profit(subsidy_values):
    df_model_copy = df_model.copy()
    df_model_copy.loc[:15, 'Internal Subsidy to Consumer(￥)'] = subsidy_values
    model_output = calculate_model_output(df_model_copy, df_model_personal_vehicle_market, df_model_fleet_vehicle_market)
    total_profit = model_output['Profit (￥/veh)'].sum()
    return total_profit, model_output  # Return as a tuple

# Calculate the adjusted profit and model output
adjusted_profit, adjusted_output = calculate_profit(adjusted_individual)

# Print the results
print("Optimal Internal Subsidy combination (adjusted):", adjusted_individual)
print("Maximum profit (adjusted):", adjusted_profit)
print("Optimal output model_output (adjusted):")
print(adjusted_output.map(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))

Evolopy
# In[208]:


# Define the objective function
def calculate_profit(subsidy_values):
    df_model_copy = df_model.copy()
    df_model_copy.loc[:15, 'Internal Subsidy to Consumer(￥)'] = subsidy_values
    model_output = calculate_model_output(df_model_copy, df_model_personal_vehicle_market, df_model_fleet_vehicle_market)
    
    # Ensure NEV Credits are non-negative
    if (df_model_output['NEV Credits'] < 0).any():
        return float('inf')  # Return a large number as penalty

    total_profit = model_output['Profit (￥/veh)'].sum()
    return -total_profit  # Since Evolopy minimizes the objective function, we return negative profit

def calculate_model_output(df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market):
    model_output = pd.concat([df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market], axis=1)
    # Call the new Profit function to calculate profit
    model_output['Profit (￥/veh)'] = Profit(
        df_model_personal_vehicle_market['Sales (1000)'],
        df_model_personal_vehicle_market['Price to consumers (￥)'],
        df_model_fleet_vehicle_market['Sales (1000)'],
        df_model_fleet_vehicle_market['Price to consumers (￥)'],
        df_model['Total Sales (1000)'],
        df_model['Additional Subsidy to Consumers (￥)'],
        df_model['Central Gov Sub to Consumers (￥)'],
        df_model['Avg Local Gov Sub to Consumers (￥)'],
        df_model['Internal Subsidy to Consumer(￥)'],
        df_model['Production Cost (￥)'],
        df_model_output['Veh Total Sales (1,000)'])
    return model_output

def Profit(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    result = ((np.sum(x1[:16] * x2[:16]) +
              np.sum(x3[:16] * x4[:16]) +
              np.sum(x5[:16] * x6[:16]) +
              np.sum(x5[:16] * x7[:16]) +
              np.sum(x5[:16] * x8[:16]) -
              np.sum(x5[:16] * x9[:16]) -
              np.sum(x5[:16] * x10[:16])) * 1000) / x11 / 1000
    return result

run(optimizer, objectivefunc, NumOfRuns, params, export_flags)


# In[209]:


# Function to handle folder selection and display the CSV content
def display_csv_content(foldername):
    # Construct the full path to the CSV file
    filename = os.path.join(foldername, 'experiment.csv')
    
    # Read and display the first 6 rows of the CSV file
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        display(df.head(6).applymap(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))
    else:
        print("File not found:", filename)

# Create a dropdown widget for folder selection
foldernames = [filename for filename in os.listdir() if filename.startswith(str(datetime.datetime.now().year)) and os.path.isdir(filename)]
drop_folder = widgets.Dropdown(options=foldernames, description='Select folder:')
drop_folder

# Show the widget and bind the display function to changes in the dropdown
widgets.interact(display_csv_content, foldername=drop_folder)


# In[210]:


def display_csv_and_convergence_curve(foldername):
    # Display the CSV content
    csv_filename = os.path.join(foldername, 'experiment.csv')
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        display(df.head(6).applymap(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))
    else:
        print("CSV file not found:", csv_filename)
    
    # Function to handle convergence plot selection and display
    def show_convergence_curve(plot_filename):
        image_path = os.path.join(foldername, plot_filename)
        if os.path.exists(image_path):
            display(Image(filename=image_path))
        else:
            print("Image file not found:", image_path)
    
    # Select convergence curve to show
    filenames = [filename for filename in os.listdir(foldername) if filename.startswith('convergence')]
    drop_plot_convergence = widgets.Dropdown(options=filenames, description='Select plot:')
    
    # Interact for plot selection and display
    widgets.interact(show_convergence_curve, plot_filename=drop_plot_convergence)

# Create a dropdown for folder selection
foldernames = [filename for filename in os.listdir() if filename.startswith(str(datetime.datetime.now().year)) and os.path.isdir(filename)]
drop_folder = widgets.Dropdown(options=foldernames, description='Select folder:')

# Show the widget and bind the display function to changes in the dropdown
widgets.interact(display_csv_and_convergence_curve, foldername=drop_folder)


# In[211]:


# Function to display the CSV content and boxplots
def display_boxplot_image(foldername):
    # Display the CSV content
    csv_filename = os.path.join(foldername, 'experiment.csv')
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        display(df.head(6).applymap(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))
    else:
        print("CSV file not found:", csv_filename)
    
    # Function to handle boxplot selection and display
    def show_boxplot_image(plot_filename):
        image_path = os.path.join(foldername, plot_filename)
        if os.path.exists(image_path):
            display(Image(filename=image_path))
        else:
            print("Image file not found:", image_path)
    
    # Select boxplot to show
    filenames = [filename for filename in os.listdir(foldername) if filename.startswith('boxplot')]
    if filenames:
        drop_plot_box = widgets.Dropdown(options=filenames, description='Select plot:')
        widgets.interact(show_boxplot_image, plot_filename=drop_plot_box)
    else:
        print("No boxplot images found in the selected folder.")

# Create a dropdown for folder selection
foldernames = [filename for filename in os.listdir() if filename.startswith(str(datetime.datetime.now().year)) and os.path.isdir(filename)]
if foldernames:
    drop_folder = widgets.Dropdown(options=foldernames, description='Select folder:')
    widgets.interact(display_boxplot_image, foldername=drop_folder)
else:
    print("No folders from the current year found.")


# In[212]:


def newrun(optimizer, objectivefunc, NumOfRuns, params, export_flags, df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market):
    results = {}
    global_best_solution = None
    global_best_value = np.inf

    for opt in optimizer:
        best_solution = None
        best_value = np.inf
        for _ in range(NumOfRuns):
            solution = np.random.uniform(0, 10000, 16)  # Generate 16 subsidy values
            value = calculate_profit(solution)
            if value < best_value:
                best_value = value
                best_solution = solution
        # Normalize the best solution by subtracting the smallest subsidy value
        best_solution -= np.min(best_solution)
        results[opt] = (best_solution, -best_value)
        print(f"Best solution for {opt}: {best_solution}, Profit: {-best_value}")

        # Track the overall best across optimizers
        if best_value < global_best_value:
            global_best_value = best_value
            global_best_solution = best_solution

    # Update the model with the best overall solution normalized
    global_best_solution -= np.min(global_best_solution)  # Ensure at least one subsidy value is zero
    df_model.loc[:15, 'Internal Subsidy to Consumer(￥)'] = global_best_solution

    # Recalculate the optimal model output using the best subsidy values
    optimal_model_output = calculate_model_output(df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market)

    # Print the formatted optimal model output
    print("Optimal output model_output:")
    print(optimal_model_output.applymap(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x))

# Set up and run the optimization including direct handling
newrun(optimizer, objectivefunc, NumOfRuns, params, export_flags, df_model, df_model_personal_vehicle_market, df_model_fleet_vehicle_market)


# In[256]:


# Car types and their corresponding indices
cars = ['ICE-Hig (Sedan)', 'ICE-Avg (Sedan)', 'ICE-Low (Sedan)', 'PHEV (Sedan)', 'BEV-Sht (Sedan)',
        'BEV-Mid (Sedan)', 'BEV-Lng (Sedan)', 'FCEV (Sedan)', 'ICE-Hig (SUV)', 'ICE-Avg (SUV)',
        'ICE-Low (SUV)', 'PHEV (SUV)', 'BEV-Sht (SUV)', 'BEV-Mid (SUV)', 'BEV-Lng (SUV)', 'FCEV (SUV)']

# Optimal Internal Subsidy combination (adjusted) and Maximum profit (adjusted) for four scenarios
data = {
    'No rules': {
        'Optimal Internal Subsidy combination': [8, 22, 32, 46, 12, 15, 24, 1919, 6, 12, 19, 21, 11, 0, 13, 6322],
        'Maximum profit': 56936.51445293339
    },
    'Dual-credit Policy': {
        'Optimal Internal Subsidy combination': [4, 24, 34, 21, 14, 14, 26, 3042, 8, 14, 0, 8, 6, 2, 15, 9154],
        'Maximum profit': 56936.58641432177
    },
    'CAFC Policy Only': {
        'Optimal Internal Subsidy combination': [4, 3, 19, 5, 14, 4, 26, 5613, 8, 14, 0, 2, 6, 2, 3, 5891],
        'Maximum profit': 56941.726610584235
    },
    'NEV Policy Only': {
        'Optimal Internal Subsidy combination': [0, 3, 17, 5, 3, 4, 12, 6062, 6, 5, 0, 2, 6, 2, 3, 6478],
        'Maximum profit': 56946.37670315736
    }
}

# Prepare the data
x = np.arange(len(cars))
width = 0.2

# Create subplots
fig, ax = plt.subplots(figsize=(22, 12))

# Plot the bar chart
for i, (policy, values) in enumerate(data.items()):
    subsidy_combination = values['Optimal Internal Subsidy combination']
    profit = values['Maximum profit']
    
    # Plot the bars
    bars = ax.bar(x + i*width, subsidy_combination, width, label=f'{policy}\nMaximum profit(￥/veh): {profit:.2f}')
    
    # Add values on top of each bar
    for bar, value in zip(bars, subsidy_combination):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, value,
                ha='center', va='bottom', fontsize=10)

# Set axis labels and title
ax.set_xlabel('Car Types', fontsize=20)
ax.set_ylabel('Optimal Internal Subsidy Combination (adjusted), (￥/veh)')
ax.set_title('Optimal Internal Subsidy Combination (adjusted) under Different Policies', fontsize=25)

# Set x-axis ticks
ax.set_xticks(x + 1.5*width)
ax.set_xticklabels(cars, rotation=45, ha='right', fontsize=20)

# Set y-axis label font size
ax.yaxis.set_label_coords(-0.05, 0.5)
ax.yaxis.label.set_size(20)
ax.tick_params(axis='y', labelsize=20)

# Add legend
ax.legend(prop={'size': 20})

# Adjust layout
plt.subplots_adjust(left=0, right=0.9, top=0.9, bottom=0.1)

# Show the plot
plt.tight_layout()
plt.savefig(r'D:\Users\11379\Desktop\博士\博士小论文\NEOCC\optimal_subsidy_combination.png')
plt.show()


# In[ ]:




