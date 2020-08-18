# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:34:58 2020

@author: Zhou Yan
"""


import numpy as np
import pandas as pd
from arch.unitroot import ADF
import statsmodels.api as sm
from linearmodels import PanelOLS
import scipy.stats as ss


JC3 = pd.read_csv('#amended_JC3.csv', header=0)
JC3.index = JC3['date']
JC3 = JC3.drop(JC3[['year', 'date']], axis=1)


### Table 1
table1 = pd.DataFrame(index=['Iron Ore Price $', 'Growth Rate*10^-3', 'VIX Index', 'Ln (Distance)',
                             'Ln (Fuel Price)', 'Freight Rate $', 'Cargo Volume*10^4', 'BDI'],
                      columns=['Mean', 'Std Dev', '25%', 'Median', '75%'])
JC3_cols = JC3.columns.tolist()
pos = [2, 3, 6, 7, 8, 9, 10, 11] #position
t1_ind = [] #table1 index related to JC3 columns
for i in range(len(pos)):
    t1_ind.append(JC3_cols[pos[i]])
JC3 = JC3[t1_ind]
t1_cols = table1.columns.tolist()

table1[t1_cols[0]] = np.mean(JC3).tolist()
table1[t1_cols[1]] = np.std(JC3).tolist()
table1[t1_cols[2]] = np.percentile(JC3, 25, axis=0).tolist()
table1[t1_cols[3]] = np.percentile(JC3, 50, axis=0).tolist()
table1[t1_cols[4]] = np.percentile(JC3, 75, axis=0).tolist()
table1.iloc[1, :] = table1.iloc[1, :]*1000
table1.iloc[6, :] = table1.iloc[6, :]/10000


### Table 2
JC1 = pd.read_csv('JC1.csv', header=0)
JC1 = JC1[['ore_price', 'growth', 'VIX', 'BDI']]
JC1_cols = JC1.columns.tolist()

adf_iron = ADF(JC1[JC1_cols[0]])
adf_iron.trend = 'c'

adf_growth = ADF(JC1[JC1_cols[1]])
adf_growth.trend = 'c'

adf_vix = ADF(JC1[JC1_cols[2]])
adf_vix.trend = 'c'

adf_BDI = ADF(JC1[JC1_cols[3]])
adf_BDI.trend = 'c'

adf_stat = [adf_iron.stat, adf_growth.stat, adf_vix.stat, adf_BDI.stat]
adf_p = [adf_iron.pvalue, adf_growth.pvalue, adf_vix.pvalue, adf_BDI.pvalue]

table2 = pd.DataFrame(index=['ADF-Statistics', 'p-Value', '1%', '5%', '10%'],
                      columns=['Iron Ore Price $', 'Growth Rate', 'VIX Index', 'BDI'])

t2_ind = table2.index.tolist()
t2_cols = table2.columns.tolist()

table2.loc[t2_ind[0], :] = adf_stat
table2.loc[t2_ind[1], :] = adf_p
table2.loc[t2_ind[2:5], t2_cols[0]] = adf_iron.critical_values
table2.loc[t2_ind[2:5], t2_cols[1]] = adf_growth.critical_values
table2.loc[t2_ind[2:5], t2_cols[2]] = adf_vix.critical_values
table2.loc[t2_ind[2:5], t2_cols[3]] = adf_BDI.critical_values


#table2.to_csv('Table_2.csv')


### Table 3
#JC1.iloc[:, 2] = JC1.iloc[:, 2]*(-1)

x = JC1[['growth', 'VIX', 'BDI']]
x = sm.add_constant(x)
y = JC1['ore_price']

model_t3 = sm.OLS(y, x).fit(cov_type = 'HC1')

model_t3.summary()

residual_t3 = model_t3.resid
adf_resi_t3 = ADF(residual_t3)
adf_resi_t3.summary()

       # Table for coefficient
table3_1 = pd.DataFrame(index=['theta_0', 'theta_1', 'theta_2', 'theta_3'],
                        columns=['Estimated Coefficient', 'Standard Error', 't-stats', 'p-Value'])

t3_1_cols = table3_1.columns.tolist()
table3_1[t3_1_cols[0]] = model_t3.params.tolist()
table3_1[t3_1_cols[1]] = model_t3.bse.tolist()
table3_1[t3_1_cols[2]] = model_t3.tvalues.tolist()
table3_1[t3_1_cols[3]] = model_t3.pvalues.tolist()

      # Table for F-stat, R^2, ADF
table3_2 = pd.DataFrame(index=['F-statistic', 'F-stat-pvalue', 'R^2', 'ADF-statistics Residual Value',
                               'ADF-stat-pvalue'],
                        columns=['Value'])

t3_2_ind = table3_2.index.tolist()
table3_2.loc[t3_2_ind[0], :] = model_t3.fvalue
table3_2.loc[t3_2_ind[1], :] = model_t3.f_pvalue.tolist()
table3_2.loc[t3_2_ind[2], :] = model_t3.rsquared_adj.tolist()
table3_2.loc[t3_2_ind[3], :] = adf_resi_t3.stat.tolist()
table3_2.loc[t3_2_ind[4], :] = adf_resi_t3.pvalue.tolist()



### Table 4
JC3 = pd.read_csv('###amended_JC3.csv', header=0)
JC3 = JC3.set_index(['port', 'year'])

JC3_cols = JC3.columns.tolist()
pos = [2, 3, 6, 7, 8, 9, 10, 11] #position
t1_ind = [] #table1 index related to JC3 columns
for i in range(len(pos)):
    t1_ind.append(JC3_cols[pos[i]])
JC3 = JC3[t1_ind]

#JC3['VIX'] = JC3['VIX']*(-1)

x = ['growth', 'logd', 'logf', 'ore_price', 'BDI']
exog = sm.add_constant(JC3[x])

#model_t4 = PanelOLS(JC3.avefreight, exog, entity_effects=True).fit(cov_type='clustered', 
#                                                                   cluster_entity=True)

model_t4 = PanelOLS(JC3.avefreight, exog, entity_effects=False).fit(cov_type='clustered', 
                                                                    cluster_entity=True)


      # Table for coefficients
table4_1 = pd.DataFrame(index=['gamma_0', 'gamma_1', 'gamma_2', 'gamma_3', 'gamma_4', 'gamma_5'],
                        columns=['Estimated Coefficient', 'Standard Error', 't-stats', 'p-Value'])

t4_1_cols = table4_1.columns.tolist()
table4_1[t4_1_cols[0]] = model_t4.params.tolist()
table4_1[t4_1_cols[1]] = model_t4.std_errors.tolist()
table4_1[t4_1_cols[2]] = model_t4.tstats.tolist()
table4_1[t4_1_cols[3]] = model_t4.pvalues.tolist()


      # Table for F-stat and R^2
table4_2 = pd.DataFrame(index=['F-statistic', 'F-stat-pvalue', 'R^2'], columns=['Value'])

t4_2_ind = table4_2.index.tolist()
table4_2.loc[t4_2_ind[0], :] = model_t4.f_statistic.stat
table4_2.loc[t4_2_ind[1], :] = model_t4.f_statistic.pval
table4_2.loc[t4_2_ind[2], :] = model_t4.rsquared


### Table 5

#x = ['growth', 'VIX']
#exog = sm.add_constant(JC3[x])
        ### first stage regression on oil price
#model_t5_1 = PanelOLS(JC3.ore_price, exog, entity_effects=True).fit(cov_type='clustered', 
                                                                    #cluster_entity=True)

#theta0 = model_t5_1.params[0]
#theta1 = model_t5_1.params[1]
#theta2 = model_t5_1.params[2]

#JC3['P estimate'] = model_t5_1.predict()
#JC3['P estimate'] = theta0 + theta1*JC3['growth'] + theta2*JC3['VIX']

#JC3['P estimate'] = JC3['P estimate']*(-1)

    
JC3['P estimate'] = list(model_t3.predict())*4

#JC3['P estimate'] = JC3['P estimate']*(-1)

        ### second stage regression on freight
x = ['growth', 'logd', 'logf', 'P estimate', 'BDI']
exog = sm.add_constant(JC3[x])

model_t5_2 = PanelOLS(JC3.avefreight, exog, entity_effects=False).fit(cov_type='clustered', 
                                                                    cluster_entity=True)

model_t5_2.summary

     # Table for coefficient
table5_1 = pd.DataFrame(index=['gamma_0', 'gamma_1', 'gamma_2', 'gamma_3', 'gamma_4', 'gamma_5'],
                        columns=['Estimated Coefficient', 'Standard Error', 't-stats', 'p-Value'])

t5_1_cols = table5_1.columns.tolist()
table5_1[t5_1_cols[0]] = model_t5_2.params.tolist()
table5_1[t5_1_cols[1]] = model_t5_2.std_errors.tolist()
table5_1[t5_1_cols[2]] = model_t5_2.tstats.tolist()
table5_1[t5_1_cols[3]] = model_t5_2.pvalues.tolist()

      # Table for F-stat and R^2
table5_2 = pd.DataFrame(index=['F-statistic', 'F-stat-pvalue', 'R^2'],
                        columns=['Value'])

t5_2_ind = table5_2.index.tolist()
table5_2.loc[t5_2_ind[0], :] = model_t5_2.f_statistic.stat
table5_2.loc[t5_2_ind[1], :] = model_t5_2.f_statistic.pval
table5_2.loc[t5_2_ind[2], :] = model_t5_2.rsquared




### Table 6
        #ADF test
JC3['residual'] = model_t5_2.resids.tolist()


port = ['DAMPIER', 'PORT HEDLAND', 'SALDANHA BAY', 'TUBARAO']

adf_nu1 = ADF(JC3.loc[port[0], 'residual'])
adf_nu2 = ADF(JC3.loc[port[1], 'residual'])
adf_nu3 = ADF(JC3.loc[port[2], 'residual'])
adf_nu4 = ADF(JC3.loc[port[3], 'residual'])

adf_stat = [adf_nu1.stat, adf_nu2.stat, adf_nu3.stat, adf_nu4.stat]
adf_p = [adf_nu1.pvalue, adf_nu2.pvalue, adf_nu3.pvalue, adf_nu4.pvalue]

table6_1 = pd.DataFrame(index=['ADF-Statistics of nu1', 'ADF-Statistics of nu2', 
                             'ADF-Statistics of nu3', 'ADF-Statistics of nu4'],
                        columns=['Estimate', 'p-Value'])

t6_ind = table6_1.index.tolist()
t6_cols = table6_1.columns.tolist()

table6_1.loc[:, t6_cols[0]] = adf_stat
table6_1.loc[:, t6_cols[1]] = adf_p

JC3['residual_op'] = list(model_t3.resid)*4

r1 = ss.pearsonr(JC3.loc[port[0], 'residual_op'], JC3.loc[port[0], 'residual'])
r2 = ss.pearsonr(JC3.loc[port[1], 'residual_op'], JC3.loc[port[1], 'residual'])
r3 = ss.pearsonr(JC3.loc[port[2], 'residual_op'], JC3.loc[port[2], 'residual'])
r4 = ss.pearsonr(JC3.loc[port[3], 'residual_op'], JC3.loc[port[3], 'residual'])

r_stat = [r1[0], r2[0], r3[0], r4[0]]
r_p = [r1[1], r2[1], r3[1], r4[1]]

table6_2 = pd.DataFrame(index=['Sample correlation(e, nu1)', 'Sample correlation(e, nu2)', 
                               'Sample correlation(e, nu3)', 'Sample correlation(e, nu4)'],
                        columns=['Estimate', 'p-Value'])

t6_ind = table6_2.index.tolist()
t6_cols = table6_2.columns.tolist()

table6_2.loc[:, t6_cols[0]] = r_stat
table6_2.loc[:, t6_cols[1]] = r_p
table6 = pd.concat([table6_1, table6_2])


print(table1.round(2))

print(table2)

print(table3_1)

print(table3_2)

print(table4_1)

print(table4_2)

print(table5_1)

print(table5_2)

print(table6)



