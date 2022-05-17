# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 14:39:25 2021

@author: hunte
"""

import Optimizer as opt
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

def scramble_projections(fpts_column, ceil_column=None, floor_column=None):
    res = [np.random.normal(1,0.5)*x for x in fpts_column]
    return res
        

def standard_sims(df, sport, count, fpts_col_name='Fpts'):
    '''input a df of projections and a model object from the optimizer class'''
    if sport == 'mlb':
        model = opt.MLB(df)
        df = model.prep_df()
    

    lineup_list = []
    
    for i in range(count):
        df['Observed Fpts'] = scramble_projections(df[fpts_col_name])
        lineup = model.standard_optimizer(df, objective_fn_column='Observed Fpts')
        lineup_list.append(lineup)

    player_list = []
    for lineup in lineup_list:
        for player in lineup['Name']:
            player_list.append(player)
            
    counts = pd.DataFrame(player_list).rename(columns = {0 : 'Name'}).value_counts()
    counts = pd.DataFrame(counts).rename(columns = {0 : 'Count'}).reset_index()
    
    df = df.merge(counts, how='left', on='Name')
    #calculations
    df['Optimal Ownership'] = (df['Count']/count)*100
    df['Leverage'] = df['Optimal Ownership'] - df['Pown']
 
    #filter and sort
    df = df[['Name','Position','Team','Opp','Salary','Fpts','Pown','Leverage','Optimal Ownership']]
    df = df.sort_values(by = ['Position','Leverage'], ascending = False).set_index('Name')
    return df

#%%showdown

def showdown_sims(df):

    model = opt.NFL(df=df)
    
    lineup_list = []
    
    for i in range(1000):
        results = model.fpts_scrambler(team_corr = False, passer_corr = True)
        results = results.rename(columns = {'observed fpts':'avg fpts'})
        lineup = model.showdown_optimizer(results)
        lineup_list.append(lineup)
    
    player_list = []
    for lineup in lineup_list:
        for player in lineup:
            player_list.append(player)
    
    counts = pd.DataFrame(player_list).rename(columns = {0 : 'Name'}).value_counts()
    counts = pd.DataFrame(counts).rename(columns = {0 : 'Count'}).reset_index()
    cpt = counts[counts['Name'].str.contains('cpt')].rename(columns = {'Count':'Cpt Count'})
    cpt['Name'] = [x[:-4] for x in cpt['Name']]
    
    df = df.merge(counts, how='right', on='Name')
    df = df.merge(cpt, how='left', on='Name')
    df = df[df['Name'].str.contains(' cpt')==False]
    df['Optimal Ownership'] = df['Count']/1000
    df['optimal Cpt'] = df['Cpt Count']/1000
    
    df.to_excel('showdown optimal Ownership.xlsx')
    
    return df
