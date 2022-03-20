# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 14:39:25 2021

@author: hunte
"""

import Optimizer as opt
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None


df = pd.read_excel('NFL DK Projections.xlsx')


def standard_sims(df):
    model = opt.NFL(df=df)
    
    lineup_list = []
    
    for i in range(1000):
        results = model.fpts_scrambler(team_corr = True, passer_corr = True)
        lineup = model.standard_optimizer(results, objective_fn_column='observed fpts')
        lineup_list.append(lineup)
    
    player_list = []
    for lineup in lineup_list:
        for player in lineup:
            player_list.append(player)
            
    counts = pd.DataFrame(player_list).rename(columns = {0 : 'Name'}).value_counts()
    counts = pd.DataFrame(counts).rename(columns = {0 : 'Count'}).reset_index()
    
    df = df.merge(counts, how='left', on='Name')
    df['Optimal Ownership'] = df['Count']/1000
    
    #add way to see most frequent pairings (all players by all other players, do counts)
    pairings = pd.DataFrame(np.zeros((len(set(player_list)), len(set(player_list)))), columns=list(set(player_list)), index = list(set(player_list)))
    for lineup in lineup_list:
        for player_1 in lineup:
            for player_2 in lineup:
                if player_1 != player_2:
                    pairings.loc[player_1, player_2] += 1
            
    
    writer = pd.ExcelWriter('Players with optimal ownership.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='optimal ownership')
    pairings.to_excel(writer, sheet_name='player pairings')
    writer.save()
    return df

#%%showdown
df = pd.read_csv('blitz.csv')

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