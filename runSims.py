# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 14:39:25 2021

@author: HTreschl
"""

import Optimizer as opt
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None


def scramble_projections(df, fpts_column,correlation_values = {'QB':{}}, ceil_column=None, floor_column=None):
    '''
    returns a formatted dataframe of scrambled fpts results for players in a dataframe. 
    Includes correlations between positions and scaling between ceiling and floor estimates, if applicable.
    
    correlations passed as nested dict. 
    '''
    #store variables
    qb_correlated_positions = list(correlation_values['QB'].keys())
    #prep work, fill nulls
    if ceil_column:
        df[ceil_column] = df[ceil_column].fillna(df[fpts_column]*2)
    if floor_column:
        df[floor_column] = df[floor_column].fillna(df[fpts_column]/2)
        
    fpts_values = list(df[fpts_column]) #predicted fpts values
    slice_list = [] #placeholder for df slices
    
    #get QB results
    qbs = df[df['Pos']=='QB']
    qbs['results'] = [np.random.normal(1,1) for i in range(len(qbs))]
    slice_list.append(qbs)
    
    #get other results -- same team
    for team in qbs['Team']:
        team_qb_result = qbs.loc[qbs['Team']==team, 'results'].reset_index().at[0, 'results'] #the value for that teams' QB; use as base for the other slices
        opponent_qb_result = qbs.loc[qbs['Opp']==team, 'results'].reset_index().at[0, 'results']
        net_qb_result = team_qb_result + (opponent_qb_result*correlation_values['QB']['Opp_QB'])
        for pos in qb_correlated_positions:
            pos_sl = df[(df['Pos'] == pos) & (df['Team'] == team)]
            pos_sl['results'] = [np.random.normal(net_qb_result*correlation_values['QB'][pos],1) for i in range(len(pos_sl))]
            slice_list.append(pos_sl)
    correlated_results = pd.concat(slice_list)[['results']]
    tmp = df.merge(correlated_results, how = 'left', left_index=True, right_index = True)
    tmp = tmp.drop_duplicates(subset = ['Name','Team',fpts_column])
    #fill the empties with simple random values
    tmp['results'] = [np.random.normal(1,1) if np.isnan(x) else x for x in tmp['results']]
    res = list(tmp['results'])

    #generate fpts values
    observed_results = []
    if ceil_column and floor_column: #ceiling and floor supplied
        ceil_column = list(df[ceil_column])
        floor_column = list(df[floor_column])
        results_arr = np.array((fpts_values, ceil_column, floor_column, res))
        for i in range(results_arr.shape[1]):
            mean_dist = results_arr[3,i]-1
            if mean_dist < 0:
                floor_dist = results_arr[0,i] - results_arr[2,i]
                r = results_arr[0,i] - (floor_dist*abs(mean_dist))
                observed_results.append(r)
            else:
                ceil_dist = results_arr[1,i] - results_arr[0,i]
                r = results_arr[0,i] + (ceil_dist*mean_dist)
                observed_results.append(r)
    else: #no ceiling and floor
        observed_results = [res[i]*fpts_values[i] for i in range(len(fpts_values))]
    return observed_results


def standard_sims(df, sport, count,correlation_values = {'QB':{'WR':.66, 'TE':.33, 'Opp_QB':0.0}}, fpts_col_name='Fpts', ceil_column=None, floor_column=None,ownership_column = None):
    '''
    returns a datarame of optimal rates as well as an array of simulated winning lineups
    
    input a df of projections and a model object from the optimizer class
    '''
    
    df = df.drop_duplicates(subset = ['Name','Team',fpts_col_name])
    if sport == 'mlb':
        model = opt.MLB(df)
        df = model.prep_df()
    elif sport == 'nfl':
        model = opt.NFL(df)
        optimizer = model.standard_optimizer
    
        
    lineup_list = []
    
    for i in range(count):
        df['Observed Fpts'] = scramble_projections(df, fpts_col_name,correlation_values, ceil_column, floor_column)
        lineup = optimizer(df, objective_fn_column='Observed Fpts')
        lineup_list.append(set(lineup))
        

    player_list = []
    for lineup in lineup_list:
        for player in lineup:
            player_list.append(player)
            
    counts = pd.DataFrame(player_list).rename(columns = {0 : 'Name'}).value_counts()
    counts = pd.DataFrame(counts).rename(columns = {0 : 'Count'}).reset_index()
    
    df = df.merge(counts, how='left', on='Name')
    #calculations
    df['Optimal Ownership'] = (df['Count']/count)*100
    include_columns = ['Name','Pos','Team','Opp','Salary','Optimal Ownership', fpts_col_name]
    if ownership_column is not None:
        df['Leverage'] = df['Optimal Ownership'] - df[ownership_column] 
        include_columns += [ownership_column, 'Leverage']
    #filter and sort
    df = df[include_columns]
    df = df.sort_values(by = ['Pos','Optimal Ownership'], ascending = False).set_index('Name')
    df = df[df['Optimal Ownership'].isnull()==False]
    return df, lineup_list

def showdown_sims(df, count, fpts_column, ceil_column = None, floor_column = None):

    model = opt.NFL(df=df)
    
    lineup_list = []
    
    for i in range(count):
        df['observed fpts'] = scramble_projections(df, fpts_column, ceil_column, floor_column)
        lineup = model.showdown_optimizer(df, 'observed fpts')
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
    df['Optimal Ownership'] = df['Count']/count
    df['optimal Cpt'] = df['Cpt Count']/count
    
    #df.to_excel('showdown optimal Ownership.xlsx')
    
    return df

def get_team_optimal(sims, include_defense = False, include_rb = False):
    if not include_defense:
        sims = sims[sims['Pos']!= 'DST']
    if not include_rb:
        sims = sims[sims['Pos']!= 'RB']
    res = sims.groupby('Team')[['Leverage', 'Optimal Ownership']].sum()
    return res

def parse_lineup_list(lineups):
    '''
    given a list of lists made up of the sims optimal, returns the most common player pairings for each player
    '''
    players = set()
    for lineup in lineups:
        [players.add(x) for x in lineup]
      
    players = list(players)
    res_dict = {}    
    for player in players:
        rel_lineups = [x for x in lineups if player in x]
        flat_lineups = [x for y in rel_lineups for x in y]
        counts = pd.DataFrame(flat_lineups, columns=['Name']).groupby('Name')['Name'].count()
        counts = pd.DataFrame(counts).rename(columns = {'Name':'Count'}).reset_index().sort_values(by = 'Count', ascending=False)
        res_dict[player] = counts
    return res_dict
        
def lineup_parser(lineups, crit):
    '''given a list of lineups from sims, returns the lineups including the players specified in crit'''
    rel_lineups = [x for x in lineups if crit.issubset(x)]
    flat_lineups = [x for y in rel_lineups for x in y]
    counts = pd.DataFrame(flat_lineups, columns=['Name']).groupby('Name')['Name'].count()
    counts = pd.DataFrame(counts).rename(columns = {'Name':'Count'}).reset_index().sort_values(by = 'Count', ascending=False)
    return counts
