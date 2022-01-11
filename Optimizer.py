# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 18:24:22 2021

@author: Hunter Treschl
"""
import pandas as pd
import pulp
import numpy as np
import requests
import math
from bs4 import BeautifulSoup
import pickle
import webbrowser
import glob
import re
import os
import time

def get_download_path():
     """Returns the default downloads path for linux or windows"""
     if os.name == 'nt':
         import winreg
         sub_key = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
         downloads_guid = '{374DE290-123F-4565-9164-39C4925E467B}'
         with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
             location = winreg.QueryValueEx(key, downloads_guid)[0]
         return location
     else:
        return os.path.join(os.path.expanduser('~'), 'downloads')    

class MLB():
    '''MLB Optimizer'''
    
    def __init__(self, df): 
        self.salary_cap = 50000
        self.roster =['P','P','1B','2B','3B','SS','OF','OF','OF']
        self.df = df
        self.solver = pulp.getSolver('CPLEX_CMD')
        
    def prep_df(self):
        positions = self.df['Position'].apply(lambda x: pd.Series(x.split('/')))
        
        #create dummies for position
        dummies_1 = pd.get_dummies(positions[0])
        dummies_2 = pd.get_dummies(positions[1])
        
        #merge into one set of dummies
        p_dummies = dummies_1.merge(dummies_2, left_index=True, right_index=True)
        pos = ['1B','2B','3B','OF','SS']
        for p in pos:
            if p not in [col for col in p_dummies]: #corner case if a position has only players that don't flex
                p_dummies[p] = p_dummies[p+'_x'] + p_dummies[p + '_y']
        p_dummies = p_dummies[pos + ['P', 'C']]
        #merge
        df = self.df.merge(p_dummies, left_index=True, right_index=True)
        
        #create team dummies
        team_dummies = pd.get_dummies(df['Team'], prefix='t')
        df = df.merge(team_dummies, left_index=True, right_index=True)
        
        #create opponent dummies
        o_dummies = pd.get_dummies(df['Opponent'], prefix='o')
        df = df.merge(o_dummies, left_index=True, right_index=True)
        
        #pitchers and hitters
        df['hitters'] = np.where(df['Position']!='P', 1, 0)
        df['pitchers'] = np.where(df['Position']=='P', 1, 0)
        return MLB(df)
    
    
    def scale_projections(self, scaler=2.0, threshold = 12.0):
        '''Fpts columns named Fpts; owned = Ownership '''
        df = self.df
        df['adjuster'] =np.where(df['Ownership'].isnull(),100, 100 +  ((threshold - df['Ownership'])*scaler))
        df['Fpts'] = df['Fpts'] * df['adjuster']/100
        return MLB(df)
    
    
    def optimize_lineup(self, five_three=True, no_opps = True, prev_lineups=[], max_exposure=1.0):
        '''Columns: Name, Fpts, Position,Team, Salary'''
        
        players= self.df
        #prep lists
        teams = np.unique(np.array(players['Team'].astype(str)))
        opponents = np.unique(np.array(players['Opponent'].astype(str)))[:-1]
        lplayers = players['Name']
        #constraints
        constraint_dict = {}
        for col in players:
            if col != 'Name':
                d = dict(zip(lplayers,players[col]))
                constraint_dict[col] = d
        
        #define optimization
        prob = pulp.LpProblem('MLB', pulp.LpMaximize)
        
        #create lineup list
        lineup = pulp.LpVariable.dicts('players',lplayers, cat='Binary')
        
        #add max player constraint
        prob += pulp.lpSum(lineup[i] for i in lplayers)==10
        
        #player can only appear in certain # of lineups (format: dict of dataframes)
        if prev_lineups:
            for l in prev_lineups:
                prob += pulp.lpSum([prev_lineups[l][prev_lineups[l]['Name']==p]['in_lineup'] \
                                    *lineup[p] for p in lplayers]) <= math.ceil(max_exposure*len(prev_lineups))
        
        #create position count requirements
        prob += pulp.lpSum([constraint_dict['P'][f]*lineup[f] for f in lplayers]) == 2
        prob += pulp.lpSum([constraint_dict['C'][f]*lineup[f] for f in lplayers]) == 1
        prob += pulp.lpSum([constraint_dict['1B'][f]*lineup[f] for f in lplayers]) == 1   
        prob += pulp.lpSum([constraint_dict['2B'][f]*lineup[f] for f in lplayers]) == 1
        prob += pulp.lpSum([constraint_dict['3B'][f]*lineup[f] for f in lplayers]) == 1
        prob += pulp.lpSum([constraint_dict['SS'][f]*lineup[f] for f in lplayers]) == 1
        prob += pulp.lpSum([constraint_dict['OF'][f]*lineup[f] for f in lplayers]) == 3
        
        #add salary constraint
        prob += pulp.lpSum([lineup[f]*constraint_dict['Salary'][f] for f in lplayers]) <= 50000
        
        #add stack constraints (5/3)
        if five_three == True:
            stack_3 = pulp.LpVariable.dicts('teams', teams, cat='Binary')
            #stack_2 = pulp.LpVariable.dicts('teams1', teams, cat='Binary')
            prob += pulp.lpSum(stack_3[i] for i in teams) >= 1
            #prob += pulp.lpSum(stack_2[i] for i in teams) >= 2
            for t in teams:
                prob += (3*stack_3[t] <= pulp.lpSum([lineup[f]*constraint_dict['t_'+t][f]*constraint_dict['hitters'][f] for f in lplayers]))
                #prob += (2*stack_2[t] <= pulp.lpSum([lineup[f]*constraint_dict['t_'+t][f]*constraint_dict['hitters'][f] for f in lplayers]))
        
    
        #pitchers can't play against hitters
        if no_opps == True:
            for o in opponents:
                prob += (pulp.lpSum([lineup[f]*constraint_dict['pitchers'][f]*constraint_dict['o_'+o][f] for f in lplayers]) \
                    + pulp.lpSum([lineup[f]*constraint_dict['hitters'][f]*constraint_dict['t_'+o][f] for f in lplayers])) \
                    <=1
             
        #add objective function
        prob += pulp.lpSum([constraint_dict['Fpts'][f]*lineup[f] for f in lplayers])
        
        #solve the problem
        prob.solve(self.solver)
        
        #write to list of playernames
        
        lineup = [[v.name[8:].replace('_',' '), v.varValue] for v in prob._variables]
        return lineup[1:]
        
    
    def lineup_to_df(self, lineup):
        slineup = pd.DataFrame(lineup, columns=['Name','in_lineup'])
        df = self.df.merge(slineup, left_on='Name',right_on = 'Name', how='inner')
        return df
    
class NFL():
    
    def __init__(self,df):
        self.df=df[df['Salary'].isnull()==False]
        self.salary = 50000
        self.roster = ['QB','RB','RB','WR','WR','WR','TE','DST']
        self.num_players = 9
        self.solver = pulp.getSolver('CPLEX_CMD')
        
    def standard_optimizer(self, df, objective_fn_column = 'avg fpts'):
        '''returns the top lineup from the given dataframe for the standard contest type
        Columns = Name, Salary, Pos, Team, avg fpts'''

        
        #initial cleanup; get dummy variables for positions and drop nulls in target column
        pos_dummies = pd.get_dummies(df['Pos'])
        df = df.merge(pos_dummies,how='inner', left_index=True, right_index = True).set_index('Name')
        df = df[df[objective_fn_column].isnull() == False]
        
        #define the problem
        prob = pulp.LpProblem('NFL', pulp.LpMaximize)
        
        #create lineup list
        lineup = pulp.LpVariable.dicts('players',df.index, cat='Binary')
        
        #add max player constraint
        prob += pulp.lpSum([lineup[i] for i in df.index]) == 9
        
        #add position contraints
        prob += pulp.lpSum([df['QB'][f]*lineup[f] for f in df.index]) == 1
        prob += pulp.lpSum([df['RB'][f]*lineup[f] for f in df.index]) >= 2
        prob += pulp.lpSum([df['WR'][f]*lineup[f] for f in df.index]) >= 3
        prob += pulp.lpSum([df['TE'][f]*lineup[f] for f in df.index]) >= 1
        prob += pulp.lpSum([df['DST'][f]*lineup[f] for f in df.index]) == 1
        
        #add salary constraint
        prob += pulp.lpSum([df['Salary'][f]*lineup[f] for f in df.index]) <= 50000
        
        #add objective function
        prob.setObjective(pulp.lpSum([df[objective_fn_column][f]*lineup[f] for f in df.index]))
        
        prob.solve(self.solver)
        slns = [x.name[8:].replace('_',' ') for x in prob.variables() if x.varValue == 1]
        return slns
    
    def showdown_optimizer(self, df):
        '''returns the optimal lineup for a showdown slate
        columns = Name, Salary, avg fpts'''
        
        #df = pd.read_csv('showdown test.csv')
        
        #initial cleanup
        df = df[df['avg fpts'].isnull() == False]
        df = df[df['avg fpts']!=0]
        
        #get player dummies to dedupe captain
        df = df.merge(pd.get_dummies(df['Name']), how='inner', left_index=True, right_index=True)
        df = df.set_index('Name')
        players = list(df.index)
        
        #add in the CPT projections
        cpt_df = df.copy().reset_index()
        cpt_df['Name'] = cpt_df['Name'] + ' cpt'
        cpt_df = cpt_df.set_index('Name')
        cpt_df['is cpt'] = 1
        cpt_df['Salary'] = cpt_df['Salary'] *1.5
        cpt_df['avg fpts'] = cpt_df['avg fpts'] *1.5
        df = cpt_df.append(df)
        df['is cpt'] = df['is cpt'].fillna(0)
        
        #define the problem and add constraints
        prob = pulp.LpProblem('NFL', pulp.LpMaximize)
        
        #create lineup list
        lineup = pulp.LpVariable.dicts('players',df.index, cat='Binary')
        
        #add max player constraint
        prob += pulp.lpSum([lineup[i] for i in df.index]) == 6
        
        #add position contraints -- captain
        prob += pulp.lpSum([df['is cpt'][f]*lineup[f] for f in df.index]) == 1
        
        #ensure captains can't duplicate other players in the lineup
        for col in players:
            prob += pulp.lpSum([df[col][f]*lineup[f] for f in df.index]) <= 1
        #add salary constraint
        prob += pulp.lpSum([df['Salary'][f]*lineup[f] for f in df.index]) <= 50000
        
        #add objective function
        prob.setObjective(pulp.lpSum([df['avg fpts'][f]*lineup[f] for f in df.index]))
        
        prob.solve(self.solver)
        slns = [x.name[8:].replace('_',' ') for x in prob.variables() if x.varValue == 1]
        return slns
    
    def fpts_scrambler(self, team_corr = True, passer_corr = True):
        df = self.df
        
        #fill missing floor/ceil values
        df['avg floor'] = df['avg floor'].fillna(df['avg fpts'] * .5)
        df['avg ceil'] = df['avg ceil'].fillna(df['avg fpts'] * 1.5)
        
        #get observed results for teams -> correlates player results on teams (needs to be fixed)
        if team_corr == True:
            teams = df[['Team']].drop_duplicates()
            teams['Team Observed'] = np.random.normal(0,0.25,len(teams))
            df = df.merge(teams, how='left', on='Team')
        else:
            df['Team Observed'] = 0
        
        #create baseline results for players
        df['Player Observed'] = np.random.gamma(2,0.5, len(df))
        
        
        #add correlation for QBs and receivers
        if passer_corr == True:
            qb_results = df[df['Pos']=='QB'][['Team', 'Player Observed']].groupby('Team').mean()
            qb_results['rec adjuster'] = qb_results['Player Observed']*.25
            df = df.merge(qb_results.drop(columns='Player Observed'), how='left', on ='Team')
            df['Player Observed'] = np.where((df['Pos'] == 'WR') | (df['Pos']=='TE'), 
                                           df['Player Observed']+df['rec adjuster'], 
                                           df['Player Observed'])
        
        #convert into fpts by creating a ceil scaler and floor scaler that converts each .01 of observed to an fpts value
        df['adjust amount'] = np.where(df['Player Observed']<=0, \
                                 (df['avg fpts']-df['avg floor'])*df['Player Observed'], \
                                 (df['avg ceil'] - df['avg fpts'])*df['Player Observed'])
        df['observed fpts'] = df['avg fpts'] + df['adjust amount']
        #clean
        df.loc[df['observed fpts'] < 0, 'observed fpts'] = 0
        df = df.drop(columns = ['avg fpts'])
        return df
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    