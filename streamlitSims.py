# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 09:50:37 2023

@author: hunte
"""
import Optimizer as opt
import pandas as pd
import numpy as np
import concurrent.futures
pd.options.mode.chained_assignment = None


class sims():
    
    def __init__(self):
        return
    
    def scramble_projections(df, fpts_column,ceil_column=None, floor_column=None):
        return
        
    def standard_sims(self, df, count, fpts_col_name='Fpts', ceil_column=None, floor_column=None,ownership_column = None,status_bar=None,stack = None):
        '''
        returns a datarame of optimal rates as well as an array of simulated winning lineups
        
        input a df of projections and a model object from the optimizer class
        '''
        
        df = df.drop_duplicates(subset = ['Name','Team',fpts_col_name])
        df = self.optimizer.prep_df()

        #run sims
        inputs = [(df, fpts_col_name,ceil_column,floor_column,'Observed Fpts',stack)] * count
        with concurrent.futures.ThreadPoolExecutor(max_workers = 3) as e:
            lineup_list = list(e.map(self.scramble_and_optimize,inputs))

        #format output
        observed_scores = [x[1] for x in lineup_list]
        lineup_list = [x[0] for x in lineup_list]
        player_list = []
        for lineup in lineup_list:
            for player in lineup:
                player_list.append(player)

        #add results to dataframe
        counts = pd.DataFrame(player_list, columns = ['Name','Position','Team']).value_counts()
        counts = pd.DataFrame(counts).rename(columns = {0 : 'count'}).reset_index()
        df = df.merge(counts, how='left', on=['Name','Position','Team'])
        
        #calculations
        df['Optimal Ownership'] = (df['count']/count)*100
        include_columns = ['Name','Position','Team','Opp','Salary','Optimal Ownership','Efficiency', fpts_col_name]
        if ownership_column is not None:
            df['Leverage'] = df['Optimal Ownership'] - df[ownership_column] 
            include_columns += [ownership_column, 'Leverage']
            
        #filter and sort
        df = df[include_columns]
        df = df.sort_values(by = ['Position','Optimal Ownership'], ascending = False).set_index('Name')
        df = df[df['Optimal Ownership'].isnull()==False]
        
        #get lineup total scores
        expected_scores = [self.get_total_lineup_score(lineup, fpts_col_name) for lineup in lineup_list]
        #take the weighted average of the actual score and the expected score
        scores = [((expected_scores[i]*10)+(observed_scores[i]))/11 for i in range(count)]
        lineups = list(zip(lineup_list,scores))
        
        #lineups format: list of tuples of form ([name,position,team],lineup_score)
        return df, lineups
    
    def get_total_lineup_score(self, lineup, fpts_col_name):
        d = pd.DataFrame(lineup, columns = ['Name','Position','Team'])
        df = d.merge(self.df, how = 'left', on = ['Name','Position','Team'])
        score = df[fpts_col_name].sum()
        return score
    
class mlb(sims):
    
    def __init__(self, df):
        self.correlation_values = {'Hitter':.40, 'Pitcher':.30}
        self.optimizer = opt.MLB(df)
        self.df = df
        return
    
    def scramble_and_optimize(self,args):
        df, fpts_col_name,ceil_column,floor_column,objective_fn_column,stack = args
        df['Observed Fpts'] = self.scramble_projections(df, fpts_col_name, ceil_column, floor_column)
        lineup,score = opt.MLB(df).standard_optimizer(df, objective_fn_column='Observed Fpts',return_score = True,stack=stack)
        return lineup,score
 
    def scramble_projections(self, df, fpts_column, ceil_column=None, floor_column=None):
       '''
       returns a formatted dataframe of scrambled fpts results for players in a dataframe
       
       correlations passed as nested dict; nulls should be passed as 0's'
       
       TO: add correlation for opponents
       '''
       
       #prep work, fill nulls
       if ceil_column:
           df[ceil_column] = df[ceil_column].fillna(df[fpts_column]*2)
       if floor_column:
           df[floor_column] = df[floor_column].fillna(df[fpts_column]/2)
           
       fpts_values = list(df[fpts_column]) #predicted fpts values
       
       #get team hitter results, use as base for hitters
       teams = list(set(df['Team']))
       team_result = [np.random.normal(1,.40) for t in teams]
       t_df = pd.DataFrame({'Team':teams, 'team_result':team_result})
       df = df.merge(t_df, how = 'left', on = 'Team')
       
       #bias down opposing pitchers and get the individual results
       slices = []
       for t in teams:
           try:
               opp_team_result = df.loc[df['Opp']==t, 'team_result'].reset_index().at[0, 'team_result']
           except KeyError:
               opp_team_result = 0
           p_sl = df[(df['Team']==t)&(df['Position'] == 'SP')]
           h_sl = df[(df['Team']==t)&(df['Position'] != 'SP')]
           p_sl['pitcher_result'] = 1-(.33*opp_team_result)
           p_sl['results'] = [np.random.normal(pr, .50) for pr in p_sl['pitcher_result']]
           h_sl['results'] = [np.random.normal(hr, .50) for hr in h_sl['team_result']]
           slices.append(p_sl)
           slices.append(h_sl)
           
       tmp = pd.concat(slices)
       
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

class nfl(sims):

    def __init__(self, df): 
        self.correlation_values = {'QB':{'WR':.66, 'TE':.33, 'Opp_QB':0.0}}
        self.optimizer = opt.NFL(df)
        self.df = df
        return
    
    def scramble_projections(self, df, fpts_column, ceil_column=None, floor_column=None):
        '''
        returns a formatted dataframe of scrambled fpts results for players in a dataframe
        
        correlations passed as nested dict; nulls should be passed as 0's'
        
        TO: add correlation for opponents
        '''
        #store variables
        qb_correlated_positions = list(self.correlation_values['QB'].keys())
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
            net_qb_result = team_qb_result + (opponent_qb_result*self.correlation_values['QB']['Opp_QB'])
            for pos in qb_correlated_positions:
                pos_sl = df[(df['Pos'] == pos) & (df['Team'] == team)]
                pos_sl['results'] = [np.random.normal(net_qb_result*self.correlation_values['QB'][pos],1) for i in range(len(pos_sl))]
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
