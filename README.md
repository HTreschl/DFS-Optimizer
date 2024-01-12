# DFS Optimizer

A library designed to run sims and optimize lineups for DFS using python's pulp module. Includes NFL and MLB. Implemented in a webapp at https://roster-tech.streamlit.app/

# NFL Example -- Single Optimized Lineup for a Classic Game-Type
Optimize a single lineup without running any sims using the example projections for a classic dfs game

Import the data and initialize the NFL optimizer

```
import Optimizer as opt
df = pd.read_excel('NFL DK Projections.xlsx')
model = opt.nfl(df=df)
```

Optimize using the projection set we want, in this case averages of all available projections

```
lineup = model.standard_optimizer(df, objective_fn_column='avg fpts')
print(lineup)
```

Returns a list of players in the lineup, in this case 

['Curtis Samuel', 'Davante Adams', 'David Montgomery', 'Deebo Samuel', 'Derrick Henry', 'Josh Allen', 'Lions', 'Mike Davis', 'Will Dissly']

# Running Sims

the runSims module adds functionality to simulate results from each player and provide an optimal ownership calculation. Players' point predictions are sampled from a gamma distribution centered on their fantasy point projection and with a 95% CI between their floor and ceiling projections. Returns optimal ownership and leverage, where optimal ownership is the percentage of simulations in which the player was in the optimal lineup, and leverage is the difference between optimal ownership and projected ownership.

To run sims, implement something like below:
df returns a dataframe of each player and the number of times they were in the top-scoring lineup. lineups returns a raw list of each optimal lineup, which can be used to determine which players tend to group together.

```
import Optimizer as opt
df = pd.read_excel('NFL DK Projections.xlsx') #or wherever your data is stored
count = 100
ceil_column = 'ceil' #from your data
floor_column = 'floor' #from your data
fpts_col_name = 'fpts' #from your data
sim_class = opt.nfl(df = df)
df, lineups = sim_class.standard_sims(df,
 count,
 fpts_col_name = 'fpts'
 ceil_column = ceil_column,
 floor_column = floor_column
)
```
