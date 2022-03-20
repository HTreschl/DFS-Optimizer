# DFS Optimizer

A library designed to run sims and optimize lineups for DFS using python's pulp module. Includes NFL and MLB.

# NFL Example -- Single Optimized Lineup for a Classic Game-Type
Optimize a single lineup without running any sims using the example projections for a classic dfs game

Import the data and initialize the NFL optimizer

```
df = pd.read_excel('NFL DK Projections.xlsx')
model = opt.NFL(df=df)
```

Optimize using the projection set we want, in this case averages of all available projections

```
lineup = model.standard_optimizer(df, objective_fn_column='avg fpts')
print(lineup)
```
