import pandas as pd 

df = pd.read_csv('dataset.csv')

avg_weekly_strain = df.groupby('player_id')['minutes_last_week'].mean()
avg_weekly_strain = avg_weekly_strain.to_frame()
avg_weekly_strain = avg_weekly_strain.rename(columns={"minutes_last_week": "avg_weekly_strain"})
df = pd.merge(df, avg_weekly_strain, on='player_id')

avg_weekly_sprints = df.groupby('player_id')['sprints_last_week'].mean()
avg_weekly_sprints = avg_weekly_sprints.to_frame()
avg_weekly_sprints = avg_weekly_sprints.rename(columns={"sprints_last_week": "avg_weekly_sprints"})
df = pd.merge(df, avg_weekly_sprints, on='player_id')

df = df.groupby('player_id').tail(1)

print(df.head())
