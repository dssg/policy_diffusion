import pandas as pd

def func(x):
   x['weight'] = x.count()
   return x
df = pd.read_csv("/Users/mattburg/Downloads/interest_groups_to_state_network_fixed.csv")
df = df[df.score>100]
df = df.groupby(df.edge_id).count()

alec_total = 2208.
alice_total = 1500.

index = df.index
ids = df['lobby_id'].tolist()

print "Source,Target,Weight,Type"
for x,y in zip(index,ids):
    s,t = x.split("_")
    if s == "alec":
        y = float(y)/alec_total
    elif s == "alice":
        y = float(y)/alice_total
    else:
        continue
    print "{0},{1},{2},{3}".format(s,t,y,"undirected")

