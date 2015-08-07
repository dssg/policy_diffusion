import database
import random

random.seed(10)


states = open("/Users/mattburg/Dropbox/dssg/policy_diffusion/data/states.txt").readlines()
ec = database.ElasticConnection(host = "54.203.12.145",port = "9200")
out_file = open("/Users/mattburg/Dropbox/dssg/policy_diffusion/data/state_bill_samples.txt",'w')
state_samples = []
for state in states:
    state_bills = ec.get_bills_by_state(state.strip())
    random.shuffle(state_bills)
    for x in state_bills[0:50]:
        state_samples.append(x)
        out_file.write("{0}\n".format(x))
    


