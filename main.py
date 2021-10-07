import Environment



env = Environment.Environment()
env.import_data()
print(env.nr_ambulances)
max_key = max(env.nr_ambulances, key= lambda x: env.nr_ambulances[x])
print(env.nr_ambulances[max_key])


print("00000000000000000000000000000")
print(env.nr_ambulances[1])
print(env.bases)
print(env.postcode_dic[1])
print(env.prob_acc[1])