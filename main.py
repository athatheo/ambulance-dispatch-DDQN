import Environment



env = Environment.Environment()
env.import_data()
print(env.nr_ambulances)
max_key = max(env.nr_ambulances, key= lambda x: env.nr_ambulances[x])
print(env.nr_ambulances[max_key])


print("00000000000000000000000000000")

