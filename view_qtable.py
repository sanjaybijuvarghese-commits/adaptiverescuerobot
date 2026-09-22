import pickle

with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

print("Number of states:", len(q_table))

for state, actions in q_table.items():
    print("\nState:", state)
    print("Q-values:", actions)