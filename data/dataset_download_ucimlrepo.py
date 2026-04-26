from ucimlrepo import fetch_ucirepo
import pandas as pd

# fetch dataset
default_of_credit_card_clients = fetch_ucirepo(id=350)

# data
X = default_of_credit_card_clients.data.features
y = default_of_credit_card_clients.data.targets

# combine features and target
df = pd.concat([X, y], axis=1)

# save to CSV
df.to_csv("default_of_credit_card_clients.csv", index=False)

print("Dataset saved as default_of_credit_card_clients.csv")
