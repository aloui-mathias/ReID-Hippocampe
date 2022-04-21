import json
from sklearn.model_selection import StratifiedKFold

train_set = json.load(open("trainIndivNbDates.json", "r"))

indivs = list(train_set.keys())
nb_dates = list(train_set.values())

KFold = StratifiedKFold(10)
folds_indices = KFold.split(indivs, nb_dates)

print(list(folds_indices))
