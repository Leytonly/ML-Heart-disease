from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle

def save_model(model,filename):
    pickle.dump(model, open(filename, 'wb'))

# Load Dataset

path = 'heart.csv'
dataset = read_csv(path, delimiter=',')

array = dataset.values
X = array[:,1:13]
y = array[:,13]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=2, shuffle=True)

models = []

models.append(('LDA', LinearDiscriminantAnalysis()))

model = LinearDiscriminantAnalysis(shrinkage=0.0, solver='lsqr')

model.fit(X_train, Y_train)
training = model.predict(X_validation)

# save model
save_model(model, 'model.pkl')