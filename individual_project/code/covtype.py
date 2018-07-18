# esempio qui: http://peekaboo-vision.blogspot.com/2012/12/kernel-approximations-for-efficient.html

# import functions
import encode_csv
from functions import * 
from encode_csv import * 

covtype = pd.read_csv('../datasets/covtype.data')

X = covtype.drop(['5'], axis=1) 
le = LabelEncoder().fit(covtype['5']) 
y = le.transform(covtype['5'])

# encode data
encoded, encoders = number_encode_features(X)

# scale columns between -1 and 1
X = scale_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

print("train:", len(X_train), ", test:", len(X_test))

# taking a subset

n = 20000 
X_train = X_train[:n]
y_train = y_train[:n]
X_test = X_test[:n//20]
y_test = y_test[:n//20]

covtype_fit = fit_all(X_train, y_train, X_test, y_test)

# covtype_fit = fit_all(X_train, y_train, X_test, y_test, gamma=0.031, C=100)

save(covtype_fit, 'covtype_fit_20000')

