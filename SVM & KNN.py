##### 1a #####
# DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH DATA GENERATION,
# MAKE A COPY OF THIS FUNCTION AND THEN EDIT
#
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier


def generate_data(n_samples, tst_frac=0.2, val_frac=0.2):
    # Generate a non-linear data set
    X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=42)
    
    # Take a small subset of the data and make it VERY noisy; that is, generate outliers
    m = 30
    np.random.seed(30) # Deliberately use a different seed
    ind = np.random.permutation(n_samples)[:m]
    X[ind, :] += np.random.multivariate_normal([0, 0], np.eye(2), (m, ))
    y[ind] = 1 - y[ind]

    # Plot this data
    cmap = ListedColormap(['#b30065', '#178000'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')

    # First, we use train_test_split to partition (X, y) into training and test sets
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)

    # Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
    X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)
    
    return (X_trn, y_trn), (X_val, y_val), (X_tst, y_tst)

#
# DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH VISUALIZATION,
# MAKE A COPY OF THIS FUNCTION AND THEN EDIT
#
def visualize(models, param, X, y):
    # Initialize plotting
    if len(models) % 3 == 0:
        nrows = len(models) // 3
    else:
        nrows = len(models) // 3 + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5.0 * nrows))
    cmap = ListedColormap(['#b30065', '#178000'])

    # Create a mesh
    xMin, xMax = X[:, 0].min() - 1, X[:, 0].max() + 1
    yMin, yMax = X[:, 1].min() - 1, X[:, 1].max() + 1
    xMesh, yMesh = np.meshgrid(np.arange(xMin, xMax, 0.01),
                            np.arange(yMin, yMax, 0.01))

    for i, (p, clf) in enumerate(models.items()):
        # if i > 0:
        # break
        r, c = np.divmod(i, 3)
        ax = axes[r, c]

        # Plot contours
        zMesh = clf.decision_function(np.c_[xMesh.ravel(), yMesh.ravel()])
        zMesh = zMesh.reshape(xMesh.shape)
        ax.contourf(xMesh, yMesh, zMesh, cmap=plt.cm.PiYG, alpha=0.6)

        if (param == 'C' and p > 0.0) or (param == 'gamma'):
            ax.contour(xMesh, yMesh, zMesh, colors='k', levels=[-1, 0, 1],
                        alpha=0.5, linestyles=['--', '-', '--'])

        # Plot data
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')
        ax.set_title('{0} = {1}'.format(param, p))

# Generate the data
n_samples = 300 # Total size of data set
(X_trn, y_trn), (X_val, y_val), (X_tst, y_tst) = generate_data(n_samples)

# Learn support vector classifiers with a radial-basis function kernel with
# fixed gamma = 1 / (n_features * X.std()) and different values of C
C_range = np.arange(-3.0, 6.0, 1.0)
C_values = np.power(10.0, C_range)

models = dict()
trnErr = dict()
valErr = dict()
    
for C in C_values:
    models[C] = SVC(kernel='rbf', probability=True, C=C, gamma='scale')
    fitted = models[C].fit(X_trn, y_trn)
    trnErr[np.log10(C)] = 1-models[C].score(X_trn, y_trn)
    valErr[np.log10(C)] = 1-models[C].score(X_val, y_val)

plt.figure()
plt.plot(trnErr.keys(), trnErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(valErr.keys(), valErr.values(), marker='s', linewidth=3, markersize=12)
plt.title('SVM error rates')
plt.xlabel('C')
plt.ylabel('Error Rates')
plt.legend(['Training Error', 'Validation Error'], fontsize=16)
    
visualize(models, 'C', X_trn, y_trn)

C_best = min(valErr, key=valErr.get)
C_tstAccuracy = models[10**C_best].score(X_tst, y_tst)
print('##### 1(a) #####')
print('SVC Test Accuracy with the best C value =', 10**C_best, 'is', C_tstAccuracy)
print(' ')

##### 1b #####
# Learn support vector classifiers with a radial-basis function kernel with
# fixed C = 10.0 and different values of gamma
gamma_range = np.arange(-2.0, 4.0, 1.0)
gamma_values = np.power(10.0, gamma_range)

models = dict()
trnErr = dict()
valErr = dict()

for G in gamma_values:
    models[G] = SVC(kernel='rbf', probability=True, C=10, gamma=G)
    fitted = models[G].fit(X_trn, y_trn)
    trnErr[np.log10(G)] = 1-models[G].score(X_trn, y_trn)
    valErr[np.log10(G)] = 1-models[G].score(X_val, y_val)
    
plt.figure()
plt.plot(trnErr.keys(), trnErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(valErr.keys(), valErr.values(), marker='s', linewidth=3, markersize=12)
plt.title('SVM error rates')
plt.xlabel('gamma')
plt.ylabel('Error Rates')
plt.legend(['Training Error', 'Validation Error'], fontsize=16)
  
visualize(models, 'gamma', X_trn, y_trn)

G_best = min(valErr, key=valErr.get)
G_tstAccuracy = models[10**G_best].score(X_tst, y_tst)
print('##### 1(b) #####')
print('SVC Test Accuracy with the best gamma value =', 10**G_best, 'is', G_tstAccuracy)
print(' ')
##### 2 #####
# Load the Breast Cancer Diagnosis data set; download the files from eLearning
# CSV files can be read easily using np.loadtxt()
trn = np.loadtxt('wdbc_trn.csv', delimiter=',')
val = np.loadtxt('wdbc_val.csv', delimiter=',')
tst = np.loadtxt('wdbc_tst.csv', delimiter=',')

# Split X and y for each datasets
X_trn = trn[:,1:]
y_trn = trn[:,0]

X_val = val[:,1:]
y_val = val[:,0]

X_tst = tst[:,1:]
y_tst = tst[:,0]

C_range = np.arange(-2.0, 5.0, 1.0)
C_values = np.power(10.0, C_range)
gamma_range = np.arange(-3.0, 3.0, 1.0)
gamma_values = np.power(10.0, gamma_range)

n_models = dict()
n_trnErr = dict()
n_valErr = dict()
    
for C in C_values:
    for G in gamma_values:
        n_models[(C,G)] = SVC(kernel='rbf', probability=True, C=C, gamma=G)
        fitted = n_models[C,G].fit(X_trn, y_trn)
        n_trnErr[(C,G)] = 1-n_models[C,G].score(X_trn, y_trn)
        n_valErr[(C,G)] = 1-n_models[C,G].score(X_val, y_val)

print('##### 2 #####')
print('Training errors for different c and gamma values')
print('(C,gamma)         Training Error rate')
for key, value in n_trnErr.items():
    print(key, ' : ', value)
print(' ')

print('Validation errors for different c and gamma values')
print('(C,gamma)         Validation Error rate')
for key, value in n_valErr.items():
    print(key, ' : ', value)     
print(' ')
n_CG_best = min(n_valErr, key=n_valErr.get)
n_C_tstAccuracy = n_models[n_CG_best].score(X_tst, y_tst)

print('SVC Test Accuracy with the best C and G values =', n_CG_best, 'is', n_C_tstAccuracy)
print(' ')

##### 3 #####
trnErr = []
valErr = []

k_values = [1, 5, 11, 15, 21]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, algorithm = 'kd_tree')
    knn.fit(X_trn, y_trn)
    pred_trn = knn.predict(X_trn)
    pred_val = knn.predict(X_val)
    trnErr.append(np.mean(pred_trn != y_trn))
    valErr.append(np.mean(pred_val != y_val))

plt.figure()
plt.plot([1, 5, 11, 15, 21], trnErr, marker='o', linewidth=3, markersize=12)
plt.plot([1, 5, 11, 15, 21], valErr, marker='s', linewidth=3, markersize=12)
plt.title('KNN error rates vs. k value')
plt.xlabel('K')
plt.ylabel('Error Rates')
plt.legend(['Training Error', 'Validation Error'], fontsize=16)

print('##### 3 #####')
print('The minimum error rate with k =', k_values[valErr.index(min(valErr))], 'is', min(valErr))
print(' ')

knn.fit(X_tst, y_tst)
pred_tst = knn.predict(X_tst)
tstErr = np.mean(pred_tst != y_tst)
print('KNN Test Accuracy with best k = 5 is', 1-tstErr)








