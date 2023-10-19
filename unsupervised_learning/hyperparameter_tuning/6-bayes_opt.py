import GPy
import GPyOpt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from GPyOpt.methods import BayesianOptimization

# loads the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# defines the SVM classifier and hyperparameter space to optimize
def svm_classifier(C, gamma):
    clf = SVC(C=C, gamma=gamma, random_state=0)
    return np.mean(cross_val_score(clf, X, y, cv=5))

hyperparameter_space = [
    {'name': 'C', 'type': 'continuous', 'domain': (0.1, 10)},
    {'name': 'gamma', 'type': 'continuous', 'domain': (0.001, 1)}
]

# defines the objective function for Bayesian optimization
def objective_function(params):
    C = params[0]
    gamma = params[1]
    return -svm_classifier(C, gamma)  # Negative for maximization

# initialize bayesian optimization
opt = BayesianOptimization(f=objective_function, domain=hyperparameter_space)

# optimizes for a maximum of 30 iterations
opt.run_optimization(max_iter=30)

# gest the optimal hyperparameters and the corresponding value
optimal_params = opt.x_opt
optimal_value = -opt.fx_opt  # Convert back to positive for accuracy

# saves the optimal hyperparameters to a checkpoint file
checkpoint_filename = f'optimal_params_C_{optimal_params[0]}_gamma_{optimal_params[1]}.pkl'
with open(checkpoint_filename, 'wb') as file:
    pickle.dump(optimal_params, file)

# printss the optimal hyperparameters and value
print('Optimal Hyperparameters:')
print(f'C: {optimal_params[0]}, gamma: {optimal_params[1]}')
print(f'Optimal Value (Accuracy): {optimal_value:.4f}')

# saves a report of the optimization
report_filename = 'bayes_opt.txt'
with open(report_filename, 'w') as report_file:
    report_file.write(f'Optimal Hyperparameters: C={optimal_params[0]}, gamma={optimal_params[1]}\n')
    report_file.write(f'Optimal Value (Accuracy): {optimal_value:.4f}\n')

# lpots the convergence
opt.plot_convergence()
