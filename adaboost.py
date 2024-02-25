import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# Function to load the data and preprocess it
def load_and_preprocess_data(file_path):
    # Assuming the data is in a CSV format and the last column is the label
    data = pd.read_csv(file_path, sep=" ", header=None)
    # Last column is the label, the rest are features
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    return X, y


# Function to define the set of all lines formed by pairs of points
def define_lines(S):
    # print("creating lines")
    lines = []
    for (x1, y1), (x2, y2) in combinations(S, 2):
        # Calculate the coefficients of the line
        if x2 - x1 != 0:
            m = ((y2 - y1) / (x2 - x1))
        else:
            m = 0
        b = ((y1 - m * x1))
        # add both lines
        lines.append([m, b, -1])
        lines.append([m, b, 1])
    return lines

# Function to define the set of all circles formed by pairs of points
def define_circle(S):
    circles = []
    for (x1, y1), (x2, y2) in combinations(S, 2):
        # calc the radius
        r = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        # add both circles
        circles.append([x1, y1, r, -1])
        circles.append([x1, y1, r, 1])
    return circles

# Function to compute the margin of a point from a line
def margin_from_line(point, line):
    x, y = point
    m, b, sign = line
    return sign * (y - (m * x + b))

# Function to compute the margin of a point from a circle
def margin_from_circle(point, circle):
    x2, y2 = point
    x, y, r, sign = circle
    dist = np.sqrt((x2 - x) ** 2 + (y2 - y) ** 2)
    return sign * (r-dist)


# add 2 list together spotwise
def add_lists(l1, l2):
    l = []
    for i in range(len(l1)):
        l.append(l1[i] + l2[i])
    return l


# Adaboost step to select the best line based on weights
def adaboost_select_line(S, y, D, lines):
    best_line = None
    min_error = float('inf')
    for line in lines:
        # Calculate weighted error for this line
        predictions = np.sign([margin_from_line(point, line) for point in S])  # pos=above, neg=below, 0=on
        weighted_error = np.sum(D[predictions != y])
        # Update best line if this one is better
        if weighted_error < min_error:
            min_error = weighted_error
            best_line = line
    return best_line, min_error

# Adaboost step to select the best circle based on weights
def adaboost_select_circle(S, y, D, circles):
    best_circle = None
    min_error = float('inf')
    for circle in circles:
        # Calculate weighted error for this circle
        predictions = np.sign([margin_from_circle(point, circle) for point in S])  # pos=above, neg=below, 0=on
        weighted_error = np.sum(D[predictions != y])
        # Update best circle if this one is better
        if weighted_error < min_error:
            min_error = weighted_error
            best_circle = circle
    return best_circle, min_error


# Adaboost algorithm to find the best set of lines and their weights
def adaboost_line(S, y, lines, T, y_test, num_iter=8):

    # Initialize weights
    D = np.full(len(S), 1 / len(S))
    # List to store classifiers and their weights
    classifiers = []
    alphas = []
    # Empirical and true error lists
    empirical_errors = []
    true_errors = []

    for k in range(num_iter):
        # Select the best line
        best_line, error = adaboost_select_line(S, y, D, lines)
        # Calculate alpha
        alpha = 0.5 * np.log((1 - error) / error)
        # Save the classifier and its weight
        classifiers.append(best_line)
        alphas.append(alpha)

        #calculte the errors
        pred = [0] * len(S)
        for j in range(len(alphas)):
            predictions = ([alphas[j] * margin_from_line(point, classifiers[j]) for point in S])
            pred = add_lists(pred, predictions)
        predictions = np.sign(pred)
        error_train = np.sum([predictions != y])/len(S)

        pred_test = [0] * len(T)
        for j in range(len(alphas)):
            predictions_test = ([alphas[j] * margin_from_line(point, classifiers[j]) for point in T])
            pred_test = add_lists(pred_test, predictions_test)
        predictions_test = np.sign(pred_test)
        error_test = np.sum([predictions_test != y_test])/len(T)
        empirical_errors.append(error_train)
        true_errors.append(error_test)

        # Update weights
        D *= np.exp(-alpha * y * predictions)
        D /= np.sum(D)  # Normalize

    return classifiers, alphas, empirical_errors, true_errors


# run adaboost X amount of times line
def execute_adaboost_runs_line(file_path, num_runs=50):

    # Load and preprocess data
    X, y = load_and_preprocess_data(file_path)
    avg_empirical_errors = np.zeros(8)
    avg_true_errors = np.zeros(8)
    avg = np.ones(8)*num_runs
    # Perform multiple runs of Adaboost
    for i in range(num_runs):
        # Split the data
        S, T, y_S, y_T = train_test_split(X, y, test_size=0.5, random_state=i)

        # Define the lines from training set
        lines = define_lines(S)
        # Run Adaboost
        classifiers, alphas, empirical_errors, true_errors = adaboost_line(S, y_S, lines, T, y_T, num_iter=8)
        avg_empirical_errors +=empirical_errors
        avg_true_errors +=true_errors

    print("************")
    print("avg empirical_error: ",avg_empirical_errors/avg)
    print("avg true errors: ",avg_true_errors/avg)


# run adaboost X amount of times line
def adaboost_circle(S, y, lines, T, y_test, num_iter=8):

    # Initialize weights
    D = np.full(len(S), 1 / len(S))

    # List to store classifiers and their weights
    classifiers = []
    alphas = []
    # Empirical and true error lists
    empirical_errors = []
    true_errors = []

    for k in range(num_iter):
        # Select the best line
        best_circle, error = adaboost_select_circle(S, y, D, lines)
        # Calculate alpha
        alpha = 0.5 * np.log((1 - error) / error)
        # Save the classifier and its weight
        classifiers.append(best_circle)
        alphas.append(alpha)

        # calvulate errors
        pred = [0] * len(S)
        for j in range(len(alphas)):
            predictions = ([alphas[j] * margin_from_circle(point, classifiers[j]) for point in S])
            pred = add_lists(pred, predictions)
        predictions = np.sign(pred)
        error_train = np.sum([predictions != y]) /len(S)

        pred_test = [0] * len(T)
        for j in range(len(alphas)):
            predictions_test = ([alphas[j] * margin_from_circle(point, classifiers[j]) for point in T])
            pred_test = add_lists(pred_test, predictions_test)
        predictions_test = np.sign(pred_test)
        error_test = np.sum([predictions_test != y_test])/len(T)

        empirical_errors.append(error_train)
        true_errors.append(error_test)

        # Update weights
        D *= np.exp(-alpha * y * predictions)
        D /= np.sum(D)  # Normalize

    return classifiers, alphas, empirical_errors, true_errors

# run adaboost X amount of times circle
def execute_adaboost_runs_circle(file_path, num_runs=50):

    # Load and preprocess data
    X, y = load_and_preprocess_data(file_path)
    avg_empirical_errors = np.zeros(8)
    avg_true_errors = np.zeros(8)
    avg = np.ones(8) * num_runs
    # Perform multiple runs of Adaboost
    for i in range(num_runs):
        # Split the data
        S, T, y_S, y_T = train_test_split(X, y, test_size=0.5, random_state=i)

        # Define the lines from training set
        circles = define_circle(S)
        # Run Adaboost
        classifiers, alphas, empirical_errors, true_errors = adaboost_circle(S, y_S, circles, T, y_T, num_iter=8)
        avg_empirical_errors += empirical_errors
        avg_true_errors += true_errors

    print("************")
    print("avg empirical_error: ", avg_empirical_errors / avg)
    print("avg true errors: ", avg_true_errors / avg)


def main():
    execute_adaboost_runs_line('circle_separator.txt',50)
    execute_adaboost_runs_circle('circle_separator.txt',50)

if __name__ == '__main__':
    main()