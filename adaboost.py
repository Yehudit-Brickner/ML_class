import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# Function to load the data and preprocess it
def load_and_preprocess_data(file_path):
    print("loading df")
    # Assuming the data is in a CSV format and the last column is the label
    data = pd.read_csv(file_path, sep=" ", header=None)
    # Last column is the label, the rest are features
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # xx1 = []
    # yy1 = []
    # size = []
    # for j in X:
    #     xx1.append(j[0])
    #     yy1.append(j[1])
    #     size.append(60)
    # plt.figure(figsize=(5, 5))
    # plt.scatter(xx1, yy1, c=color_list(y), s=size)
    # t = "real data "
    # plt.title(t)
    # plt.show()

    return X, y


# Function to define the set of all lines formed by pairs of points
def define_lines(S):
    # print("creating lines")
    lines = []
    for (x1, y1), (x2, y2) in combinations(S, 2):
        # Calculate the coefficients of the line
        # A = y2 - y1
        # B = x1 - x2
        # C = x2 * y1 - x1 * y2
        # lines.append((A, B, C))
        if x2 - x1 != 0:
            m = ((y2 - y1) / (x2 - x1))
        else:
            m = 0
        b = ((y1 - m * x1))
        # sign=1
        # lines.append([m,b,sign])
        lines.append([m, b, -1])
        lines.append([m, b, 1])
    return lines

def define_circle(S):
    circles = []
    for (x1, y1), (x2, y2) in combinations(S, 2):
        r = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        circles.append([x1, y1, r, -1])
        circles.append([x1, y1, r, 1])
    return circles

# Function to compute the margin of a point from a line
def margin_from_line(point, line):
    # print("calc margin from line")
    x, y = point
    m, b, sign = line

    return sign * (y - (m * x + b))
    # A, B, C = line
    # return A * x + B * y + C


def margin_from_circle(point, circle):
    x2, y2 = point
    x, y, r, sign = circle
    dist = np.sqrt((x2 - x) ** 2 + (y2 - y) ** 2)
    return sign * (r-dist)


def color_list(l):
    color_list=[]
    for i in l:
        if i>0:
            color_list.append("red")
        elif i<0:
            color_list.append("blue")
        else:
            color_list.append("yellow")
    return color_list


def compare(l1, l2):
    count_correct = 0
    count_wrong = 0
    count_zero = 0
    for i in range(len(l1)):
        if l2[i] == 0:
            count_zero += 1
        if l1[i] == l2[i]:
            count_correct += 1
        else:
            count_wrong += 1
    return count_correct, count_zero, count_wrong


def change(l1):
    l = []
    for i in l1:
        l.append(i * -1)
    return l


def add_lists(l1, l2):
    l = []
    for i in range(len(l1)):
        l.append(l1[i] + l2[i])
    return l


def calc_error(D, predictions, y):
    sum = 0
    for i in range(len(D)):
        if predictions[i] != y[i]:
            sum += 1 * D[i]
    return sum


def show_graphs1(S, predictions,best_line,k,alphas,classifiers,y):
    xx1 = []
    yy1 = []
    size = []
    for j in S:
        xx1.append(j[0])
        yy1.append(j[1])
        size.append(60)

    plt.figure(figsize=(5, 5))
    plt.scatter(xx1, yy1, c=color_list(predictions), s=size)

    y1 = 0 * best_line[0] + best_line[1]
    y2 = 1 * best_line[0] + best_line[1]
    plt.axline((0, y1), (1, y2), linewidth=1, color='black')
    t = "best line iter "
    t += str(k)
    plt.title(t)
    plt.show()

    pred = [0] * len(S)
    for j in range(len(alphas)):
        # print("weight: ", alphas[j])
        # print(classifiers[j])
        predictions = ([alphas[j] * margin_from_line(point, classifiers[j]) for point in S])
        # print("pridictions: ", j)
        # print(predictions)
        pred = add_lists(pred, predictions)
        # print("pred:")
        # print(pred)
    final_prediction = np.sign(pred)
    # print("final prediction")
    # print(final_prediction)
    correct, zero, wrong = compare(y, final_prediction)
    # print("correct: ", correct, "zero: ", zero, "wrong: ", wrong)

    xx1 = []
    yy1 = []
    size = []
    for j in S:
        xx1.append(j[0])
        yy1.append(j[1])
        size.append(60)
    plt.figure(figsize=(5, 5))
    plt.scatter(xx1, yy1, c=color_list(predictions), s=size)
    plt.scatter(xx1, yy1, c=color_list(y), marker='*')
    for j in classifiers:
        y1 = 0 * j[0] + j[1]
        y2 = 1 * j[0] + j[1]
        plt.axline((0, y1), (1, y2), linewidth=1, color='black')
    t = "traing data on "
    t += str(k + 1)
    t += " lines"
    plt.title(t)
    plt.show()


def show_graphs2(S,alphas,classifiers,y_S,i,T,y_T):
    pred = [0] * len(S)
    for j in range(len(alphas)):
        predictions = ([alphas[j] * margin_from_line(point, classifiers[j]) for point in S])
        pred = add_lists(pred, predictions)
    final_prediction = np.sign(pred)
    # print("final prediction: ", final_prediction)
    # correct, zero, wrong = compare(y_S, final_prediction)
    # print("TRAIN DATA- correct: ", correct, "zero: ", zero, "wrong: ", wrong)
    # xx1 = []
    # yy1 = []
    # for j in S:
    #     xx1.append(j[0])
    #     yy1.append(j[1])
    # plt.figure(figsize=(8, 8))
    # plt.scatter(xx1, yy1, c=color_list(predictions))
    # plt.scatter(xx1, yy1, c=color_list(y_S), marker='*')
    # for j in classifiers:
    #     y1 = 0 * j[0] + j[1]
    #     y2 = 1 * j[0] + j[1]
    #     plt.axline((0, y1), (1, y2), linewidth=1, color='black')
    # t = "train fig "
    # t += str(i)
    # plt.title(t)
    # plt.show()

    pred = [0] * len(T)
    for j in range(len(alphas)):
        predictions = ([alphas[j] * margin_from_line(point, classifiers[j]) for point in T])
        pred = add_lists(pred, predictions)
        # print(pred)
    final_prediction = np.sign(pred)
    # print(final_prediction)
    # correct, zero, wrong = compare(y_T, final_prediction)
    # print("TEST DATA- correct: ", correct, "zero: ", zero, "wrong: ", wrong)
    # xx2 = []
    # yy2 = []
    # for j in T:
    #     xx2.append(j[0])
    #     yy2.append(j[1])
    # plt.figure(figsize=(8, 8))
    # plt.scatter(xx2, yy2, c=color_list(final_prediction))
    # plt.scatter(xx2, yy2, c=color_list(y_T), marker='*')
    # for j in classifiers:
    #     y1 = 0 * j[0] + j[1]
    #     y2 = 1 * j[0] + j[1]
    #     plt.axline((0, y1), (1, y2), linewidth=1, color='black')
    # t = "test fig "
    # t += str(i)
    # plt.title(t)
    # plt.show()


def show_graphs3(S, predictions,best_circle,k,alphas,classifiers,y_test):

    xx1 = []
    yy1 = []
    size = []
    for j in S:
        xx1.append(j[0])
        yy1.append(j[1])
        size.append(60)
    figure, axes = plt.subplots()
    x,y,r,sign =best_circle
    Drawing_uncolored_circle = plt.Circle((x, y),r, fill=False)
    axes.add_artist(Drawing_uncolored_circle)
    axes.scatter(xx1, yy1, c=color_list(predictions), s=size)

    t = "best circle iter "
    t += str(k)
    plt.title(t)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.show()

    pred = [0] * len(S)
    for j in range(len(alphas)):
        predictions = ([alphas[j] * margin_from_circle(point, classifiers[j]) for point in S])
        pred = add_lists(pred, predictions)
    final_prediction = np.sign(pred)
    # correct, zero, wrong = compare(y, final_prediction)
    # print("correct: ", correct, "zero: ", zero, "wrong: ", wrong)

    xx1 = []
    yy1 = []
    size = []
    for j in S:
        xx1.append(j[0])
        yy1.append(j[1])
        size.append(60)

    figure, axes = plt.subplots()
    for j in classifiers:
        x, y, r, sign = j
        Drawing_uncolored_circle = plt.Circle((x, y), r, fill=False)
        axes.add_artist(Drawing_uncolored_circle)
    axes.scatter(xx1, yy1, c=color_list(final_prediction), s=size)

    axes.scatter(xx1, yy1, c=color_list(y_test), marker='*')

    t = "traing data on "
    t += str(k + 1)
    t += " circles"
    plt.title(t)
    plt.xlim(-2.5,2.5)
    plt.ylim(-2.5,2.5)
    plt.show()


def show_graphs4(S,alphas,classifiers,y_S,i,T,y_T):
    pred = [0] * len(S)
    for j in range(len(alphas)):
        predictions = ([alphas[j] * margin_from_circle(point, classifiers[j]) for point in S])
        pred = add_lists(pred, predictions)
    final_prediction = np.sign(pred)
    # print("final prediction: ", final_prediction)
    # correct, zero, wrong = compare(y_S, final_prediction)
    # print("TRAIN DATA- correct: ", correct, "zero: ", zero, "wrong: ", wrong)
    # xx1 = []
    # yy1 = []
    # for j in S:
    #     xx1.append(j[0])
    #     yy1.append(j[1])
    # plt.figure(figsize=(8, 8))
    # plt.scatter(xx1, yy1, c=color_list(predictions))
    # plt.scatter(xx1, yy1, c=color_list(y_S), marker='*')
    # t = "train fig "
    # t += str(i)
    # plt.title(t)
    # plt.show()

    pred = [0] * len(T)
    for j in range(len(alphas)):
        predictions = ([alphas[j] * margin_from_circle(point, classifiers[j]) for point in T])
        pred = add_lists(pred, predictions)
        # print(pred)
    final_prediction = np.sign(pred)
    # print(final_prediction)
    # correct, zero, wrong = compare(y_T, final_prediction)
    # print("TEST DATA- correct: ", correct, "zero: ", zero, "wrong: ", wrong)
    # xx2 = []
    # yy2 = []
    # for j in T:
    #     xx2.append(j[0])
    #     yy2.append(j[1])
    # plt.figure(figsize=(8, 8))
    # plt.scatter(xx2, yy2, c=color_list(final_prediction))
    # plt.scatter(xx2, yy2, c=color_list(y_T), marker='*')
    # t = "test fig "
    # t += str(i)
    # plt.title(t)
    # plt.show()



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

def adaboost_select_circle(S, y, D, circles):
    best_circle = None
    min_error = float('inf')
    for circle in circles:
        # Calculate weighted error for this line
        predictions = np.sign([margin_from_circle(point, circle) for point in S])  # pos=above, neg=below, 0=on
        # correct,zero,wrong = compare(y,predictions)
        # print("correct: ",correct,"zero: ",zero, "wrong: ",wrong )
        # if wrong > correct:
        #     predictions = change(predictions)
        #     line[2]=line[2]*-1
        weighted_error = np.sum(D[predictions != y])
        # Update best line if this one is better
        if weighted_error < min_error:
            min_error = weighted_error
            best_circle = circle
    return best_circle, min_error


# Adaboost algorithm to find the best set of lines and their weights
def adaboost_line(S, y, lines, T, y_test, num_iter=8):
    # print("run adaboost line")
    # Initialize weights
    D = np.full(len(S), 1 / len(S))
    D_test = np.full(len(T), 1 / len(T))
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
        # Update weights
        predictions = np.sign([margin_from_line(point, best_line) for point in S])
        # correct, zero, wrong = compare(y, predictions)
        # print("correct: ", correct, "zero: ", zero, "wrong: ", wrong)

        error_train = np.sum(D[predictions != y])
        predictions_test = np.sign([margin_from_line(point, best_line) for point in T])
        error_test = np.sum(D_test[predictions_test != y_test])
        empirical_errors.append(error_train)
        true_errors.append(error_test)


        D *= np.exp(-alpha * y * predictions)
        D /= np.sum(D)  # Normalize

        # Save the classifier and its weight
        classifiers.append(best_line)
        alphas.append(alpha)
        # Calculate empirical and true error
        # H_x = np.sign(sum(alpha * np.sign(margin_from_line(point, line))
        #                   for alpha, line in zip(alphas, classifiers) for point in S))
        # empirical_error = np.mean(y != H_x)
        # H_x_test = np.sign(sum(alpha * np.sign(margin_from_line(point, line))
        #                        for alpha, line in zip(alphas, classifiers) for point in T))
        # true_error = np.mean(y_test != H_x_test)
        # empirical_errors.append(empirical_error)
        # true_errors.append(true_error)





        # show_graphs1(S, predictions, best_line, k, alphas, classifiers, y)
    return classifiers, alphas, empirical_errors, true_errors


# Placeholder for the final Adaboost execution function (to be completed)
def execute_adaboost_runs_line(file_path, num_runs=50):
    # print("execute line")
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
        # print(len(avg_empirical_errors),len(avg_true_errors))
        # print (classifiers)
        # print (alphas)
        # print("empirical_errors : ",empirical_errors )
        # print("true_errors : ",true_errors )

        # get final prediction using all rules
        # show_graphs2(S,alphas,classifiers,y_S,i,T,y_T)
    print("************")
    print("avg empirical_error: ",avg_empirical_errors/avg)
    print("avg true errors: ",true_errors/avg)

def adaboost_circle(S, y, lines, T, y_test, num_iter=8):
    # print("run adaboost circle")
    # Initialize weights
    D = np.full(len(S), 1 / len(S))
    D_test = np.full(len(T), 1 / len(T))
    # List to store classifiers and their weights
    classifiers = []
    alphas = []
    # Empirical and true error lists
    empirical_errors = []
    true_errors = []

    for k in range(num_iter):
        # Select the best line
        best_circle, error = adaboost_select_circle(S, y, D, lines)
        # print(best_line,error)
        # Calculate alpha
        alpha = 0.5 * np.log((1 - error) / error)
        # Update weights
        predictions = np.sign([margin_from_circle(point, best_circle) for point in S])
        # correct, zero, wrong = compare(y, predictions)
        # print("correct: ", correct, "zero: ", zero, "wrong: ", wrong)

        error_train = np.sum(D[predictions != y])
        predictions_test = np.sign([margin_from_circle(point, best_circle) for point in T])
        error_test = np.sum(D_test[predictions_test != y_test])
        empirical_errors.append(error_train)
        true_errors.append(error_test)

        D *= np.exp(-alpha * y * predictions)
        D /= np.sum(D)  # Normalize

        # Save the classifier and its weight
        classifiers.append(best_circle)
        alphas.append(alpha)
        # Calculate empirical and true error
        # H_x = np.sign(sum(alpha * np.sign(margin_from_circle(point, circle))
        #                   for alpha, circle in zip(alphas, classifiers) for point in S))
        # empirical_error = np.mean(y != H_x)
        # H_x_test = np.sign(sum(alpha * np.sign(margin_from_circle(point, circle))
        #                        for alpha, circle in zip(alphas, classifiers) for point in T))
        # true_error = np.mean(y_test != H_x_test)
        # empirical_errors.append(empirical_error)
        # true_errors.append(true_error)
        # print("empirical_error: ",empirical_error)
        # print("true_error: ", true_error)
        # show_graphs3(S, predictions, best_circle, k, alphas, classifiers, y)



    return classifiers, alphas, empirical_errors, true_errors


def execute_adaboost_runs_circle(file_path, num_runs=50):
    # print("execute circle")
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
        # print(len(avg_empirical_errors),len(avg_true_errors))
        # print (classifiers)
        # print (alphas)
        # print("empirical_errors : ",empirical_errors )
        # print("true_errors : ",true_errors )

        # get final prediction using all rules
        # show_graphs4(S, alphas, classifiers, y_S, i, T, y_T)
    print("************")
    print("avg empirical_error: ", avg_empirical_errors / avg)
    print("avg true errors: ", true_errors / avg)


def main():
    execute_adaboost_runs_line('circle_separator.txt',10)
    execute_adaboost_runs_circle('circle_separator.txt',10)

if __name__ == '__main__':
    main()