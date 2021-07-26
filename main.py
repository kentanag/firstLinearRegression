import numpy as np
import matplotlib.pyplot as plt


def compute_error(b, m, points):

    # initialize at zero
    total_error = 0
    for i in range(0, len(points)):
        # get the x value
        x = points[i, 0]
        # get the y value
        y = points[i, 1]
        # get the difference, square it, add it ot the total
        total_error += (y - (m * x + b)) ** 2

    return total_error / float(len(points))


def step_gradient(b_current, m_current, points, learning_rate):

    # starting points for the gradients
    b_gradient = 0
    m_gradient = 0

    n = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # direction with respect to b and m
        # computing partial derivative
        b_gradient += -(2/n) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/n) * x * (y - ((m_current * x) + b_current))

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return new_b, new_m


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):

    b = starting_b
    m = starting_m

    # gradient descent
    for i in range(num_iterations):
        b, m = step_gradient(b, m, points, learning_rate)
        print(f'updated b: {b}, m: {m}')
    return b, m


def run():

    # step 1: collect data
    points = np.genfromtxt('wwtwoweather/min_max_temp.csv', delimiter=',')
    # grade_data.csv
    points = np.array(points)
    print(points)

    # step 2: define hyperparameters
    learning_rate = 0.0001  # how fast should our model converge?
    # y = mx + b (slope formula)
    initial_b = 0
    initial_m = 0
    num_iterations = 100

    # step 3: train the model
    print(f'starting gradient descent at b = {initial_b}, m = {initial_m}, '
          f'error = {compute_error(initial_b, initial_m, points)}')
    b, m = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    print(f'ending point at b = {b}, m = {m}, error = {compute_error(b, m, points)}')

    # data for plotting points
    x_ar = points[:, 0]
    y_ar = points[:, 1]
    plt.plot(x_ar, y_ar, 'o')

    # plot the best fitted line
    plt.plot(x_ar, m*x_ar + b)
    plt.show()


if __name__ == '__main__':
    run()
