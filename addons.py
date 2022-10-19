import random
import numpy as np
import matplotlib.pyplot as plt


# takes in the order of the equation, and then the arguments, C0, C1, C2... Cn , such as the equation is
# in the form : C0 + C1 X + C2 X^2 + .... + Cn X^n
# write the jitter in percent
def generate_fuzzy_data(coef_list: list, x_boundary: list, jitter=0, show_coeficients=False):
    data = []
    abscisse = []
    order = len(coef_list)
    start = x_boundary[0]
    stop = x_boundary[1]

    number_of_points = 100

    if show_coeficients:
        if coef_list:
            print("Using coeficients:")
            for coef in coef_list:
                print(f"{coef}")

        else:
            print("using random coeficients")
            coef_list = [random.uniform(-1, 1) for _i in range(order)]

    order_list = list(range(order))

    for x in range(number_of_points):
        data.append(0)
        abscisse.append(start+x*(stop-start)/(number_of_points-1))
        for index in range(order):
            data[x] += coef_list[index] * pow(abscisse[x], order_list[index])

    span = (np.max(data) - np.min(data))

    return abscisse, list(map(lambda y: y + span * random.uniform(-jitter/100, jitter/100), data))


# Assuming that the equation can be written in the form f(x,y)=g(x)*h(y), with g(x) a polynomial fonction
# of order n, arguments C0, C1, C2... Cn , such as g(x) = C0 + C1 X + C2 X^2 + .... + Cn X^n and h(y) a polynomial
# fonction of order m, arguments D0, D1, D2, ... Dm, such as h(y) = D0 + D1 X + D2 X^2 + ... + Dm X^m
# write order as (n,m)
# write the boundaries parameters in the form of (start,stop)
# write the jitter in percent
# arguments are the coeficients, in the form of (C0, C1, ..., Cn) and (D0, D1, ..., Dm)
def generate_fuzzy_data_2d(order: tuple, boundaries_x: tuple, boundaries_y: tuple,  *args):
    size_x = 33
    size_y = 33
    data_g = [[] for _ in range(size_x)]
    data_h = [[] for _ in range(size_x)]
    data = [[] for _ in range(size_x)]
    coef_g = []
    coef_h = []
    abscisse_x = []
    abscisse_y = []

    if args:
        if len(args) == 2:
            for i in range(len(args[0])):
                coef_g.append(args[0][i])
            for i in range(len(args[1])):
                coef_h.append(args[1][i])
        else:
            print("number of coefficients should be equal to the order of the equation + 1")
            print("For instance: for fitting second order equation, you need to have 3 coefficients")
    else:
        coef_g = [random.uniform(-1, 1) for _i in range(order[0])]
        coef_h = [random.uniform(-1, 1) for _i in range(order[1])]

    order_g_list = list(range(order[0]))
    order_h_list = list(range(order[1]))

    for x in range(size_x):
        abscisse_x.append(boundaries_x[0] + x * (boundaries_x[1] - boundaries_x[0]) / (size_x - 1))

    for y in range(size_y):
        abscisse_y.append(boundaries_y[0] + y * (boundaries_y[1] - boundaries_y[0]) / (size_y - 1))

    for x in range(size_x):
        for y in range(size_y):
            data_g[x].append(0)
            for index in order_g_list:
                data_g[x][y] += coef_g[index] * pow(abscisse_x[x], order_g_list[index])
            data_h[x].append(0)
            for index in order_h_list:
                data_h[x][y] += coef_h[index] * pow(abscisse_y[y], order_h_list[index])
            data[x].append(data_g[x][y] * data_h[x][y])

    # span = (np.max(data) - np.min(data))

    return abscisse_x, abscisse_y, data


def polynomial_grid(data_g, abscisse_x, coef_g, order_g_list):
    for x in range(len(data_g)):
        for y in range(len(data_g[x])):
            data_g[x].append(0)
            for index in order_g_list:
                data_g[x][y] += coef_g[index] * pow(abscisse_x[x], order_g_list[index])
    return data_g


def polynomial_test(coord, coef, order):
    data = 0
    for index in order:
        data += coef[index] * pow(coord, order[index])
    return data


# Assuming that the equation can be written in the form f(x,y)=g(x)*h(y), with g(x) a polynomial fonction
# of order n, arguments C0, C1, C2... Cn , such as g(x) = C0 + C1 X + C2 X^2 + .... + Cn X^n and h(y) a polynomial
# fonction of order m, arguments D0, D1, D2, ... Dm, such as h(y) = D0 + D1 X + D2 X^2 + ... + Dm X^m
# write order as (n,m)
# write the boundaries parameters in the form of (start,stop)
# write the jitter in percent
# arguments are the coeficients, in the form of (C0, C1, ..., Cn) and (D0, D1, ..., Dm)
def generate_fuzzy_data_2d_functional(order: tuple, boundaries_x: tuple, boundaries_y: tuple,  *args):
    size_x = 10
    size_y = 10
    # data_g = [[] for _ in range(size_x)]
    # data_h = [[] for _ in range(size_x)]
    data = [[] for _ in range(size_x)]
    coef_g = []
    coef_h = []
    # abscisse_x = []
    # abscisse_y = []

    if args:
        if len(args) == 2:
            for i in range(len(args[0])):
                coef_g.append(args[0][i])
            for i in range(len(args[1])):
                coef_h.append(args[1][i])
        else:
            print("number of coefficients should be equal to the order of the equation + 1")
            print("For instance: for fitting second order equation, you need to have 3 coefficients")
    else:
        coef_g = [random.uniform(-1, 1) for _ in range(order[0])]
        coef_h = [random.uniform(-1, 1) for _ in range(order[1])]

    order_g_list = list(range(order[0]))
    order_h_list = list(range(order[1]))

    abscisse_x = [boundaries_x[0] + x * (boundaries_x[1] - boundaries_x[0]) / (size_x - 1) for x in range(size_x)]
    abscisse_y = [boundaries_y[0] + y * (boundaries_y[1] - boundaries_y[0]) / (size_y - 1) for y in range(size_y)]

    data_g = [[0 for _ in range(size_x)] for _ in range(size_y)]
    data_h = [[0 for _ in range(size_x)] for _ in range(size_y)]

    data_g = polynomial_grid(data_g, abscisse_x, coef_g, order_g_list)
    # map(polynomial_grid())

    for x in range(size_x):
        for y in range(size_y):
            data_g[x].append(0)
            for index in order_g_list:
                data_g[x][y] += coef_g[index] * pow(abscisse_x[x], order_g_list[index])
            data_h[x].append(0)
            for index in order_h_list:
                data_h[x][y] += coef_h[index] * pow(abscisse_y[y], order_h_list[index])
            data[x].append(data_g[x][y] * data_h[x][y])

    # span = (np.max(data) - np.min(data))

    return abscisse_x, abscisse_y, data


def create_random_couples(list_of_elements:list):
    random.shuffle(list_of_elements)
    index = 0
    result = []
    for _ in range(len(list_of_elements) // 2):
        result.append([list_of_elements[index], list_of_elements[index+1]])
        index += 2
    if len(list_of_elements) % 2:
        result.append([list_of_elements[-1]])
    return result


if __name__ == '__main__':
    ab_x, ab_y, height = generate_fuzzy_data_2d_functional((2, 3), (-2, 2), (-2, 2), (5, -3), (1, 2, 3))
    test = np.array(height)
    plt.imshow(height)
    plt.show()
    print("G")

