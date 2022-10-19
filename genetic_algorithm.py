import numpy as np
import random
import addons
import matplotlib.pyplot as plt
from rich.progress import track
import time
import statistics


class PolynomialGeneticAlgorithm:
    def __init__(self, fitting_order):
        self.data_x = []
        self.data_y = []
        self.order = fitting_order
        self.population = []
        self.fitness = []

    def load_data(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def create_population(self, number_of_individual):
        self.population = np.zeros((number_of_individual, self.order), dtype=float)
        for individual in range(number_of_individual):
            for coefficient in range(self.order):
                self.population[individual][coefficient] = random.uniform(-10, 10)
        self.fitness = list(np.zeros(number_of_individual, dtype=float))

    def get_fitness(self):
        self.fitness = []
        individual_index = 0

        for _ in self.population:
            temp_data = []
            for x in self.data_x:
                temp_value = 0
                for coef in range(self.order):
                    temp_value += self.population[individual_index][coef] * pow(x, coef)
                temp_data.append(temp_value)
            try:
                self.fitness.append(1/(sum([(x1 - x2)**2 for (x1, x2) in zip(temp_data, self.data_y)])))
            except ZeroDivisionError:
                self.fitness.append(99999)
            individual_index += 1

    def select_mating(self, number_of_mates):
        selected = np.zeros((number_of_mates, self.order))
        # print(max(self.fitness))
        for i in range(number_of_mates):
            max_fitness = max(self.fitness)
            max_fitness_index = np.where(self.fitness == max_fitness)[0][0]
            selected[i] = self.population[max_fitness_index]
            self.fitness[int(max_fitness_index)] = 0.0
        return selected

    def crossover(self, parents, random_crossover=False):
        offspring = np.zeros((len(parents), self.order), dtype=float)
        couples = addons.create_random_couples(parents.tolist())
        index = 0
        for couple in couples:
            if len(couple) == 2:
                if random_crossover:
                    crossover_point = np.uint8(np.random.uniform(1, self.order))
                else:
                    crossover_point = np.uint8(self.order / 2)
                offspring[index] = couple[0][:crossover_point] + couple[1][crossover_point:]
                offspring[index+1] = couple[1][:crossover_point] + couple[0][crossover_point:]
                index += 2
            else:
                # There are no mate for this individual, so it is skipped.
                offspring[index] = couple[0]
                index += 1

        return offspring

    def mutation(self, population, mutation_range=1, mutation_rate=0.5):
        to_mutate = np.zeros(population.shape)
        for i in range(population.shape[0]):
            for j in range(population.shape[1]):
                to_mutate[i][j] = random.uniform(-mutation_range/2, mutation_range/2) if random.uniform(0, 1) > \
                                                                                         mutation_rate else 0

        return population + to_mutate


if __name__ == '__main__':
    gen = PolynomialGeneticAlgorithm(5)

    # Create the dataset, along a polynomial function of 5th order, with the coefficients [5, 3, 0.5, -0.03, -0.3]
    #  To make things more interesting, I add some random noise in the dataset.

    # polynomial_coefs = [5, 3, 0.5, -0.03, -0.3]    # -> Equivalent to 5 + 3x + 0.5x^2 - 0.03x^3 - 0.3x^4
    polynomial_coefs = [-2.2, 6.4, 1.3, -0.5, 0.2]
    data_boundary_x = [-4, 5]

    data_test_abs, data_test = addons.generate_fuzzy_data(polynomial_coefs, data_boundary_x, jitter=0)

    gen.load_data(data_x=data_test_abs, data_y=data_test)

    # Create a population consisting of 20 individuals
    gen.create_population(32)

    offsprings = []
    fitness_result = []
    fitness_timing = []
    select_mating_timing = []
    offspring_timing = []
    mutation_timing = []
    average_fitness_result = []


    # Loop for 1500 times. This needs to be adjusted until the result converges.
    for _ in track(range(3000), description="Iterating"):
        start = time.perf_counter()
        gen.get_fitness()
        fitness_timing.append(time.perf_counter()-start)

        fitness_result.append(max(gen.fitness))
        average_fitness_result.append(statistics.mean(gen.fitness))

        start = time.perf_counter()
        mates = gen.select_mating(16)
        select_mating_timing.append(time.perf_counter()-start)

        start = time.perf_counter()
        offsprings = gen.crossover(mates, random_crossover=True)
        offspring_timing.append(time.perf_counter()-start)

        start = time.perf_counter()
        mutants = gen.mutation(offsprings, mutation_range=2, mutation_rate=0.7)
        mutation_timing.append(time.perf_counter()-start)

        gen.population = np.vstack([mates, mutants])

    print(f"average fitness step : {statistics.mean(fitness_timing)}")
    print(f"average mating step : {statistics.mean(select_mating_timing)}")
    print(f"average offspring step : {statistics.mean(offspring_timing)}")
    print(f"average mutation step : {statistics.mean(mutation_timing)}")

    plt.plot(fitness_result, color='red')
    plt.plot(average_fitness_result, color='blue')
    plt.show()

    plt.scatter(gen.data_x, gen.data_y, marker="+")

    # find the best result in the population
    gen.get_fitness()
    winner = gen.population[gen.fitness.index(max(gen.fitness))]

    fitting_data_x, fitting_data_y = addons.generate_fuzzy_data([winner[0], winner[1], winner[2], winner[3], winner[4]],
                                                                data_boundary_x)

    plt.plot(fitting_data_x, fitting_data_y, color="red")
    plt.show()
