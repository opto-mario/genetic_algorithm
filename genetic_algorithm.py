import numpy as np
import random
import addons
import matplotlib.pyplot as plt


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
            self.fitness[int(max_fitness_index)] = -9999999
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

    data_test_abs, data_test = addons.generate_fuzzy_data(5, -10, 3, 5, 5, 3, 0.5, -0.03, -0.3)

    gen.load_data(data_x=data_test_abs, data_y=data_test)
    gen.create_population(20)
    offsprings = []

    for index in range(150):
        gen.get_fitness()
        mates = gen.select_mating(10)
        offsprings = gen.crossover(mates, random_crossover=True)
        test = gen.mutation(offsprings, mutation_range=10, mutation_rate=0.5)
        gen.population = np.vstack([mates, test])

    plt.plot(gen.data_x, gen.data_y)
    fitted_data_x, fitted_data_y = addons.generate_fuzzy_data(5, -10, 3, 0, offsprings[0][0], offsprings[0][1],
                                                              offsprings[0][2], offsprings[0][3], offsprings[0][4])

    plt.plot(fitted_data_x, fitted_data_y, color="red")
    plt.show()