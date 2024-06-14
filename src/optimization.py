# Part b: Optimization with DEAP

from deap import base, creator, tools, algorithms

# Fitness Function Definition
def fitness_function(individual):
    x1, x2, x3 = individual
    return x1**2 + x2**2 + x3**2,

# Initialize DEAP creator
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)  # Minimizing fitness

# Instantiating and populating DEAP toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -10.0, 50.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=3)

# Random seed for reproducibility
random.seed(2000)

# GA parameters
runs = 50
pop_size = 100
generations = 100
crossover_prob = 0.8
mutation_prob = 0.01

best_fitness_values, worst_fitness_values, mean_fitness_values, variance_fitness_values = [], [], [], []

for run in range(runs):
    population = toolbox.population(n=pop_size)  # Initializing population
    stats = tools.Statistics(lambda ind: ind.fitness.values)  # Calculating statistics
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    stats.register("var", np.var)

    algorithms.eaSimple(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, ngen=generations, stats=stats, verbose=True)  # Algorithm

    best_ind = tools.selBest(population, k=1)[0]
    best_fitness = best_ind.fitness.values[0]

    best_fitness_values.append(best_fitness)
    worst_fitness_values.append(stats.compile(population)["max"])
    mean_fitness_values.append(stats.compile(population)["avg"])
    variance_fitness_values.append(stats.compile(population)["var"])

best_fitness_values = np.array(best_fitness_values)
best_fitness = min(best_fitness_values)  # Minimization
worst_fitness = max(best_fitness_values)
mean_fitness = np.mean(mean_fitness_values)
variance_fitness = np.mean(variance_fitness_values)
best_solution = tools.selBest(population, k=1)[0]

print(f"\n------------------------------- Statistics after {num_runs} runs -------------------------------\n")
print(f"Best Fitness: {best_fitness:.4f}")
print(f"Worst Fitness: {worst_fitness:.4f}")
print(f"Mean Fitness: {mean_fitness:.4f}")
print(f"Variance Fitness: {variance_fitness:.4f}")
print(f"Best Solution Vector: {best_solution}")

# Part b: Optimization with Platypus

from platypus import Problem, Real, GeneticAlgorithm

# Fitness Function Definition
def fitness_function(x):
    x1, x2, x3 = x
    return x1**2 + x2**2 + x3**2

# Problem and optimization settings
problem = Problem(3, 1)
problem.types[:] = [Real(-10, 50)] * 3
problem.function = fitness_function

# Random seed for reproducibility
random.seed(2000)

# GA Parameters
num_runs = 50
pop_size = 100
crossover_prob = 0.8
mutation_prob = 0.01
num_generations = 100

# Create an instance of the Genetic Algorithm
algorithm = GeneticAlgorithm(problem, population_size=pop_size, crossover_probability=crossover_prob, mutation_probability=mutation_prob, verbose=True)

best_fitness_values = []

for _ in range(num_runs):
    print(f"\n\nRun : {_+1} statistics \n\n")
    algorithm.run(num_generations)  # Run the GA for a fixed number of generations

    # Get the best solution for this run
    best_solution = algorithm.result[0]
    best_fitness_values.append(best_solution.objectives[0])

# Calculate statistics
best_fitness_values = np.array(best_fitness_values)
best_fitness = np.min(best_fitness_values)  # Minimization
worst_fitness = np.max(best_fitness_values)
mean_fitness = np.mean(best_fitness_values)
variance_fitness = np.var(best_fitness_values)

print(f"\n------------------------------- Statistics after {num_runs} runs -------------------------------\n")
print(f"Best Fitness: {best_fitness:.4f}")
print(f"Worst Fitness: {worst_fitness:.4f}")
print(f"Mean Fitness: {mean_fitness:.4f}")
print(f"Variance Fitness: {variance_fitness:.4f}")
print(f"Best Solution Vector: {best_solution.variables}")
