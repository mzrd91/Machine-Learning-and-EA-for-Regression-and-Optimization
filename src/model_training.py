noise_scenarios = [0.0, 2.0]

def target_func(x):
    return np.sqrt(x) + np.log(3*x) + 1

def generate_data(sigma):
    X_train = np.random.uniform(1.0, 10.0, size=(30, 1))
    y_train = target_func(X_train) + np.random.normal(0, sigma**2, size=(30, 1))
    X_test = np.random.uniform(1.0, 10.0, size=(10, 1))
    y_test = target_func(X_test) + np.random.normal(0, sigma**2, size=(10, 1))
    return X_train, X_test, y_train, y_test

function_set = [f_add, f_mul, f_sub, f_div, f_sqrt, f_log, f_abs, f_max, f_min, f_inv, f_neg]
X_train, X_test, y_train, y_test = generate_data(sigma=0)
terminal_set = create_terminal_set(X_train)

def initialize_algorithm():
    return SimpleEvolution(
        Subpopulation(creators=RampedHalfAndHalfCreator(init_depth=(2, 4),
                                                        terminal_set=terminal_set,
                                                        function_set=function_set,
                                                        erc_range=(-100, 100),
                                                        bloat_weight=0.0001),
        population_size=1000,
        evaluator=RegressionEvaluator(),
        higher_is_better=False,
        elitism_rate=0.05,
        operators_sequence=[
            SubtreeCrossover(probability=0.80, arity=2),
            SubtreeMutation(probability=0.01, arity=1)
        ],
        selection_methods=[
            (TournamentSelection(tournament_size=4, higher_is_better=False), 1)
        ]),
        breeder=SimpleBreeder(),
        max_workers=5,
        max_generation=10,
        statistics=None
    )

results = []

for sigma in noise_scenarios:
    mae_train_list, mae_test_list = [], []
    best_test_mae = float('inf')
    best_solutions = []

    for _ in range(10):
        X_train, X_test, y_train, y_test = generate_data(sigma=sigma)
        algo = initialize_algorithm()
        regressor = SKRegressor(algo)
        regressor.fit(X_train, y_train)
        best_solution = algo.best_of_run_
        best_solutions.append(best_solution)
        y_pred_train = regressor.predict(X_train)
        y_pred_test = regressor.predict(X_test)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mae_train_list.append(mae_train)
        mae_test_list.append(mae_test)
        min_test_mae = min(mae_test_list)
        if min_test_mae < best_test_mae:
            best_test_mae = min_test_mae

    best_test_index = np.argmin(mae_test_list)
    best_solution_test = best_solutions[best_test_index]
    result = {
        "Noise Scenario": sigma,
        "Training Set - Best": min(mae_train_list), "Training Set - Worst": max(mae_train_list), "Training Set -  Mean": np.mean(mae_train_list),
        "Training Set - Variance": np.var(mae_train_list),
        "Test Set - Best": min(mae_test_list), "Test Set - Worst": max(mae_test_list), "Test Set - Mean": np.mean(mae_test_list), "Test Set - Variance": np.var(mae_test_list),
        "Best Regression Equation (Test) - Value": best_test_mae, "Best Regression Equation (Test) - Individual": best_solution_test
    }
    results.append(result)

df = pd.DataFrame(results)
df


for index, row in df.iterrows():
    print(f"Noise Scenario: {row['Noise Scenario']}")
    print("Training Set:")
    print(f"  - Best MAE: {row['Training Set - Best']:.4f}")
    print(f"  - Worst MAE: {row['Training Set - Worst']:.4f}")
    print(f"  - Mean MAE: {row['Training Set -  Mean']:.4f}")
    print(f"  - Variance of MAE: {row['Training Set - Variance']:.4f}")
    print("Test Set:")
    print(f"  - Best MAE: {row['Test Set - Best']:.4f}")
    print(f"  - Worst MAE: {row['Test Set - Worst']:.4f}")
    print(f"  - Mean MAE: {row['Test Set - Mean']:.4f}")
    print(f"  - Variance of MAE: {row['Test Set - Variance']:.4f}")
    print(f"Best Regression Equation (Test) - Value: {row['Best Regression Equation (Test) - Value']:.4f}")
    print(f"Best Regression Equation (Test) - Individual: {row['Best Regression Equation (Test) - Individual']}")
    print("\n")



def _target_func(x1, x2, x3):
    return x1**2 + x2**2 + x3**2

class SymbolicRegressionEvaluator(SimpleIndividualEvaluator):
    def __init__(self):
        super().__init__()
        data = np.random.uniform(-10, 50, size=(100, 3))
        self.df = pd.DataFrame(data, columns=['x1', 'x2', 'x3'])
        self.df['target'] = _target_func(self.df['x1'], self.df['x2'], self.df['x3'])

    def evaluate_individual(self, individual):
        x1, x2, x3 = self.df['x1'], self.df['x2'], self.df['x3']
        return np.mean(np.abs(individual.execute(x1=x1, x2=x2, x3=x3) - self.df['target']))

function_set = [f_add, f_mul, f_sub, f_div, f_sqrt, f_log, f_abs, f_max, f_min, f_inv, f_neg]
terminal_set = ['x1', 'x2', 'x3', 0, 1, -1]
num_runs = 50
pop_size = 1000
num_generations = 1000

def initialize_algorithm():
    return SimpleEvolution(
        Subpopulation(creators=RampedHalfAndHalfCreator(init_depth=(2, 4),
                                                        terminal_set=terminal_set,
                                                        function_set=function_set,
                                                        bloat_weight=0.0001),
        population_size=pop_size,
        evaluator=SymbolicRegressionEvaluator(),
        higher_is_better=False,
        elitism_rate=0.05,
        operators_sequence=[
            SubtreeCrossover(probability=0.80, arity=2),
            SubtreeMutation(probability=0.01, arity=1)
        ],
        selection_methods=[
            (TournamentSelection(tournament_size=4, higher_is_better=False), 1)
        ]),
        breeder=SimpleBreeder(),
        max_workers=1,
        max_generation=num_generations,
        statistics = None
    )

algo = initialize_algorithm()
algo.evolve()

def generate_data():
    x1 = np.random.uniform(-10.0, 50.0)
    x2 = np.random.uniform(-10.0, 50.0)
    x3 = np.random.uniform(-10.0, 50.0)
    objective_value = x1**2 + x2**2 + x3**2
    return x1, x2, x3, objective_value

best_fitness_values = []
best_fitness, worst_fitness, mean_fitness, variance_fitness = [], [], [], []
best_fitness = float('inf')
best_x1, best_x2, best_x3 = None, None, None

for run in range(num_runs):
    print(f"For run {run+1}")
    x1, x2, x3, objective_value = generate_data()
    best_solution = algo.execute(x1=x1, x2=x2, x3=x3)
    fitness = np.abs(best_solution - objective_value)
    best_fitness_values.append(best_solution)
    if fitness < best_fitness:
        best_fitness = fitness
        best_x1, best_x2, best_x3 = x1, x2, x3

best_fitness_values = np.array(best_fitness_values)
best_fitness = np.min(best_fitness_values)
worst_fitness = np.max(best_fitness_values)
mean_fitness = np.mean(best_fitness_values)
variance_fitness = np.var(best_fitness_values)
print(f"\n------------------------------- Statistics after {num_runs} runs ---------------------------------\n")
print(f"Best Fitness: {best_fitness:.4f}")
print(f"Worst Fitness: {worst_fitness:.4f}")
print(f"Mean Fitness: {mean_fitness:.4f}")
print(f"Variance Fitness: {variance_fitness:.4f}")
print(f"Best Solution Vector: (x1 = {best_x1:.4f}, x2 = {best_x2:.4f}, x3 = {best_x3:.4f})")
