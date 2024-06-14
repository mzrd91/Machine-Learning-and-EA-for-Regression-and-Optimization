# GP parameters
runs = 50
pop_size = 100
crossover_prob = 0.8
mutation_prob = 0.01
num_generations = 50
noise_scenarios = [0.0, 2.0]

final_results_df = pd.DataFrame(columns=["Noise Scenario", "Training Set - Best", "Training Set - Worst", "Training Set - Mean", "Training Set - Variance", "Test Set - Best", "Test Set - Worst", "Test Set - Mean", "Test Set - Variance", "Best Regression Equation (Test) - Value", "Best Regression Equation (Test) - Individual"])

# Initialize lists to store results
mae_train_results_all = {sigma: [] for sigma in noise_scenarios}
mae_test_results_all = {sigma: [] for sigma in noise_scenarios}
equations_all = {sigma: [] for sigma in noise_scenarios}

for sigma in noise_scenarios:
    print(f"\nRunning GP for Noise Scenario (sigma = {sigma}) ")
    mae_train_list, mae_test_list = [], []
    best_individuals, best_equations = [], []

    for run in range(runs):
        print(f"For Run {run+1} ")
        # Generate training and test data using the provided function
        X_train, X_test, y_train, y_test = generate_data(sigma)
        toolbox.register("evaluate", evaluate, x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)
        # Create the initial population
        population = toolbox.population(n=pop_size)
        # Create a hall of fame to keep the best individuals
        hof = tools.HallOfFame(1)
        # Create a statistics object
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        # Define the genetic algorithm
        algorithms.eaMuPlusLambda(population, toolbox, mu=pop_size, lambda_ = 2 * pop_size, cxpb=crossover_prob, mutpb=mutation_prob, ngen=num_generations, stats=stats, halloffame=hof, verbose=False)
        # Get the best individual
        best_individual = hof[0]
        # Store the best individual for this run
        best_individuals.append(best_individual)
        # Retrieve MAE and MSE from the fitness values
        mae_train, mae_test = best_individual.fitness.values
        # Append MAE results to the lists for all runs
        mae_train_list.append(mae_train)
        mae_test_list.append(mae_test)

    # Get the best individual among all runs for each noise scenario (test)
    best_individual_idx = np.argmin(mae_test_list)
    best_individual_test = best_individuals[best_individual_idx]
    # Extract the human-readable equation from the best individual
    best_equation = str(best_individual_test)
    best_equations.append(best_equation)
    mae_train_results_all[sigma] = mae_train_list
    mae_test_results_all[sigma] = mae_test_list
    equations_all[sigma] = best_equation

for sigma in noise_scenarios:
    mae_train_all, mae_test_all = np.array(mae_train_results_all[sigma]), np.array(mae_test_results_all[sigma])
    # Best MAE (Minimum MAE)
    best_mae_train, best_mae_test = np.min(mae_train_all), np.min(mae_test_all)
    # Worst MAE (Maximum MAE)
    worst_mae_train, worst_mae_test = np.max(mae_train_all), np.max(mae_test_all)
    # Mean MAE
    mean_mae_train, mean_mae_test = np.mean(mae_train_all), np.mean(mae_test_all)
    # Variance of MAE
    var_mae_train, var_mae_test = np.var(mae_train_all), np.var(mae_test_all)
    # Best regression equation - Individual
    best_equation_test = equations_all[sigma]
    # Best regression equation - Value
    best_equations_for_sigma = equations_all[sigma]
    mae_test_results_for_sigma = mae_test_results_all[sigma]
    # Find the index of the equation with the lowest MAE on the test set
    best_index = np.argmin(mae_test_results_for_sigma)
    # Get the best equation and its corresponding MAE
    best_equation = best_equations_for_sigma[best_index]
    best_mae_test = mae_test_results_for_sigma[best_index]
    final_results_df = final_results_df.append({
        "Noise Scenario": sigma,
        "Training Set - Best": best_mae_train, "Training Set - Worst": worst_mae_train, "Training Set - Mean": mean_mae_train, "Training Set - Variance": var_mae_train,
        "Test Set - Best": best_mae_test, "Test Set - Worst": worst_mae_test, "Test Set - Mean": mean_mae_test, "Test Set - Variance": var_mae_test,
        'Best Regression Equation (Test) - Value' : best_mae_test,
        'Best Regression Equation (Test) - Individual': best_equation_test
    }, ignore_index=True)

final_results_df

for index, row in final_results_df.iterrows():
    print(f"Noise Scenario: {row['Noise Scenario']}")
    print("Training Set:")
    print(f"  - Best MAE: {row['Training Set - Best']:.4f}")
    print(f"  - Worst MAE: {row['Training Set - Worst']:.4f}")
    print(f"  - Mean MAE: {row['Training Set - Mean']:.4f}")
    print(f"  - Variance of MAE: {row['Training Set - Variance']:.4f}")

    print("Test Set:")
    print(f"  - Best MAE: {row['Test Set - Best']:.4f}")
    print(f"  - Worst MAE: {row['Test Set - Worst']:.4f}")
    print(f"  - Mean MAE: {row['Test Set - Mean']:.4f}")
    print(f"  - Variance of MAE: {row['Test Set - Variance']:.4f}")

    print(f"Best Regression Equation (Test) - Value: {row['Best Regression Equation (Test) - Value']:.4f}")
    print(f"Best Regression Equation (Test) - Individual: {row['Best Regression Equation (Test) - Individual']}")
    print("\n")
