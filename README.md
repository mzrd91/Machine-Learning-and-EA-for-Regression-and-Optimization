# Machine-Learning-and-EA-for-Regression-and-Optimization
This project demonstrates the application of machine learning techniques, specifically symbolic regression, and optimization using evolutionary algorithms. The main focus is on the regression problem with different noise scenarios and optimizing a multi-variable function using various libraries and techniques, including DEAP and Platypus.

## Algorithm Parameters

| Parameter | Value |
|---|---|
| Algorithm | GP |
| Package Used | EC-KitY |
| Training Data Size | 30 |
| Test Data Size | 10 |
| x lower bound | 1.0 |
| x upper bound | 10.0 |
| Noise Values | 0.2 |
| Evaluation Metric | MAE |
| Number of Runs | 50 |
| Number of Generations | 10 |
| Population Size | 1000 |
| Selection Type | Tournament Selection |
| Tournament Size | 4 |
| Crossover Probability | 0.80 |
| Mutation Probability | 0.01 |

## Project Structure
### Part A: Learning

**Noise Scenarios**: Two scenarios are considered (sigma=0.0 and sigma=2.0) to analyze the effect of noise on the performance of symbolic regression.

**Target Function**: The target function for regression is defined as $sqrt(x) + log(3*x) + 1$.

**Data Generation**: Training and testing datasets are generated with specified noise levels.

**Evolutionary Algorithm Initialization**: A symbolic regression model is initialized using a set of primitive functions and terminals.
**Evaluation**: The model is trained and evaluated multiple times for each noise scenario, and the results are collected and summarized.

### Part B: Optimization

**Target Function**: A function $x_1^2 + x_2^2 + x_3^2$  is optimized using DEAP and Platypus libraries.
Evolutionary Algorithm Setup: Genetic algorithms are set up to find the optimal values of the variables that minimize the target function.
Statistics Collection: The performance of the algorithms is analyzed over multiple runs, and key statistics such as best, worst, mean, and variance of the fitness values are recorded.


## Machine Learning Techniques:

**Symbolic Regression**: Symbolic regression is used to find a mathematical expression that best fits a set of data points. This project uses a combination of primitive functions (add, sub, mul, div, etc.) to build complex models that approximate the target function.
Evolutionary Algorithms: Genetic programming is employed to evolve the population of models over generations, selecting the best models based on their performance on the training and test sets.

# Results:
## Learning Part:

### Results Obtained for Noise Scenario 0

| Metric         | Training Data | Test Data |
|----------------|---------------|-----------|
| **Best MAE**   | 0.0339        | 0.0260    |
| **Worst MAE**  | 0.3389        | 0.8482    |
| **Mean MAE**   | 0.1312        | 0.1530    |
| **Variance MAE**| 0.0056       | 0.0192    |
| **Best Solution**|             | 0.0260    |

## Results Obtained for Noise Scenario 2

| Metric         | Training Data | Test Data |
|----------------|---------------|-----------|
| **Best MAE**   | 2.0205        | 1.6805    |
| **Worst MAE**  | 3.9593        | 5.9986    |
| **Mean MAE**   | 2.9399        | 3.3676    |
| **Variance MAE**| 0.1741       | 0.7980    |
| **Best Solution**|             | 1.6805    |


## Optimization Part:

### General Results Obtained

| Metric             | Value       |
|--------------------|-------------|
| **Best Fitness**   | 189.1859    |
| **Worst Fitness**  | 4656.6467   |
| **Mean Fitness**   | 2159.8262   |
| **Variance Fitness**| 1591022.7807|
| **Best Solution Vector** | (x1 = 6.6700, x2 = 37.0461, x3 = 19.0318) |

### Conclusion

These results demonstrate the performance of the model under different noise scenarios. The tables above summarize the key metrics for both training and test data, providing a clear comparison of the model's accuracy and stability.

