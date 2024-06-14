# Machine-Learning-and-EA-for-Regression-and-Optimization
This project demonstrates the application of machine learning techniques, specifically symbolic regression, and optimization using evolutionary algorithms. The main focus is on the regression problem with different noise scenarios and optimizing a multi-variable function using various libraries and techniques, including DEAP and Platypus.

## Project Structure
### Part A: Learning

**Noise Scenarios**: Two scenarios are considered (sigma=0.0 and sigma=2.0) to analyze the effect of noise on the performance of symbolic regression.
**Target Function**: The target function for regression is defined as sqrt(x) + log(3*x) + 1.
**Data Generation**: Training and testing datasets are generated with specified noise levels.
**Evolutionary Algorithm Initialization**: A symbolic regression model is initialized using a set of primitive functions and terminals.
**Evaluation**: The model is trained and evaluated multiple times for each noise scenario, and the results are collected and summarized.
Part B: Optimization

**Target Function**: A function x1^2 + x2^2 + x3^2 is optimized using DEAP and Platypus libraries.
Evolutionary Algorithm Setup: Genetic algorithms are set up to find the optimal values of the variables that minimize the target function.
Statistics Collection: The performance of the algorithms is analyzed over multiple runs, and key statistics such as best, worst, mean, and variance of the fitness values are recorded.
Machine Learning Techniques
**Symbolic Regression**: Symbolic regression is used to find a mathematical expression that best fits a set of data points. This project uses a combination of primitive functions (add, sub, mul, div, etc.) to build complex models that approximate the target function.
Evolutionary Algorithms: Genetic programming is employed to evolve the population of models over generations, selecting the best models based on their performance on the training and test sets.
