# Using DEAP

from deap import algorithms, base, creator, tools, gp

# Define the target function
def target_func(x):
    return np.sqrt(x) + np.log(3 * x) + 1

def generate_data(sigma):
    X_train = np.random.uniform(1.0, 10.0, size=(30, 1))
    y_train = target_func(X_train) + np.random.normal(0, sigma**2, size=(30, 1))
    X_test = np.random.uniform(1.0, 10.0, size=(10, 1))
    y_test = target_func(X_test) + np.random.normal(0, sigma**2, size=(10, 1))
    return X_train, X_test, y_train, y_test

def evaluate(individual, x_train, y_train, x_test, y_test):
    func = gp.compile(expr=individual, pset=pset)
    y_pred_train = [func(x) for x in x_train]
    y_pred_test = [func(x) for x in x_test]
    mae_train = np.mean(np.abs(np.array(y_pred_train) - y_train))
    mae_test = np.mean(np.abs(np.array(y_pred_test) - y_test))
    return mae_train, mae_test

max_depth = 5

# Create the DEAP types and primitives
pset = gp.PrimitiveSet("MAIN", arity=1)
pset.addPrimitive(operator.add, arity=2)
pset.addPrimitive(operator.sub, arity=2)
pset.addPrimitive(operator.mul, arity=2)
pset.addEphemeralConstant("rand_const", lambda: random.uniform(-1, 1))
pset.renameArguments(ARG0='x')
pset.context["x"] = float

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_depth)

# Set random seed for reproducibility
random.seed(142)
