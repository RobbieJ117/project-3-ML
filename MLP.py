import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import math
import matplotlib.pyplot as plt
import os.path
import numba
import copy

class MLP(object):

    def __init__(self, experiment_id, in_dim, h_nodes, out_dim):
        self.id = "{}.txt".format(experiment_id)
        self.in_dim = in_dim #dimensions/features of the input data
        self.h_nodes = h_nodes #number of hidden nodes
        self.out_dim = out_dim # number of output nodes
        # weights_ih are the weights from the input data vectors to the hidden layer
        self.weights_ih = np.random.normal(loc=0.0, scale=.001, size=(in_dim, h_nodes))
        # weights_ho are the weights from the hidden nodes to the output layer
        self.weights_ho = np.random.normal(loc=0.0, scale=.1, size=(h_nodes, out_dim))
        self.epochs = 3 #number of iterations of weight correction steps
        self.ada = -.1
        self.loss_history = [] #used to store error for error_vs_time graphs
        self.batch_size = 1800
        self.iteration=0
        self.product= None
        self.samples_by_nodes = None
        self.out_vector = None
        self.current_pop = []


    def feed_forward(self, X):
        #multiply inputs across first set of weights into h_nodes
        self.product = X.dot(self.weights_ih)
        #apply sigmoid function to values in h_nodes
        # this matrix is used also to calculate backpropagated error
        self.samples_by_nodes = np.tanh(self.product)
        # mutiply values in h_nodes across weights to output node
        # out_vector is the vector of outputs from the network
        # its dimensions are (number samples in X)x(1)
        self.out_vector = self.samples_by_nodes.dot(self.weights_ho)
        return self.out_vector

    def backprop(self, X, Y):
        batch_X = np.array
        batch_Y_act = np.array
        batch_Y_pred = np.array
        # if the number of data points can be evenly divided by the batch size
        if (len(Y) % self.batch_size == 0):
            num_batches = len(Y) / self.batch_size  # number of batches given data size
        # if the number of data points in X can not be evenly divided by the batch size
        else:
            num_batches = (len(Y) // self.batch_size) + 1
        # This loop repeats for each batch
        num_batches = (int)(num_batches + 0)
        for i in range(0, num_batches):
            '''
            The below block of code splits the data into smaller batches so that
            gradient descent and weight update can be done in batches
            '''
            for j in range(0, self.epochs):
                print("epoch:{}\n".format(j))
                # if it is the last batch, do not define array ending index
                if (i == (num_batches - 1)):
                    batch_X = self.batch_split(X, i, 0)
                    batch_Y_act = self.batch_split(Y, i, 0)
                    batch_Y_pred = self.feed_forward(batch_X)
                else:
                    batch_X = self.batch_split(X, i, 1)
                    batch_Y_act = self.batch_split(Y, i, 1)
                    batch_Y_pred = self.feed_forward(batch_X)
                error_0 = (batch_Y_act - batch_Y_pred)
                error_1 = -1*((batch_Y_act - np.tanh(batch_Y_pred)))/len(batch_X)
                error_2 = np.multiply(error_0, error_0)
                loss = .5 * np.sum(error_2) / len(batch_X)
                self.loss_history.append(loss)
                if (loss > .01):
                    # The below applies gradient descent for each batch for each epoch
                    # derivative of the hidden layer
                    temp1 = np.ndarray(shape=(self.samples_by_nodes.shape))
                    temp1.fill(1)
                    h_deriv = (temp1 - np.multiply(np.tanh(self.product), np.tanh(self.product)))
                    temp2 = np.ndarray(shape=(self.out_vector.shape))
                    temp2.fill(1)
                    o_deriv = (temp2 - np.multiply((np.tanh(batch_Y_pred)), (np.tanh(batch_Y_pred))))
                    d3 = np.multiply((error_1), o_deriv)
                    dJdW2 = np.dot(self.samples_by_nodes.T, d3)

                    d2 = np.multiply(np.dot(d3, self.weights_ho.T), h_deriv)
                    dJdW1 = np.dot(batch_X.T, d2)
                    self.weights_ih += self.ada * dJdW1
                    self.weights_ho += self.ada * dJdW2
                    self.iteration += 1
                else:
                    j+=1

    def batch_split(self, X, i, switch):
        start = (self.batch_size * i)
        end = ((self.batch_size *(1+ i)))
        if(switch==0):
            temp = X[start:]
            return temp
        else:
            temp = X[start:end]
            return temp

    def test(self, X, Y):
        i = self.iteration-1
        G = self.feed_forward(X)
        #print(Y)
        #print(G)
        mae = metrics.mean_absolute_error(Y, G)
        rmse = metrics.mean_squared_error(Y, G)
        rmse = math.sqrt(rmse)
        mean_y = Y[10:20]
        mean_g = G[10:20]
        #A = np.hstack((X,G)) #set of vectors of predicted points
        #B = np.hstack((X,Y)) #set of vectors for actual points
        #res = 1 - np.dot(A / np.linalg.norm(A, axis=1)[..., None], (B / np.linalg.norm(B, axis=1)[..., None]).T)# compute cosine distance between vectors
        #cos_dist = res.mean()# mean cosine distance
        cos_dist = "\t Not currently implemented"
        reults_string = "\nIteration{}\n\nRMSE:{}\nMAE:{}\nMean Cosine similarity{}\nMean Y{}\n\n Mean G{}\n\n".format(i, rmse, mae, cos_dist, mean_y, mean_g)
        if not os.path.isfile(self.id):
            f = open(self.id, "w")
            header = "{}:\nAda:{}\nEpochs:{}\nhidden nodes{}\nBatch size: {}\n".format(self.id, self.ada, self.epochs, self.h_nodes, self.batch_size)
            f.write(header)
        else:
            f = open(self.id, "a")
        f.write(reults_string)
        f.close()


    '''
    Prints a plot of the error versus training iterations
    '''

    def print_results(self):
        out_name = self.id.replace(".txt", "")
        file_name = "{}.png".format(out_name)
        fig = plt.figure()
        plt.plot(np.arange(0, len(self.loss_history)), self.loss_history)
        fig.suptitle("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        fig.savefig(file_name)
        plt.close(fig)
        self.lossHistory = []

################################################# DIFFERENTIAL EVOLUTION PORTION ##################################################

    '''
    difEvoTrain performs Differential Evolution tuning of a feedforward Neural Network

    @param X: input vector
           Y: class / expected output vector      
           beta: scaling factor B ϵ (0, inf), controls the amplification of differential variations (xi2-xi3).
           pr: probability of recombination pr ϵ (0, 1)
           population: size of the population to be generated population ϵ (1, inf), default size is 500

    @return: evolved weight matrices from the configuration with the best fitness

    '''
    def difEvoTrain(self, X, Y, beta, pr, population_size=20, batch_size=50):
        training_batches, validation_batches, num_batches = self.difEvoBatching(X, Y, batch_size)
        generation = 0
        maxGen = 50
        population = self.difEvoPopGen(population_size)
        minFitness = float("inf")
        mostFitIndividual = []
        while(generation < maxGen):
            for i in range(0, len(population)):
                xit = population[i]
                random_index = np.random.randint(0, num_batches)
                ################## EVALUATE FITNESS OF XIT ####################
                self.weights_ih = xit[0]
                self.weights_ho = xit[1]
                xit_sq_error = 0
                for j in range(0,len(training_batches[random_index])):
                    xit_sq_error += (validation_batches[random_index][j] - self.feed_forward(training_batches[random_index][j]))**2
                mse_f_xit = xit_sq_error / len(training_batches[random_index])
                ###############################################################
                ################## MUTATION THEN CROSSOVER ####################
                uit = self.difMutation(population, i, beta)
                xit_prime = self.difCrossover(xit, uit, pr)
                ###############################################################
                ############### EVALUATE FITNESS OF XIT_PRIME #################
                self.weights_ih = xit_prime[0]
                self.weights_ho = xit_prime[1]
                xit_prime_sq_error = 0
                for j in range(0,len(training_batches[random_index])):
                    xit_prime_sq_error += (validation_batches[random_index][j] - self.feed_forward(training_batches[random_index][j]))**2
                mse_f_xit_prime = xit_prime_sq_error / len(training_batches[random_index])
                ###############################################################
                ############# PUT MOST FIT BACK INTO POPULATION ###############
                if mse_f_xit_prime < mse_f_xit:
                    population[i] = xit_prime
                    if mse_f_xit_prime < minFitness:
                        minFitness = mse_f_xit_prime
                        mostFitIndividual = xit_prime
                else:
                    population[i] = xit
                    if mse_f_xit < minFitness:
                        minFitness = mse_f_xit
                        mostFitIndividual = xit
                ## EXTRA STOP CONDITION ####
                if minFitness < 0.01:
                    generation = maxGen
                    print('Early Termination.')
            generation += 1
            self.loss_history.append(minFitness)
        ############# RETURN MOST FIT OF POPULATION ################
        self.weights_ih = mostFitIndividual[0]
        self.weights_oh = mostFitIndividual[1]


    '''
    difMutation is a helper method for performing Differential Evolution Mutation

    @param population: a list of solutions
           i: current index being evaluated
           beta: scaling factor B ϵ (0, inf)

    @return uit: a trial vector

    '''
    def difMutation(self, population, i, beta):
        xi1, xi2, xi3 = 0, 0, 0
        limit = len(population)
        randint = np.random.randint
        while(True):
            xi1 = randint(0, limit)
            xi2 = randint(0, limit)
            xi3 = randint(0, limit)
            if not (xi1 == xi2 and xi2 == xi3 and xi3 == i):
                break
        uit0 = population[xi1][0] + beta*(population[xi2][0] - population[xi3][0])
        uit1 = population[xi1][1] + beta*(population[xi2][1] - population[xi3][1])
        return [uit0, uit1]


    '''
    difCrossover is a helper method for performing Differential Evolution Crossover

    @param xit: the parent example from the population
           uit: the trial vector created through mutation
           pr: the probability of recombination
           exponential: boolean, true if exponential crossover is to be used

    @return xit_prime: offspring of parent (xit) and trial vector (uit)

    '''
    def difCrossover(self, xit, uit, pr, exponential=False):
        if not exponential: # binomial crossover
            # for all elements in the weight matrix, crossover if probability satisfied
            # select j* crossover point for each weight matrix
            randint = np.random.randint
            jstar_x0 = randint(0, len(xit[0]))
            jstar_y0 = randint(0, len(xit[0][0]))

            jstar_x1 = randint(0, len(xit[1]))
            jstar_y1 = randint(0, len(xit[1][0]))

            # loop over the first numpy array
            for i in range(0, len(xit[0])):
                for j in range(0, len(xit[0][0])):
                    # if number from uniform distribution of (0,1) < probability
                    if np.random.uniform(0,1) < pr:
                        # crossover uit element into xit_prime
                        xit[0][i][j] = uit[0][i][j]
            # stick j* into xit_prime[0]
            xit[0][jstar_x0][jstar_y0] = uit[0][jstar_x0][jstar_y0]
            # return xit_prime
            for i in range(0, len(xit[1])):
                for j in range(0, len(xit[1][0])):
                    if np.random.uniform(0,1) < pr:
                        xit[1][i][j] = xit[1][i][j]
            # stick j* into xit_prime[1]
            xit[1][jstar_x1][jstar_y1] = uit[1][jstar_x1][jstar_y1]
            return xit
        else:               # exponential crossover
            # to be completed if time permits
            pass


    '''
    difEvoPopGen is a helper method that generates a population of weight matrices to be evaluated by differential evolution

    @param size: the number of individuals to generate

    @return population: list containing size number of individuals

    '''
    def difEvoPopGen(self, size):
        population = []
        for i in range(0, size):
            population.append([np.random.uniform(low=-.2, high = .2, size=(self.in_dim, self.h_nodes)),
                                np.random.uniform(low=-.2, high=.2, size=(self.h_nodes, self.out_dim))])
        return population
    
    '''
    difEvoBatching is a helper method that chunks the training and validation data into batches and returns them as lists

    @param X, Y: training and validation sets, respectively
           batch_size: integer size of batches to be made

    @return lists for training and validation batches
    '''
    def difEvoBatching(self, X, Y, batch_size):
        training_batches = []
        validation_batches = []
        # GIO'S CODE FOR EASIER INTERFACING WITH BATCH SPLIT PROCESS
        if (len(Y) % self.batch_size == 0):
            num_batches = len(Y) / self.batch_size  # number of batches given data size
        # if the number of data points in X can not be evenly divided by the batch size
        else:
            num_batches = (len(Y) // self.batch_size) + 1

        for i in range(0, batch_size):     
            if (i == (num_batches - 1)):
                training_batches.append(self.batch_split(X, i, 0))
                validation_batches.append(self.batch_split(Y, i, 0))
            else:
                training_batches.append(self.batch_split(X, i, 1))
                validation_batches.append(self.batch_split(Y, i, 1))

        return training_batches, validation_batches, num_batches
############################################### END DIFFERENTIAL EVOLUTION PORTION ################################################

#######################///////Evolutionary Strategies///////// ################################################

    """
    The below initializes the population for ES
    An individual's genes are represented as a list of:
    [input-to-hidden weights, hidden-to-output weights, individual step size, fitness score]
    Individuals of the current population are contained in a list[] current_pop
    """
    def init_ES(self, num_samples, maxgen):
        self.num_pop = (int) (4+(3*math.log(num_samples)))
        self.num_chld = (int)(self.num_pop/2)
        self.maxGen = maxgen
        for i in range(0, self.num_pop):
            step_size = np.random.normal(loc=0.0, scale=1)
            fitness = 0
            individual = [np.random.normal(loc=0.0, scale=.001, size=(self.in_dim, self.h_nodes)), np.random.normal(loc=0.0, scale=.01, size=(self.h_nodes, self.out_dim)), step_size, fitness]
            self.current_pop.append(individual)

    """
    The training method consists of 4 steps that loop until the max number of generations is met
    or the prediction error of the fittest individual is small enough
    Selection operation used is rank based selection based on prediction error
    The steps in this loop are:
        1.) create offspring
        2.) score fitness of parents and offspring
        3.) remove all but the mu fittest individuals from the population
        4.) improvement in error between generations
    """

    def train_ES(self, features, targets):
        counter = 0
        while(counter<self.maxGen):
            self.progenate()
            self.score_fitness(features, targets)
            # sort the current mu + lambda population by fitness score
            self.current_pop.sort(key=lambda x: x[3])
            # keep only the mu best individuals in the population
            del self.current_pop[self.num_pop:]
            #Add loss to loss history (for graphical display)
            loss = self.generation_loss(features, targets)
            self.loss_history.append(loss)
            #Break from loop if error is small enough
            if(self.current_pop[0][3]<.001):
                print("Broke early")
                return
            else:
                self.iteration+=1
                counter+=1

    """
    Progenate creates a the offspring of a generation
    It uses mutation only. The weights between layers of the network represent an individual gene
    The np.random.choice function randomly selects which genes in an individual will be mutated
    The mutation is scaled by the individual's step size attribute
    """
    def progenate(self):
        for i in range(0, self.num_chld):
            random_selection = np.random.choice(self.num_pop)
            child1 = copy.deepcopy(self.current_pop[random_selection])
            mutation_rate = np.random.choice((0, 1, 2), p=[.3, .4, .3])
            # randomly mutates new children from each member of current population
            # and then adds them to the population
            # For extra randomness, different mutation rates are randomly variable
            if(mutation_rate==0):
                child1[0] += self.ada * np.multiply(child1[0], child1[2]*np.random.choice((1, 0), size=(child1[0].shape), p=[.30, .70]))
                child1[1] += self.ada * np.multiply(child1[1], child1[2]*np.random.choice((1, 0), size=(child1[1].shape), p=[.30, .70]))
            elif(mutation_rate == 1):
                child1[0] += self.ada * np.multiply(child1[0], child1[2]*np.random.choice((1, 0), size=(child1[0].shape), p=[.50, .50]))
                child1[1] += self.ada * np.multiply(child1[1], child1[2]*np.random.choice((1, 0), size=(child1[1].shape), p=[.50, .50]))
            else:
                child1[0] += self.ada * np.multiply(child1[0], child1[2]*np.random.choice((1, 0), size=(child1[0].shape), p=[.70, .30]))
                child1[1] += self.ada * np.multiply(child1[1], child1[2]*np.random.choice((1, 0), size=(child1[1].shape), p=[.70,.30]))
            #add generated child to population
            self.current_pop.append(child1)

    """
    Perform fitness assessment of all children and parents in a generation
    """
    def score_fitness(self, features, targets):
        #iterate over each individual in currently in the population
        for i in range(0, self.num_pop + self.num_chld):
            self.weights_ih = self.current_pop[i][0]
            self.weights_ho = self.current_pop[i][1]
            #Fitness based on error. An individual's fitness score is evaluated/assigned here
            error_0 = np.sum(targets - self.feed_forward(features))/len(features)
            self.current_pop[i][3] = abs(error_0)

    '''
    Auxiliary method for computing error
    '''
    #Tracks the loss per generation. Used for graphical output
    def generation_loss(self, features, targets):
        self.weights_ih = self.current_pop[0][0]
        self.weight_ho = self.current_pop[0][1]
        error = targets - self.feed_forward(features)
        loss = np.sum(np.multiply(error, error))/len(features)
        return loss


################################################# Genetic Algorithm PORTION #######################################################
    # Refrences: https://lethain.com/genetic-algorithms-cool-name-damn-simple/

#Establish the population for the Genetic Algorithm

    def init_pop_ga(self, num_samples, maxgen):
        self.num_pop = (int)(4 + (3 * math.log(num_samples)))
        self.maxGen = maxgen
        for i in range(0, self.num_pop):
            step_size = np.random.normal(loc=0.0, scale=1)
            fitness = 0
            individual = [np.random.normal(loc=0.0, scale=.001, size=(self.in_dim, self.h_nodes)),
                          np.random.normal(loc=0.0, scale=.01, size=(self.h_nodes, self.out_dim)), step_size, fitness]
            self.current_pop.append(individual)

    #The training method
    # 1. Create the initial population
    # 2. Give every member a fitness score
    # 3. Select two parents with the better fitness score
    # 4. Perform crossover of the 2 parents found in Selection
    # 5. Mutatate
    # 6. Recaculate the fitness score
    # 7. Determine error
    # 8. Loop till reach max generations
    def train_ga(self, features, targets):
        counter = 0
        self.score_fitness_ga(features, targets)
        while(counter<self.maxGen):
            self.selection_ga() # Select 2 parents
            self.crossover_mutate_ga() # Perform Crossover and mutation to repopulate
            self.score_fitness_ga(features, targets)   # Rescore
            # sort the current mu + lambda population by fitness score
            self.current_pop.sort(key=lambda x: x[3])
            # keep only the mu best individuals in the population
            del self.current_pop[self.num_pop:]
            #Add loss to loss history (for graphical display)
            loss = self.loss_ga(features, targets)
            self.loss_history.append(loss)
            #Break from loop if error is small enough
            if(self.current_pop[0][3]<.001):
                print("Broke early")
                return
            else:
                self.iteration+=1
                counter+=1

# Select 2 parents based off best fitness
    def selection_ga(self):
        self.parent1 =   self.current_pop[1]
        self.parent2 =   self.current_pop[1]
        for i in range (0, self.num_pop):
            if(self.parent1[3] < self.current_pop[i][3]):
                self.parent1 = self.current_pop[i]
            elif(self.parent2[3] < self.current_pop[i][3]):
                self.parent2 = self.current_pop[i]

    def crossover_mutate_ga(self):
        lengthParent1   =   len(self.parent1)
        lengthParent2   =   len(self.parent2)

        random_selection = np.random.choice(lengthParent1)

        child1       =   self.parent1[:random_selection] + self.parent2[random_selection:]
        child2       =   self.parent2[:random_selection] + self.parent1[random_selection:]


        mutation_rate = np.random.choice((0, 1, 2), p=[.3, .4, .3])
        # randomly mutates new children from each member of current population
        # and then adds them to the population
        # For extra randomness, different mutation rates are randomly variable
        if(mutation_rate==0):
            child1[0] += self.ada * np.multiply(child1[0], child1[2]*np.random.choice((1, 0), size=(child1[0].shape), p=[.30, .70]))
            child1[1] += self.ada * np.multiply(child1[1], child1[2]*np.random.choice((1, 0), size=(child1[1].shape), p=[.30, .70]))
        elif(mutation_rate == 1):
            child1[0] += self.ada * np.multiply(child1[0], child1[2]*np.random.choice((1, 0), size=(child1[0].shape), p=[.50, .50]))
            child1[1] += self.ada * np.multiply(child1[1], child1[2]*np.random.choice((1, 0), size=(child1[1].shape), p=[.50, .50]))
        else:
            child1[0] += self.ada * np.multiply(child1[0], child1[2]*np.random.choice((1, 0), size=(child1[0].shape), p=[.70, .30]))
            child1[1] += self.ada * np.multiply(child1[1], child1[2]*np.random.choice((1, 0), size=(child1[1].shape), p=[.70,.30]))
        #add generated child to population
        self.current_pop.append(child1)
        if (mutation_rate == 0):
            child2[0] += self.ada * np.multiply(child2[0], child2[2] * np.random.choice((1, 0), size=(child2[0].shape),
                                                                                        p=[.30, .70]))
            child2[1] += self.ada * np.multiply(child2[1], child2[2] * np.random.choice((1, 0), size=(child2[1].shape),
                                                                                        p=[.30, .70]))
        elif (mutation_rate == 1):
            child2[0] += self.ada * np.multiply(child2[0], child2[2] * np.random.choice((1, 0), size=(child2[0].shape),
                                                                                        p=[.50, .50]))
            child2[1] += self.ada * np.multiply(child2[1], child2[2] * np.random.choice((1, 0), size=(child2[1].shape),
                                                                                        p=[.50, .50]))
        else:
            child2[0] += self.ada * np.multiply(child2[0], child2[2] * np.random.choice((1, 0), size=(child2[0].shape),
                                                                                        p=[.70, .30]))
            child2[1] += self.ada * np.multiply(child2[1], child2[2] * np.random.choice((1, 0), size=(child2[1].shape),
                                                                                        p=[.70, .30]))
            # add generated child to population
        self.current_pop.append(child1)


# Score the fitness
    def score_fitness_ga(self, features, targets):
        #iterate over each individual in currently in the population
        for i in range(0, self.num_pop):
            self.weights_ih = self.current_pop[i][0]
            self.weights_ho = self.current_pop[i][1]
            #Fitness based on error. An individual's fitness score is evaluated/assigned here
            error_0 = np.sum(targets - self.feed_forward(features))/len(features)
            self.current_pop[i][3] = abs(error_0)

# Compute the error
    def loss_ga(self, features, targets):
        self.weights_ih = self.current_pop[0][0]
        self.weight_ho = self.current_pop[0][1]
        error = targets - self.feed_forward(features)
        loss = np.sum(np.multiply(error, error))/len(features)
        return loss


############################################### END Genetic Algorithm PORTION #####################################################
