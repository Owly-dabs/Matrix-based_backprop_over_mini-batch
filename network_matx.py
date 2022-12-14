import random
import numpy as np

class Network():
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
    
    def feedforward(self,a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a

    # SGD
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data):
        # turn training_data from tuple into list
        training_data = list(training_data)
        n = len(training_data)
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for e in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                # x.shape = (number of examples in mini batch, (784, 1))
                self.update_mini_batch(mini_batch,eta)

            if test_data:
                print("Epoch {}: {}/{}".format(e,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete.".format(e))


    # Update mini batch
    def update_mini_batch(self,mini_batch,eta):
        X = [x for x,y in mini_batch]
        Y = [y for x,y in mini_batch]
        nabla_b, nabla_w = self.backprop(X,Y)
        
        self.weights = [w-(eta/len(mini_batch))*nw 
                        for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                        for b,nb in zip(self.biases, nabla_b)]
       

    # backprop
    def backprop(self, X, Y):
        """
        Return a tuple `(nabla_b, nabla_w)` representing the gradient
        for the cost_function C_x.
        `nabla_b` and `nabla_w` are layer-by-layer lists of numpy arrays,
        similar to `self.biases` and `self.weights`
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = X # shape = (mini_batch_size, 784, 1) 
        activations = [X] # list to store all activations, layer-by-layer
        Zs = [] # list to store all z vectors, layer-by-layer
        for b,w in zip(self.biases,self.weights):
            Z = np.transpose(np.dot(w, activation),(1,0,2)) + b
            Zs.append(Z)
            activation = sigmoid(Z)
            activations.append(activation)
        
        # backward pass
        delta_batch = self.cost_derivative(activations[-1],Y) * \
            sigmoid_prime(Zs[-1]) # shape = (mini_batch_size,10,1)

        ##TODO: figure out how to get delta to fit into nabla and 
        # match shapes with self.weights and self.biases
        # Refer to shell 
        nabla_b[-1] = np.sum(delta_batch, axis=0)
        nabla_w[-1] = np.squeeze(np.dot(np.transpose(delta_batch,(2,1,0)),np.transpose(activations[-2],(2,0,1))))

        for l in range(2, self.num_layers):
            Z = Zs[-l]
            sp = sigmoid_prime(Z)
            delta_batch = np.transpose(np.dot(self.weights[-l+1].transpose(), delta_batch),(1,0,2)) * sp
            nabla_b[-l] = np.sum(delta_batch,axis=0)
            nabla_w[-l] = np.squeeze(np.dot(np.transpose(delta_batch,(2,1,0)), np.transpose(activations[-l-1],(2,0,1))))
        
        return(nabla_b,nabla_w)
    def evaluate(self,test_data):
        """
        Return the number of test inputs for which the neural network outputs the correct result.
        Note that the neural network's output is assumed to be the index of whichever neuron in the final layer has the highest activation.
        """
        test_results = [(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)
        
    def cost_derivative(self,output_activations,y):
        """
        Return the vector of partial derivatives (partial C_X/partial A)
        for the output activations.
        """
        return (output_activations-y)
        


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))