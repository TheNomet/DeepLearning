import time
import random
import numpy as np
from utils import *
from transfer_functions import * 


class NeuralNetwork(object):
    
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, iterations=50, learning_rate = 0.1):
        """
        input: number of input neurons
        hidden: number of hidden neurons
        output: number of output neurons
        iterations: how many iterations
        learning_rate: initial learning rate
        """
       
        # initialize parameters
        self.iterations = iterations   #iterations
        self.learning_rate = learning_rate
     
        
        # initialize arrays
        self.input = input_layer_size+1  # +1 for the bias node in the input Layer
        self.hidden = hidden_layer_size+1 #+1 for the bias node in the hidden layer 
        self.output = output_layer_size

        # set up array of 1s for activations
        self.a_input = np.ones(self.input)
        self.a_hidden = np.ones(self.hidden)
        self.a_out = np.ones(self.output)
        
        
        #create randomized weights Yann Lecun method in 1988's paper ( Default values)
        input_range = 1.0 / self.input ** (1/2)
        self.W_input_to_hidden = np.random.normal(loc = 0, scale = input_range, size =(self.input, self.hidden-1))
        self.W_hidden_to_output = np.random.uniform(size = (self.hidden, self.output)) / np.sqrt(self.hidden)
       
        
    def weights_initialisation(self,wi,wo):
        self.W_input_to_hidden=wi # weights between input and hidden layers
        self.W_hidden_to_output=wo # weights between hidden and output layers
   

       
        
    #========================Begin implementation section 1============================================="    
    
    def feedForward(self, inputs, t_func=sigmoid):
        # Compute input activations
        self.a_input = np.array(np.append(inputs,[1]))

        #Compute  hidden activations=
        self.a_hidden = np.append(self.a_input.dot(self.W_input_to_hidden),[1])

        # Compute output activations
        self.a_out = t_func(self.a_hidden).dot(self.W_hidden_to_output)
        return t_func(self.a_out)

       
    #========================End implementation section 1==============================================="   
        
        
        
        
    #========================Begin implementation section 2=============================================#    

    def backPropagate(self, targets, t_func=sigmoid, d_func=dsigmoid):
        
        # output gradients
        u_E_out = (t_func(self.a_out)-targets)*(d_func(self.a_out))
        w_E_out = t_func(self.a_hidden).reshape(len(t_func(self.a_hidden)),1).dot(u_E_out.reshape(1,len(u_E_out)))
        
        # hidden gradients
        u_E_hidden = self.W_hidden_to_output[:-1,:].dot(u_E_out)*(d_func(self.a_hidden[:-1]))
        u_E_hidden = u_E_hidden.reshape(1,len(u_E_hidden))
        w_E_hidden = self.a_input.reshape(len(self.a_input),1).dot(u_E_hidden)
 
        # update output weights
        self.W_hidden_to_output = self.W_hidden_to_output - self.learning_rate*w_E_out
        
        # update input weights
        self.W_input_to_hidden = self.W_input_to_hidden - self.learning_rate*w_E_hidden

        # calculate error
        return np.square(t_func(self.a_out)-targets).sum()
        
    #========================End implementation section 2 =================================================="   

    
    
    
    def train(self, data, validation_data, t_fnc=sigmoid, d_fnc=dsigmoid):
        start_time = time.time()
        self.errors=[]
        Training_accuracies=[]
      
        for it in range(self.iterations):
            np.random.shuffle(data)
            inputs  = [entry[0] for entry in data ]
            targets = [ entry[1] for entry in data ]
            
            error=0.0 
            for i in range(len(inputs)):
                Input = inputs[i]
                Target = targets[i]
                self.feedForward(Input, t_func=t_fnc)
                error+=self.backPropagate(Target, t_func=t_fnc, d_func=d_fnc)
            Training_accuracies.append(self.predict(data, t_func=t_fnc))
            
            error=error/len(data)
            self.errors.append(error)
            
           
            print("Iteration: %2d/%2d[==============] -Error: %5.10f  -Training_Accuracy:  %2.2f  -time: %2.2f " %(it+1,self.iterations, error, (self.predict(data, t_func=t_fnc)/len(data))*100, time.time() - start_time))
            # you can add test_accuracy and validation accuracy for visualisation 
            
        plot_curve(range(1,self.iterations+1),self.errors, "Error")
        plot_curve(range(1,self.iterations+1), Training_accuracies, "Training_Accuracy")
       
        
     

    def predict(self, test_data, t_func=sigmoid):
        """ Evaluate performance by counting how many examples in test_data are correctly 
            evaluated. """
        count = 0.0
        for testcase in test_data:
            answer = np.argmax( testcase[1] )
            prediction = np.argmax( self.feedForward( testcase[0], t_func=t_func) )
            count = count + 1 if (answer - prediction) == 0 else count 
            count= count 
        return count 
    
    
    
    def save(self, filename):
        """ Save neural network (weights) to a file. """
        with open(filename, 'wb') as f:
            pickle.dump({'wi':self.W_input_to_hidden, 'wo':self.W_hidden_to_output}, f )
        
        
    def load(self, filename):
        """ Load neural network (weights) from a file. """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        # Set biases and weights
        self.W_input_to_hidden=data['wi']
        self.W_hidden_to_output = data['wo']
        
            
                                  
                                  
    
  



    
    
   