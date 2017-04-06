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
    
    def feedForward(self, inputs):
        # Compute input activations
        self.a_input = np.array(np.append(inputs,[1]))

        #Compute  hidden activations=
        self.a_hidden = np.append(self.a_input.dot(self.W_input_to_hidden),[1])

        # Compute output activations
        self.a_out = sigmoid(self.a_hidden).dot(self.W_hidden_to_output)
        return sigmoid(self.a_out)

       
     #========================End implementation section 1==============================================="   
        
        
        
        
     #========================Begin implementation section 2=============================================#    

    def backPropagate(self, targets):
    
        u_E_out = (sigmoid(self.a_out)-targets)*(dsigmoid(self.a_out))
        w_E_out = sigmoid(self.a_hidden).reshape(len(sigmoid(self.a_hidden)),1).dot(u_E_out.reshape(1,len(u_E_out)))
        # calculate error terms for hidden
    #     print u_E_out
        u_E_hidden = self.W_hidden_to_output[:-1,:].dot(u_E_out)*(dsigmoid(self.a_hidden[:-1]))
        u_E_hidden = u_E_hidden.reshape(1,len(u_E_hidden))
   #     print u_E_hidden.shape
        w_E_hidden = self.a_input.reshape(len(self.a_input),1).dot(u_E_hidden)
    #    print w_E_hidden.shape
    #     w_E_hidden = u_E_hidden.dot(my_mnist_net.a_input.reshape(1,len(my_mnist_net.a_input)))
     #   print self.a_input.shape
    #     print len(w_E_out)
      #  print self.W_input_to_hidden.shape
        # update output weights
        self.W_hidden_to_output = self.W_hidden_to_output - self.learning_rate*w_E_out
        # update input weights
        self.W_input_to_hidden = self.W_input_to_hidden - self.learning_rate*w_E_hidden
      #  print self.W_input_to_hidden.shape
        # calculate error
        return np.square(sigmoid(self.a_out)-targets).sum()
        
     #========================End implementation section 2 =================================================="   

    
    
    
    def train(self, data,validation_data):
        start_time = time.time()
        errors=[]
        Training_accuracies=[]
      
        for it in range(self.iterations):
            np.random.shuffle(data)
            inputs  = [entry[0] for entry in data ]
            targets = [ entry[1] for entry in data ]
            
            error=0.0 
            for i in range(len(inputs)):
                Input = inputs[i]
                Target = targets[i]
                self.feedForward(Input)
                error+=self.backPropagate(Target)
            Training_accuracies.append(self.predict(data))
            
            error=error/len(data)
            errors.append(error)
            
           
            print("Iteration: %2d/%2d[==============] -Error: %5.10f  -Training_Accuracy:  %2.2f  -time: %2.2f " %(it+1,self.iterations, error, (self.predict(data)/len(data))*100, time.time() - start_time))
            # you can add test_accuracy and validation accuracy for visualisation 
            
        plot_curve(range(1,self.iterations+1),errors, "Error")
        plot_curve(range(1,self.iterations+1), Training_accuracies, "Training_Accuracy")
       
        
     

    def predict(self, test_data):
        """ Evaluate performance by counting how many examples in test_data are correctly 
            evaluated. """
        count = 0.0
        for testcase in test_data:
            answer = np.argmax( testcase[1] )
            prediction = np.argmax( self.feedForward( testcase[0] ) )
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
        
            
                                  
                                  
    
  



    
    
   