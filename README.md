# handwriten_digits_using_mlp

The program aims to detect the handwritten digits from 1 to 9. User will input the data using mouse cursor and the MLP will try to predict the output,
The training has used sigmoid and softmax function as their activation function. I have used MNIST dataset to evaulate the network. There are three files, 
# b_and_w_generator 
it generates the random weight and bias for the experiment
# mlp
it is used only for training and updating the parameters. After all the process, it saves the update in weights and biases files for future usage.
# test_training
This code tests the performance of network. It opens an editor where user can input a custom digit and the network returns the highest scoring parameter, which gives the predition of a number being a particular number.
