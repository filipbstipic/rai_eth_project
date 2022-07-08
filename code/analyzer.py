import torch
import networks
import transformers as tf
import numpy as np

NUM_ITER = 300
LEARNING_RATES = [0.1, 0.5]
NUM_RANDOM_RESTARTS = 3

def create_eps_ball(inputs, eps):
    lower = inputs - eps
    lower = lower.clamp_(min=0.0, max=1.0)
    upper = inputs + eps
    upper = upper.clamp_(min=0.0, max=1.0)
    return lower, upper


def deeppoly_analyze(net, inputs, eps, true_label):
    
    for lr in LEARNING_RATES:
                
        for epoch in range(NUM_RANDOM_RESTARTS):
                        
            lower, upper = create_eps_ball(inputs, eps)
            
            if epoch == 0:
                
                verifier_network = create_verifier_network(net, inputs, true_label, random_restart=False)
                
            else:
                
                verifier_network = create_verifier_network(net, inputs, true_label, random_restart=True)

            opt = torch.optim.Adam(verifier_network.parameters(), lr=lr)

            for i in range(NUM_ITER):
                
                opt.zero_grad()
                
                x = tf.DeepPoly(lower, upper)

                lower_bound, upper_bound = verifier_network(x)
    
                if (lower_bound > 0).all():
                    '''
                    print('verified, ', "iteration:", i, "learning rate ", lr, " rand restart no ", epoch)
                    with open("../prelim_test_cases/predictions.txt", 'a+') as file:
                        file.write(" Iteration: " + str(i))
                        file.write(" lr: " + str(lr) + " ")
                        file.write("Number of random restarts needed " + str(epoch) + " " )
                    '''
                    return True

                loss = loss_fn3(lower_bound, upper_bound)
                loss.backward()
                opt.step()
            
    
    if i == NUM_ITER - 1:
        return False
    


def loss_fn1(lower_bound, upper_bound):
  
    return torch.max(-lower_bound[lower_bound <= 0.0])


def loss_fn3(lower_bound, upper_bound):
    
    return torch.log(-lower_bound[lower_bound <= 0.0]).max() 

    
def create_verifier_network(net, inputs, true_label, random_restart):
    
    input_features = inputs.shape.numel()
    verifier_layers = []

    for layer in net.layers:

        in_features = verifier_layers[-1].out_features if len(verifier_layers) > 0 else input_features
        
        if isinstance(layer, networks.Normalization):
            verifier_layers.append(tf.Normalization_Transformer(in_features))

        if isinstance(layer, torch.nn.modules.flatten.Flatten):
            verifier_layers.append(tf.Flatten_Transformer(in_features))

        if isinstance(layer, torch.nn.modules.linear.Linear):
            verifier_layers.append(tf.Affine_Transformer(layer))

        if isinstance(layer, networks.SPU):
            verifier_layers.append(tf.SPU_transformer(in_features, random_restart))


    in_features = verifier_layers[-1].out_features
    verifier_layers.append(tf.Final_Verfication_Transformer(in_features, true_label))

    return torch.nn.Sequential(*verifier_layers)


