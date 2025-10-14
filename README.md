# Hybrid Actor-Critic Meta-Learning for Sample-Efficient Task Adaptation in Energy Management Systems




Original REPTILE update is without an optimizer is: 

\phi = \phi + \epsilon(\phi' - \phi)
     = (1 - \epsilon) \phi + \epsilon \phi',

with \phi the meta-parameters and \phi' the inner-loop task optimized parameters.
So here we see that the deltas are accumulated using (\phi' - \phi).


Using (\phi - \phi') as gradient we can use an optimizer to update the meta weights instead of manually modifying the weights:

Optimizers update parameters using gradient g as: \phi = \phi - \epsilon * g,

pluging our gradient we get:

\phi = \phi - \epsilon(\phi - \phi')
     = (1 - \epsilon) \phi + \epsilon \phi'.

We get the same exact update but we can now use any optimizers like Adam for example.
Using this method the deltas are accululated as (\phi - \phi').

We can still accumulate the gradient as (\phi' - \phi) by modifying the sign to keep the same logic for both methods:

\phi = \phi - ( -\epsilon(\phi' - \phi) )
     = \phi + \epsilon \phi' - \espilon \phi
     = (1 - \epsilon) \phi + \epsilon \phi'.


