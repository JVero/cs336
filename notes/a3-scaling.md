# Assignment 3: Scaling

Handout: `assignment3-scaling/cs336_assignment3_scaling.pdf`

One heading per graded problem. Written answers, experiment numbers, plots, and things I want to remember go here.

## chinchilla_isoflops

IsoFLOPs scaling laws (5 points)
A.
For 10^23 total flops, you'd want a model with 10^10.959730092241427 or approx 90e9 params
For 10^24 total flops, you'd want a model with 10^11.474309577264806 or approx 300e9 params
The relevant figure is in model_size.png
B. 
For 10^23 total flops, you'd want 10^11.262118657374923 = 1.8e11 tokens
For 10^24 total flops, you'd want 10^11.747539172351544 = 5.6e11 tokens
The relevant figure is in optimal_tokens.png

## scaling_laws

Constructing scaling laws leaderboard (50 points)

# Given your fixed scaling laws budget of 12 B200-hours, how did you decide which runs to query?
I first found the hyperparameters to "set and forget" (lr and batch size specifically had the best, most stable runs at 0.002 and 32, respectively)

For IsoFLOP, I ran the following shape combinations, with the corresponding tuple being (d_model, n_hidden)
3M (256, 4)
6M (512, 2) <- this was dropped from the parabola fit, as 2 layers didn't give a good loss
12M(512, 4)
25M(512, 8)
100M (1024, 8)
tokens = C / (6N), truncated to the nearest multiple of 262,144 which only caused a 2% deviation from the raw value

All the configs above are n_heads = 8 and widths a multiple of 128 for kernel efficiency, d_ff computed like in A1 (8/3 * d_model, rounded to a multiple of 128). I used a big chunk of my $30 free Modal credit, independent of the assignment budget. The assignment budget I used 20,000s of the 43,200s budget.

# How did you fit your scaling law? Describe the concrete method or methods you used. In particular, it will likely to be useful to familiarize yourself with the approaches used in J. Kaplan et al. [1] and J. Hoffmann et al. [2]. 
During IsoFLOP, you can fit parabolas at constant compute for log10 N (base of log isn't important, due to change-of-base). For a constant compute, the fit of N_opt yields a power of C^0.58

There are a few scaling laws:
L(N, D) = 2.86 + 277.5 * N^-0.38 + 2720 * D^-0.45, given the runs I did I fit the curve given L, N, and D
For the next 2, I substituted for D using D = C/(6N), set dL/dN = 0, and solved for N, which also gives D, implicitly
N = 0.02 * C^0.54, which is close to the 0.58 exponent during the isoflop fit
D = 8.44 * C^0.46
# How well does your scaling law fit the experimental data?
R^2 = 0.997
best-N line for log-log = 0.999

We measured tok/s and flops/s at 2048 d_model and got approximately 586 TFLOPS/s. Multiplying that by 48 hours you get 1e20 = C
Using the equations above:
N = 1.26B parameters
D = 13.3B tokens

Plugging those values into L(N, D) predicts a loss of 3.0313
# Hyperparameters for the predicted model. 
shape: (2048, 25) <- given the scaling law above, at a generous FLOPS count of 586, the optimal N is 1.26B and D = 8.44 * (1e20)^0.46 = 13.3B tokens
N is depenedent on a quantum decision of n_hidden_layers. (2048, 25) also yields 1.26B parameters, within rounding. D also, practically needs to be rounded to a multiple of 262,144.
heads = 16 <- changed to a multiple of 8 from 7 because of kernel efficiency, then scaled up to 16 to match the constraints of the model's d_head <= 128 constraint
d_ff = round 8/3 * 2048 to next multiple of 128
batch = 32 <- Basically just swept in the smaller models, didn't scale it up for larger models honestly because I didn't feel like juggling the memory constraints
lr = 0.002, based on the above grid search which specifies this scaling law

# Figures. None referenced yet. You have the loss-vs-N and best-N-vs-C plots in scratch/.
I really don't feel like any of the plots matter right now.