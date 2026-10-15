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


