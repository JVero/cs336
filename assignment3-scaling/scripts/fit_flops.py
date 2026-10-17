import numpy as np
import io
import pandas as pd

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

data_str = """budget,N,width,depth,tokens,C_actual,loss
5e+15,3145728,256,4,264503296,4.992e+15,4.1982421875
5e+15,6291456,512,2,132382720,4.997e+15,4.326171875
5e+15,12582912,512,4,66060288,4.987e+15,4.2041015625
5e+15,25165824,512,8,33030144,4.987e+15,4.42578125
5e+15,100663296,1024,8,8126464,4.908e+15,5.1328125
1e+16,3145728,256,4,529006592,9.985e+15,4.099609375
1e+16,6291456,512,2,264765440,9.995e+15,4.2109375
1e+16,12582912,512,4,132382720,9.995e+15,3.97265625
1e+16,25165824,512,8,66060288,9.975e+15,4.0947265625
1e+16,100663296,1024,8,16515072,9.975e+15,4.5625
1e+16,100663296,1024,8,16515072,9.975e+15,4.591796875
5e+16,3145728,256,4,2645295104,4.993e+16,3.97265625
5e+16,6291456,512,2,1324613632,5e+16,4.0556640625
5e+16,12582912,512,4,662175744,4.999e+16,3.6796875
5e+16,25165824,512,8,330825728,4.995e+16,3.669921875
5e+16,100663296,1024,8,82575360,4.987e+16,3.828125
1e+17,3145728,256,4,5290852352,9.986e+16,3.9560546875
1e+17,6291456,512,2,2649489408,1e+17,4.015625
1e+17,12582912,512,4,1324613632,1e+17,3.61328125
1e+17,25165824,512,8,661913600,9.995e+16,3.5634765625
1e+17,100663296,1024,8,165412864,9.991e+16,3.638671875"""

if __name__ == "__main__":

    df = pd.read_csv(io.StringIO(data_str))
    df = df[df.depth != 2] 
    df = df.groupby(["budget", "N", "tokens"], as_index=False).loss.mean() 
    compute = []
    best_params = []
    losses = []
    for budget, g in df.groupby("budget"):
        x: pd.Series = g["N"]
        y: pd.Series = g["loss"]
        

        parabola_coeffs = np.polyfit(np.log10(x), y, deg=2)
        parabola = np.poly1d(parabola_coeffs)
        n_opt = int(10**(-parabola_coeffs[1]/(2*parabola_coeffs[0])))
        loss_opt = parabola(np.log10(n_opt))
        # print(f"Optimal N for budget {budget:1e}: {n_opt}")
    
        best_params.append(n_opt)
        compute.append(budget)
        losses.append(loss_opt)
    
    # fit optimal model for best_params vs compute
    powerlaw_fit = np.polyfit(np.log10(compute), np.log10(best_params), deg=1)
    # print(powerlaw_fit)
    # print(best_params, losses)
    # [5826013, 9208397, 23071170, 33291001] 
    # [4.15, 3.98, 3.66, 3.55]
    losses_coeffs = np.polyfit(np.log10(best_params), np.log10(losses), deg=1)
    # print(losses_coeffs)
    
    def f(x, e, A, alpha, B, beta):           # the model: first arg is the input, the rest are unknowns
        return e + A * x[0]**-alpha + B * x[1] ** -beta
    
    rows = []
    fit_losses = []
    for _, row in df.iterrows():
        rows.append((row["N"], row["tokens"]))
        fit_losses.append(row["loss"])
    rows = np.array(rows).T
    print(rows)
    params, z = curve_fit(f, rows, fit_losses, p0=[1, 1, 1, 1, 1])
    e, A, alpha, B, beta = params
    print(z)
    print(params)
    plt.semilogx(best_params, losses)
    plt.show()
    
import numpy as np

