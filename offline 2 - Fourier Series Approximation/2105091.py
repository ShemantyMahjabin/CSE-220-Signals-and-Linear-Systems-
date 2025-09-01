import numpy as np
import matplotlib.pyplot as plt

class FourierSeries:
    def __init__(self, func, L, terms=10):
        self.func = func
        self.L=L
        self.terms = terms


    def calculate_a0(self, N=1000):
        x=np.linspace(-self.L,self.L,N)
        y=self.func(x)
        integral=np.trapz(y,x)
        a0=1/2*self.L*integral
        return a0 

    def calculate_an(self, n, N=1000):
        x=np.linspace(-self.L,self.L,N)
        y=self.func(x)*np.cos(n*np.pi*x/self.L)
        integral=np.trapz(y,x)
        an=1/self.L*integral
        return an

    def calculate_bn(self, n, N=1000):
        x=np.linspace(-self.L,self.L,N)
        y=self.func(x)*np.sin(n*np.pi*x/self.L)
        integral=np.trapz(y,x)
        bn=1/self.L*integral
        return bn

    def approximate(self, x):
        a0=self.calculate_a0()
        # Initialize the series with the a0 term
        approximation=a0
        # Compute each harmonic up to the specified number of terms
        for n in range(1, self.terms+1):
            an=self.calculate_an(n)
            bn=self.calculate_bn(n)
            approximation+=an*np.cos(n*np.pi*x/self.L) + bn*np.sin(n*np.pi*x/self.L)
        return approximation

       



    def plot(self): 
        x= np.linspace(-self.L,self.L,1000) # Implement this line
        original = self.func(x) # Implement this line
        approximation = self.approximate(x)    # Implement this line

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(x, original, label="Original Function", color="blue")
        plt.plot(x, approximation, label=f"Fourier Series Approximation (N={self.terms})", color="red", linestyle="--")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.title("Fourier Series Approximation")
        plt.grid(True)
        plt.show()


def target_function(x, function_type="square"):
    
    if function_type == "square":
        # Square wave: +1 when sin(x) >= 0, -1 otherwise
        return np.sign(np.sin(x))
    
    elif function_type == "sawtooth":
        # Sawtooth wave: linearly increasing from -1 to 1 over the period

        freq=1/np.pi
        cycles= x*freq
        sawtooth= 2*(cycles-np.floor(cycles+0.5))
        return sawtooth
    
    elif function_type == "triangle":
        # Triangle wave: periodic line with slope +1 and -1 alternately
        freq=1/np.pi
        cycles=freq*x
        triangle=2*np.abs(2*(cycles-np.floor(cycles+0.5)))-1
        return triangle
       
    
    elif function_type == "sine":
        # Pure sine wave

        return np.sin(x)
    
    elif function_type == "cosine":
        # Pure cosine wave

       return np.cos(x)
    
    else:
        raise ValueError("Invalid function_type. Choose from 'square', 'sawtooth', 'triangle', 'sine', or 'cosine'.")

# Example of using these functions in the FourierSeries class
if __name__ == "__main__":
    L = np.pi  # Half-period for all functions
    terms = 10  # Number of terms in Fourier series

    # Test each type of target function
    for function_type in ["square", "sawtooth", "triangle", "sine", "cosine"]:
        print(f"Plotting Fourier series for {function_type} wave:")
        
        # Define the target function dynamically
        fourier_series = FourierSeries(lambda x: target_function(x, function_type=function_type), L, terms)
        
        # Plot the Fourier series approximation
        fourier_series.plot()
