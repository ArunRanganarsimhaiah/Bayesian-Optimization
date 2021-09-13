import numpy as np
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):
    ax = plt.gcf().add_subplot(1, 1, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.coolwarm,label="DOE samples")
    ax.set_title(title)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("V0")
    ax.set_zlabel("Y")
    ax.legend()
def test_simulator():
    """Use this function to test 'simulator' and 'evaluate' functions"""

    alpha_test = 0.5
    V_0_test = 25.0
    x_target = 24.0
    #score, x_path, y_path = simulator(alpha=alpha_test, V_0=V_0_test, x_target=x_target)
    score, x_path, y_path = simulator(alpha=alpha_test, V_0=V_0_test) # you don't have to specify x_target

    plt.figure(figsize=(8,6))
    plt.plot(x_path, y_path, ".-", label="Ball Path")
    plt.scatter([0.], [0.], marker="o", s=200, label="Start")
    plt.scatter([x_target], [0.], marker="X", s=200, label="Target")
    plt.gca().set_xlabel("X")
    plt.gca().set_xlabel("Y")
    plt.gca().set_title(f"Score: {score}")
    plt.legend()
    plt.show()

    # the function 'evaluate' can be used to compute the output score for multiple inputs (alpha_i, V_0_i)
    X_scaled = np.array([[alpha_test, V_0_test]])
    Y = evaluate(X_scaled)    
    print("X_scaled:", X_scaled)
    print("Y:", Y)

    X_scaled = np.array([[0.3, 10.], [0.5, 15.], [0.7, 20]])
    Y = evaluate(X_scaled)
    print("X_scaled:", X_scaled)
    print("Y:", Y)


def simulator(    
    alpha: float = 0.785, 
    V_0: float = 5., 
    v_wind: float = -1.5, 
    x_target: float = 24.0, 
    dt: float = 1e-04, 
    t_max: float = 10., 
    c_d: float = 0.54, 
    c_d_var: float = 0.1,   # noise on the output. You can change this to 0 (no noise)
    A_ref: float = 0.045, 
    rho: float = 1.225, 
    m: float = 0.6, 
    g: float = 9.81
    ):
    """
    This function computes the path of one ball with the parameters alpha and V_0
    The other parameters don't need to be set.

    call it as 'f(alpha, V_0)'

    input parameters:
        - alpha:    ball throwing angle
        - V_0:      initial velocity of the ball

    returns:
        - obj_f:    squared distance to the target
        - x[:i+1]:  x postions of the ball at all timesteps before it hits the ground  
        - y[:i+1]:  y postions of the ball at all timesteps before it hits the ground  
    """
    
    # make sure the input has the right values
    assert dt >= 1e-06, "choose a bigger dt"
    assert dt <= 1e-02, "choose a smaller dt"
    assert alpha >= 0., "choose a bigger alpha"
    assert alpha <= np.pi, "choose a smaller alpha"
    assert V_0 >= 0., "choose a bigger V_0"
    assert V_0 <= 200., "choose a smaller V_0"
    
    N_steps = int(np.ceil(t_max / dt)) + 10
    #print(N_steps)
    x = np.empty(N_steps)
    y = np.empty(N_steps)
    v_x = np.empty(N_steps)
    v_y = np.empty(N_steps)
    a_x = np.empty(N_steps)
    a_y = np.empty(N_steps)       
    
    # initial conditions
    x[0] = 0.
    y[0] = 0.
    v_x[0] = V_0*np.cos(alpha)
    v_y[0] = V_0*np.sin(alpha)
    a_x[0] = 0.
    a_y[0] = 0.
    
    i = 0
    
    while i < N_steps - 10:
        v_mag = np.sqrt((v_x[i] - v_wind)**2 + v_y[i]**2)
        F_w_x = -0.5*rho*A_ref*v_mag*(v_x[i] - v_wind)*c_d*np.exp(np.random.normal(0, c_d_var))
        F_w_y = -0.5*rho*A_ref*v_mag*(v_y[i])*c_d*np.exp(np.random.normal(0, c_d_var))
        a_x[i+1] = F_w_x/m
        a_y[i+1] = F_w_y/m - g
        v_x[i+1] = v_x[i] + a_x[i+1]*dt
        v_y[i+1] = v_y[i] + a_y[i+1]*dt
        x[i+1] = x[i] + v_x[i+1]*dt
        y[i+1] = y[i] + v_y[i+1]*dt
        
        #print(F_w_x, F_w_y, x[i+1], y[i+1])
        i = i + 1
        
        if y[i] <= -1.e-06 and v_y[i] < 0.:
            break
                    
    
    obj_f = np.min([(x[i] - x_target)**2, (x[i-1] - x_target)**2])
    
    return obj_f, x[:i+1], y[:i+1]


def evaluate(X, **kwargs):
    """
    This function can be used to compute the output of simulator for a vector of multiple input settings (alpha, V_0)
    """
    n_samples, n_dim = X.shape
    
    n = np.sum([1 for _, v in kwargs.items() if v==True])
    if n == 0 and n_dim == 1:
        kwargs = {"alpha": True}
    elif n == 0 and n_dim == 2:
        kwargs = {"alpha": True, "V_0": True}
    elif n == n_dim:
        #print("okay")
        pass
    else:
        raise Exception("bad input -", "n_dim:", n_dim, " n:", n)
    
    F = np.empty(n_samples)
    for i in range(n_samples):
        sim_kwargs = {}
        for j, (k, v) in enumerate(kwargs.items()):
            if type(v) == float:
                sim_kwargs[k] = v
            elif v:
                sim_kwargs[k] = X[i,j]
        f, _, _ = simulator(**sim_kwargs)
        F[i] = f
    
    return F[:,np.newaxis]


 
def get_doe_samples(num_doe_points):
    """
    Get a number of DoE (Design of Experiment) samples
    """
    np.random.seed(seed=3)
    X = np.random.rand(num_doe_points,2)
    return X


class RBF_Kernel:
    """
    This class can be used organize the functions related to the kernel.
    """

    def __init__(self, ndim=2, logTheta=0, logRegPara=None) -> None:
        """
        Initialize the class parameters.
        Theta and the regularization parameter are stored as their log-value
        """

        self.ndim = ndim
        self.logTheta = logTheta
        self.logRegPara = logRegPara
        self.presolved = False
    
    def get_theta(self):
        """
        Getter for theta
        """
        return np.power(10, self.logTheta)
    
    def get_regPara(self):
        """
        Getter for the regularization parameter
        """
        return np.power(10, self.logRegPara)
    
    def kernel_func(self, x1, x2, theta, regPara, p=2):
        """
        Compute the kernel function between two (2D) points x1 and x2
        """

        if type(x1) is np.ndarray:        
            assert len(x1) == len(theta)
            assert type(x2) is np.ndarray
            assert x1.shape == x2.shape

            d = len(x1) # dimesionality of the input x
            cor = np.sum( [ float(theta[i]) * np.power( float(x1[i]-x2[i]), p ) for i in range(d) ] )
        else:
            assert type(x1) is float
            assert type(theta) is float
            d = 1
            cor = float(theta) * np.power( float(x1-x2), p )
        
        cor = np.exp(-float(cor))
        
        return cor

    def get_correlation_matrix(self, x, logTheta=1., logRegPara=None):
        """
        Returns the full covariance matrix of input data x
        """
        # reg has to be None or float.

        theta = np.power(10, logTheta)
        if logRegPara is not None:
            regPara = np.power(10, logRegPara)
        else:
            regPara = 0




        N,D = x.shape

        K = np.empty((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                K[i,j] = self.kernel_func(x[i], x[j], theta, regPara)
            K[i,i] += regPara

        return K
    
    def set_hyperparameters(self, hyperparameters, regularization, log=True):
        """
        Setter for theta and the regularization parameter.
        log=True if the hyperparameters are given as logs
        """

        if log:
            if regularization:
                self.logTheta   = hyperparameters[:-1]
                self.logRegPara = hyperparameters[-1]     # order to define this first
            else:
                self.logTheta = hyperparameters
                self.logRegPara = None
        else:
            if regularization:
                self.logTheta = np.log10(hyperparameters[:-1])
                self.logRegPara = np.log10(hyperparameters[-1])     # order to define this first
            else:
                self.logTheta = np.log10(hyperparameters)
                self.logRegPara = None
        
        self.regularization = regularization

        self.presolved = False
    
    def get_hyperparameters(self, log=True):
        """
        Getter for theta and the regularization parameter.
        log=True if the hyperparameters should be given as logs
        """
        hyperparameters = np.atleast_1d(self.logTheta)
        if self.regularization:
            hyperparameters = np.concatenate((hyperparameters, np.atleast_1d(self.logRegPara)))        
        if not log:
            hyperparameters = np.power(10, hyperparameters)
        
        return hyperparameters


    def calculate_likelihood(self, X, Y):
        """
        This function is supposed to calculate the log-likelihood (without the constant terms) of the data y at position(s) x 
        """
        # N: sample size
        # D: dimension
        N, D = X.shape

        K = self.get_correlation_matrix(X, self.logTheta, self.logRegPara)       # no info of y is used.
        K = K + 0.001* np.eye(N)
        #LU decomposition to compute the inverse and the determinant of "K"
        P, L, U = scipy.linalg.lu(K)
        self.Inv_of_K = np.linalg.inv(U) @ np.linalg.inv(L) @ np.linalg.inv(P)
        self.variance = (Y.T @ self.Inv_of_K @ Y) / N #variance
        self.Det_of_K = np.linalg.det(P) * np.linalg.det(L) * np.linalg.det(U)
        #Likelihood function E
        E = N/2 * np.log(self.variance) + 0.5 * np.log(self.Det_of_K)

#        if E != E:      
        if np.isnan(E):     # when E is NaN due to violation of the computations
            return 1.e6
        return E
    
    def likelihood_opt_wrapper_func(self, hyperparameters, X, Y, regularization):
        """
        This function can be used to wrap the likelihood calculation function such that it can be used by scipy.optimize
        """

        self.set_hyperparameters(hyperparameters, regularization)
        E = self.calculate_likelihood(X, Y)

        return E

def optimize_kernel_hyperparameters(kernel: RBF_Kernel, X, Y, regularization=True, bounds=None, maxiter=100, tol=0.001, disp=False, restart=False):
    """
    Optimize the kernel hyperparameters
    """

    args = (X, Y, regularization)

    if bounds is None:
        bounds_theta = np.array( [ [-6, 5],  [-6, 5] ] )
        bounds_reg   = np.array( [ [-12, 0] ] )

        if regularization:
            bounds = np.vstack((bounds_theta, bounds_reg))
        else:
            bounds = bounds_theta
        print("optimization bounds:", bounds)

    if restart:
        x0 = kernel.get_hyperparameters()
    else:
        x0 = None
    
    opt = scipy.optimize.differential_evolution(kernel.likelihood_opt_wrapper_func, bounds=bounds, args=args, maxiter=maxiter, tol=tol, disp=disp)#, x0=x0)
    #print(opt)

    kernel.set_hyperparameters(opt.x, regularization=regularization)

    return kernel

def predict_output(xpre, X, Y, kernel: RBF_Kernel):
    """
    Compute the predictive distribution, given some input data (X,Y) and a kernel
    """

    N, D_input = X.shape
    Npre, _ = xpre.shape

    ypre = np.empty(Npre, dtype=np.float64)    # muN
    yvar = np.empty(Npre, dtype=np.float64)    # sigmaSquaredN

    theta = kernel.get_theta()
    regPara = kernel.get_regPara()

    for n in range(Npre):
        k = np.empty(N      , dtype=np.float64)
        for i in range(N):
            k[i] = kernel.kernel_func(xpre[n], X[i], theta, regPara)        # so, switch-off the reg value because it is always for "new" prediction.

        ypre[n] = k.T @ kernel.Inv_of_K @ Y
        yvar[n] = kernel.variance * (1 - k.T @ kernel.Inv_of_K @ k )

    return ypre, yvar

def scale_input(X, alpha_min=0.15, alpha_max=1.5, V_0_min=5., V_0_max=40.):
    """
    Scale the input from values in [0, 1) to [alpha_min, alpha_max) and [V_0_min, V_0_max)
    """

    X_scaled = np.copy(X)
    X_scaled[:,0] = X_scaled[:,0]*(alpha_max - alpha_min) + alpha_min
    X_scaled[:,1] = X_scaled[:,1]*(V_0_max - V_0_min) + V_0_min

    return X_scaled

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    
    mu, sigma = predict_output(X, X_sample, Y_sample, gpr)
    
    sigma = np.sqrt(sigma)
    sigma = sigma.reshape(-1, 1)
    mu_opt, _ = predict_output(X_sample, X_sample, Y_sample, gpr)
    mu_sample_opt = np.min(mu_opt)
    
    with np.errstate(divide='warn'):
        imp = mu-mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei  
    

def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    #Proposes the next sampling point by optimizing the acquisition function.
    #Returns:
        #Location of the acquisition function maximum.
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)
    
    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = scipy.optimize.minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')        
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x           
            
    return np.array([min_x])


    
    
def main(n_doe_samples=30, maxiter_bayesopt=10):
    """
    This is the main function, where the main program steps need to be implemented
    """

    # set the interval of suitable (alpha, V_0) values
    alpha_max, alpha_min = 1.5, 0.15
    V_0_max, V_0_min = 40, 5

    # get input data (X,Y)
    X = get_doe_samples(n_doe_samples)
    X_scaled = scale_input(X, alpha_min, alpha_max, V_0_min, V_0_max)
    Y = evaluate(X_scaled)


    regularization = True         # boolean
    # "regularization" is a flag to indicate if we add a new hyparameter "sigma" in the slide 32 at Lecture 8. The result of the "regularization" True and False can be seen on the slide 33 and slide 30, respectively.
    
    # initialize the kernel and optimize its hyperparameters
    kernel = RBF_Kernel()

    ### Test an arbitrary theta ###
    theta_test = [-6, 5]   # please input your
    regPara_test = -1.   # You do not need to change this.

    if regularization:
        hyperparameter_test = theta_test + [regPara_test]
    else:
        hyperparameter_test = theta_test
    kernel.set_hyperparameters(hyperparameters=hyperparameter_test, regularization=regularization, log=True)         # a value from -6. to 5.
    
    #Xpre = np.array([[0.2514, 0.5014]])

    ### Determine the hyperparameter theta by MLE ###
    kernel = optimize_kernel_hyperparameters(kernel, X, Y, regularization=regularization, maxiter=100, disp=True)
    #create a meshgrid for Xpre
    rx, ry = np.arange(0, 1, 0.01), np.arange(0, 1, 0.01)
    gx, gy = np.meshgrid(rx, rx)

    X_pre = np.c_[gx.ravel(), gy.ravel()]
    X_2D=X_pre
    # predict the output at Xpre
    ypre, yvar = predict_output(X_2D, X, Y, kernel)
    #the global minimum y_min with the corresponding input X_opt
    y_min=min(ypre)
    print(y_min)
    index=np.argmin(ypre)
    Y_var_min=yvar[index]
    print("Y_var=",Y_var_min)
    X_opt=X_2D[index]
    X_opt=X_opt.reshape(-1,2)
    X_opt_sc=scale_input(X_opt,alpha_min,alpha_max,V_0_min,V_0_max)
    print(X_opt_sc)
    #plot using simulator
    score, x_path, y_path = simulator(alpha=X_opt_sc[0, 0], V_0=X_opt_sc[0, 1], x_target=24.0)
    plt.figure(figsize=(8, 6))
    plt.plot(x_path, y_path, ".-", label="Ball Path")
    plt.scatter([0.], [0.], marker="o", s=200, label="Start")
    plt.scatter([24.0], [0.], marker="X", s=200, label="Target")
    plt.gca().set_xlabel("X")
    plt.gca().set_xlabel("Y")
    plt.gca().set_title(f"Score: {score}")
    plt.legend()
    plt.show()
    print('X_opt= ',X_opt)
    #visualize the predicted mean on the input space 𝐗 as a 3D graph.
    plot_gp_2D(gx, gy,ypre, X, Y,'N=50',1)

    plt.show()


    
    # copy the DoE data for the optimization loop
    Xupdated = np.copy(X)
    Yupdated = np.copy(Y)
    
    # bayesian optimization loop

    boundsB = np.array(([0,1], [0,1])) 
    for i in range(maxiter_bayesopt):

        print("BayesOpt Iter", i)
        #calculate the acquisition function and find the next value by optimizing the input argument of the acquisition function
        Xnew = propose_location(expected_improvement, Xupdated, Yupdated, kernel, boundsB)
        Xnew = np.atleast_2d(Xnew)
        Xreal = scale_input(Xnew, alpha_min, alpha_max, V_0_min, V_0_max)
        Ynew = evaluate(Xreal)

        Xupdated = np.vstack((Xupdated, Xnew))          # updated data X
        Yupdated = np.vstack((Yupdated, Ynew))          # updated data Y
        
        kernel = optimize_kernel_hyperparameters(kernel, Xupdated, Yupdated, regularization=regularization, maxiter=100, disp=True)

    # Optimizing again with the improved model
    yopt, yvar = predict_output(X_2D, Xupdated, Yupdated, kernel)
    # the global minimum y_min with the corresponding input X_opt
    y_min = min((yopt))
    print(y_min)
    index = np.argmin((yopt))
    X_opt = X_2D[index]
    Y_var_min = yvar[index]
    print("Y_var=", Y_var_min)
    X_opt = X_opt.reshape(-1, 2)
    X_opt_sc = scale_input(X_opt, alpha_min, alpha_max, V_0_min, V_0_max)
    print(X_opt_sc)
    # plot using simulator
    score, x_path, y_path = simulator(alpha=X_opt_sc[0, 0], V_0=X_opt_sc[0, 1], x_target=24.0)
    plt.figure(figsize=(8, 6))
    plt.plot(x_path, y_path, ".-", label="Ball Path")
    plt.scatter([0.], [0.], marker="o", s=200, label="Start")
    plt.scatter([24.0], [0.], marker="X", s=200, label="Target")
    plt.gca().set_xlabel("X")
    plt.gca().set_xlabel("Y")
    plt.gca().set_title(f"Score: {score}")
    plt.legend()
    plt.show()
    # visualize the predicted mean on the input space 𝐗 as a 3D graph.
    plot_gp_2D(gx, gy, yopt, Xupdated, Yupdated, 'N=30+10', 1)
    plt.show()
    print(X_opt)







    
# this condition will be true, if you call this program as 'python exam_task1.py'
if __name__ == "__main__":

    # test and try to understand the simulator
    #test_simulator() # you can delete/comment this line, if you don't need it anymore

    # execute the main program
    main()