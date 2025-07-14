from scipy.optimize import minimize, curve_fit
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class BasicGP: # this is the "model"
    
    def __init__(self,
                 kernel='squared_exp',
                 signal_var=1,
                 lengthscale=1,
                 mu_init_method='random'):
        """Approximates an unknown latent reward function using a Gaussian
        process using a provided objective function.
        
        sigma: signal variance (dictates expected variation in the latent reward)
        l: lengthscale (dictates the smoothness of the latent reward)"""
        self.signal_var = signal_var
        self.lengthscale = lengthscale
        self.mu_init_method = mu_init_method
        self.mu = None
        self.f = None
        if kernel == 'squared_exp':
            self.kernel = self.squared_exp_kernel
        
    def setup(self, action_space, likelihood, dlikelihood=None, d2likelihood=None):
        self.actions = action_space
        self.cov = self.prior_cov()
        np.fill_diagonal(self.cov, np.diagonal(self.cov) + 1e-5)
        print('here2', self.cov.shape)
        self.cov_inv = np.linalg.inv(self.cov)
        if self.mu_init_method == 'random':
            self.mu = 2 * np.random.random(self.actions.shape[0]) - 1
        else:
            raise Exception(f'Invalid f init method: {self.mu_init_method}')
        
        self.likelihood = likelihood
        self.dlikelihood = dlikelihood
        self.d2likelihood = d2likelihood
        self.objective = lambda r: self.likelihood(r) + 0.5 * r.reshape(1,-1) @ self.cov_inv @ r.reshape((-1, 1)) #+ 0.5 * np.log(np.linalg.det(self.cov))
        self.jacobian = None
        self.hessian = None
        if dlikelihood:
            self.jacobian = lambda r: self.dlikelihood(r) + self.cov_inv @ r.reshape((-1, 1)).flatten()
        if d2likelihood:
            self.hessian = lambda r: self.d2likelihood(r) + self.cov_inv
    
    def squared_exp_kernel(self, X):
        """Computes the squared exponential kernel
        
        X: input data (list of vectors)
        sigma: signal variance
        l: lengthscale"""
        
        # K = np.zeros((X.shape[0], X.shape[0]))
        # for i, x1 in enumerate(X):
        #     for j, x2 in enumerate(X):
        #         r = np.linalg.norm(x1 - x2)
        #         K[i,j] = self.signal_var**2 * np.exp(-r**2 / (2 * self.lengthscale**2))
        # Compute the squared distances between points
        sqdist = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
        
        # Compute the kernel matrix
        K = self.signal_var**2 * np.exp(-0.5 * sqdist / self.lengthscale**2)
        return K
        return K
        
    def prior_cov(self):
        """Computes the prior covariance matrix (uses squared exponential
        kernel)
        
        actions: list of actions"""
        
        return self.kernel(self.actions)
    
    def prior(self, actions, r):
        """Gaussian prior for the action dataset
        
        actions: list of actions
        r: vectorized user latent reward function"""
        
        SIGMA = self.prior_cov(actions)
        one = 1 / ((2 * np.pi)**(cardA/2) * np.linalg.det(SIGMA)**0.5)
        two = np.exp(-0.5 * r.T @ np.linalg.inv(SIGMA) @ r)
        return one * two


    def fit(self, **kwargs):
        """Fits the Gaussian process to the user feedback
        
        actions: list of actions
        D: user feedback"""
        
        res = minimize(self.objective,
                       self.mu,
                       jac=self.jacobian if self.jacobian else None,
                       hess=self.hessian if self.hessian else None,
                       **kwargs)
        self.mu = res.x
        # print(res)
        return self.mu
    
    def polyfunc(self, x, *args):
        return Polynomial([*args])(x)
    
    def functionalize(self, degree=2):
        """Functionalizes the Gaussian process"""
        
        # self.f = lambda x: self.mu.reshape(1,-1) @ self.kernel(x)
        # popt, pcov = curve_fit(lambda *args: Polynomial([*args]), self.actions, self.mu)
        # poly = np.polyfit(self.actions, self.mu, 3)
        # self.f = poly
        # return self.f
        
        self.poly = PolynomialFeatures(degree=degree)
        X_poly = self.poly.fit_transform(self.actions)
        model = LinearRegression()
        model.fit(X_poly, self.mu)
        
        # x, y = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
        # x, y = self.actions.reshape
        X_grid = self.actions
        y_pred = model.predict(self.poly.transform(X_grid))
        self.f = lambda x: model.predict(self.poly.transform(x))
        return y_pred.reshape(len(self.actions))