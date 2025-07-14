import numpy as np

class PreferenceBasedLearning:
    
    def __init__(self, 
                 low: list[float] | np.ndarray,
                 high: list[float] | np.ndarray,
                 action_dims: list[float] | np.ndarray,
                 preference_noise: int=0.01,
                 coactive_noise: int=0.01,
                 ordinal_noise: int=0.01):
        """Sets up preference based learning based on POLAR (Tucker et. al). 
        This class is designed to organize the necessary likelihood functions
        and other variables that POLAR uses to learn preferences. Please see
        the type hints above for what each argument should be.
        
        low: the lower bound of the action parameters
        high: the upper bound of the action parameters
        action_dim: the dimension of each action component (discretization)
        preference_noise: (default=0.01) c_p, noisiness of preference feedback
        coactive_noise: (default=0.01) c_c, noisiness of coactive feedback
        ordinal_noise: (defualt=0.01) c_0, noisiness of ordinal feedback"""

        # action space setup
        self.action_space = None
        self.low          = low
        self.high         = high
        self.action_dims  = action_dims
        self.step         = (self.high - self.low) / (self.action_dims - 1)
        self.generate_action_space()
        
        # feedback setup
        self.preference_noise = preference_noise
        self.coactive_noise   = coactive_noise
        self.ordinal_noise    = ordinal_noise
        self.preference_fbk = []
        self.coactive_fbk = []
        self.ordinal_fbk = []
        
    def add_feedback(self,
                     preference: tuple[list[float] | np.ndarray, list[float] | np.ndarray, bool],
                     coactive: tuple[list[float] | np.ndarray, list[float] | np.ndarray, bool],
                     ordinal: tuple[list[float], float, float]) -> None:
        """Adds feedback to the PBL class in the form of (pairwise) preference,
        coactive, and ordinal.
        
        preference: (curr, prev, p) where curr, prev are actions and p is
                    whether curr is preferred to prev
        coactive:   (a_bar, curr, c) where a_bar, prev are actions and c is
                    whether a_bar is preferred to curr
        ordinal:    (curr, b0, b1) where curr is an action and b0, b1 are a
                    bucket to fit r[a] to"""
        if preference is not None:
            (curr, prev, p) = preference
            if not p:
                self.preference_fbk.append([self.get_idx(prev), self.get_idx(curr)])
            else:
                self.preference_fbk.append([self.get_idx(curr), self.get_idx(prev)])
                
        if coactive is not None:
            (a_bar, curr, c) = coactive
            if not c:
                self.coactive_fbk.append([self.get_idx(curr), self.get_idx(a_bar)])
            else:
                self.coactive_fbk.append([self.get_idx(a_bar), self.get_idx(curr)])
        
        if ordinal is not None:
            (curr, b0, b1) = ordinal
            self.ordinal_fbk.append([self.get_idx(curr), b0, b1])

        
    def generate_action_space(self) -> None:
        """Generates an action space given initial parameters (see __init__)"""
        action_dims = []
        for _start, _stop, _num in zip(self.low, self.high, self.action_dims):
            action_dims.append(np.linspace(start=_start, stop=_stop, num=_num))
        
        # Generate the grid
        action_space = np.array(np.meshgrid(*action_dims))
        self.action_space = action_space.reshape(len(action_space), -1).T
        
    def get_idx(self, action) -> int:
        """Gets the closest index corresponding to a particular action in the
        generated action space.
        
        action: the action to find the index of
        
        Returns the index of the action input"""
        ret = 0
        for idx, (a, low, disc) in enumerate(zip(action, self.low, self.step)):
            ret += round((a - low) / disc) * np.prod(self.action_dims[:idx])
        return int(ret)
    
    @classmethod
    def sigmoid(cls, x: float | list[float] | np.ndarray) -> float | np.ndarray:
        """Sigmoid function
        x: a (numerical) input to the sigmoid function
        Returns the sigmoid of the input"""
        x = np.minimum(100, np.maximum(-100, x)) # remove overflow
        return 1 / (1 + np.exp(-x))
    
    @classmethod
    def dsigmoid(cls, x: float | list[float] | np.ndarray) -> float | np.ndarray:
        """First derivative of sigmoid function
        x: a (numerical) input to the derivative sigmoid function
        Returns the derivative of the sigmoid of the input
        """
        return cls.sigmoid(x) * (1 - cls.sigmoid(x))
    
    @classmethod
    def d2sigmoid(cls, x: float | list[float] | np.ndarray) -> float | np.ndarray:
        """Second derivative of sigmoid function
        x: a (numerical) input to the 2nd derivative sigmoid function
        Returns the 2nd derivative of the sigmoid of the input
        """
        return cls.sigmoid(x) * (1 - cls.sigmoid(x)) * (1 - 2 * cls.sigmoid(x))
    
    def pairwise_likelihood(self,
                            r: list[float] | np.ndarray,
                            a1: list[float] | np.ndarray,
                            a2: list[float] | np.ndarray,
                            noise: float=0) -> float | np.ndarray:
        """Computes the likelihood of the user preferring action a1 over a2. 
        This is relevant both the preference and coactive likelihoods.
        
        r: user's latent reward function (discretized)
        a1: action 1 (can be np.ndarray of actions)
        a2: action 2 (can be np.ndarray of actions)
        noise: noise in the user's feedback (default 0)
        
        Returns a float if a1, a2 are 1D lists/arrays and an np.ndarray if a1,
        a2 are np.ndarrays"""
        ret = 0
        if noise != 0:
            ret = self.sigmoid((r[a1] - r[a2]) / noise) #+ 0.01
        else:
            ret = 1 if r[a1] >= r[a2] else 0.01
        return -np.log(ret)
    
    def preference_likelihood(self, r: np.ndarray):
        data = np.array(self.preference_fbk)
        a1 = data[:, 0]
        a2 = data[:, 1]
        sum = np.sum(self.pairwise_likelihood(r, a1, a2, noise=self.preference_noise))
        return sum
    
    def coactive_likelihood(self, r):
        ret = 0
        for a1, a2 in self.coactive_fbk:
            ret += self.pairwise_likelihood(r, a1, a2, noise=self.coactive_noise)
        return ret
    
    def ordinal_likelihood(self, r):
        ret = 0
        for a, b0, b1 in self.ordinal_fbk:
            temp = self.sigmoid((b1 - r[a]) / self.ordinal_noise)
            temp -= self.sigmoid((b0 - r[a]) / self.ordinal_noise)
            ret += -np.log(temp + 1e-8)
        return ret

    def overall_likelihood(self, r):
        ret = 0
        if self.preference_fbk and len(self.preference_fbk) > 0:
            ret += self.preference_likelihood(r)
        
        if self.coactive_fbk and len(self.coactive_fbk) > 0:
            ret += self.coactive_likelihood(r)
        
        if self.ordinal_fbk and len(self.ordinal_fbk) > 0:
            ret += self.ordinal_likelihood(r)
            
        return ret
    
    def pairwise_jacobian(self, r, data, noise) -> np.ndarray:
        """Computes the jacobian of the pairwise likelihood function. Applicable
        to preference and coactive feedback.
        
        r: user's latent reward function (discretized)
        data: the pairwise feedback data
        Returns the jacobian of the pairwise likelihood function"""
        # grad = np.zeros(self.action_space.shape[0])
        # zk = (r[data[:,0]] - r[data[:,1]]) / noise
        # szk = -1 / noise * self.dsigmoid(zk) / self.sigmoid(zk)
        # grad = np.bincount(data[:,0], weights=szk, minlength=self.action_space.shape[0])
        # grad -= np.bincount(data[:,1], weights=szk, minlength=self.action_space.shape[0])
        # grad = np.zeros(self.action_space.shape[0])
        # grad[data[:,0]] -= szk
        # grad[data[:,1]] += szk
        # return grad
        
        
        grad = np.zeros(self.action_space.shape[0])
        zk = (r[data[:,0]] - r[data[:,1]]) / noise
        szk = -1 / noise * self.dsigmoid(zk) / self.sigmoid(zk)
        grad += np.bincount(data[:,0], weights=szk, minlength=self.action_space.shape[0])
        grad -= np.bincount(data[:,1], weights=szk, minlength=self.action_space.shape[0])
        return grad
    
    def preference_jacobian(self, r) -> np.ndarray:
        """Computes the jacobian of the preference likelihood function
        
        r: user's latent reward function (discretized)
        Returns the jacobian of the preference likelihood function"""
        return self.pairwise_jacobian(r, np.array(self.preference_fbk), self.preference_noise)
    
    def coactive_jacobian(self, r) -> np.ndarray:
        """Computes the jacobian of the coactive likelihood function
        
        r: user's latent reward function (discretized)
        Returns the jacobian of the preference likelihood function"""
        return self.pairwise_jacobian(r, np.array(self.coactive_fbk), self.coactive_noise)
    
    def ordinal_jacobian(self, r) -> np.ndarray:
        """Computes the jacobian of the ordinal likelihood function. Applicable
        to preference and coactive feedback.
        
        r: user's latent reward function (discretized)
        data: the pairwise feedback data
        Returns the jacobian of the pairwise likelihood function"""
        # grad = np.zeros(self.action_space.shape[0])
        ofbk = np.array(self.ordinal_fbk)
        b0 = ofbk[:,1]
        b1 = ofbk[:,2]
        idxs = ofbk[:,0].astype(int)
        zm1 = (b1 - r[idxs]) / self.ordinal_noise
        zm2 = (b0 - r[idxs]) / self.ordinal_noise
        zs = 1 / self.ordinal_noise * (self.dsigmoid(zm1) - self.dsigmoid(zm2)) / (self.sigmoid(zm1) - self.sigmoid(zm2) + 1e-8)
        grad = np.bincount(idxs, weights=zs, minlength=self.action_space.shape[0])
        return grad
    
    def overall_jacobian(self, r: np.ndarray) -> np.ndarray:
        """Computes the jacobian of the overall likelihood function
        
        r: user's latent reward function (discretized)
        Returns the jacobian of the overall likelihood function"""
        jac = 0
        if self.preference_fbk and len(self.preference_fbk) > 0:
            jac += self.preference_jacobian(r)
        if self.coactive_fbk and len(self.coactive_fbk) > 0:
            print('bad')
            jac += self.coactive_jacobian(r)
        if self.ordinal_fbk and len(self.ordinal_fbk) > 0:
            print('bad2')
            jac += self.ordinal_jacobian(r)
        return jac
    
    def pairwise_hessian(self, r: np.ndarray, data: np.ndarray, noise: float) -> np.ndarray:
        """Computes the jacobian of the pairwise likelihood function. Applicable
        to preference and coactive feedback.
        
        r: user's latent reward function (discretized)
        data: the pairwise feedback data
        Returns the jacobian of the pairwise likelihood function"""
        hess = np.zeros((self.action_space.shape[0], self.action_space.shape[0]))
        zk = (r[data[:,0]] - r[data[:,1]]) / noise
        szk = 1 / noise**2 * (self.dsigmoid(zk)**2 / self.sigmoid(zk)**2 - self.d2sigmoid(zk) / self.sigmoid(zk))
        idxs = np.arange(self.action_space.shape[0])
        saikM = (idxs == data[:,0,None]).astype(int) - (idxs == data[:,1,None]).astype(int)
        M = np.repeat(szk[:,None], self.action_space.shape[0], axis=1) * saikM
        hess = M.T @ saikM
        return hess
    
    def preference_hessian(self, r) -> np.ndarray:
        """Computes the jacobian of the preference likelihood function
        
        r: user's latent reward function (discretized)
        Returns the jacobian of the preference likelihood function"""
        return self.pairwise_hessian(r, np.array(self.preference_fbk), self.preference_noise)
    
    def coactive_hessian(self, r) -> np.ndarray:
        """Computes the jacobian of the preference likelihood function
        
        r: user's latent reward function (discretized)
        Returns the jacobian of the preference likelihood function"""
        return self.pairwise_hessian(r, np.array(self.coactive_fbk), self.coactive_noise)
    
    def ordinal_hessian(self, r) -> np.ndarray:
        """Computes the jacobian of the preference likelihood function
        
        r: user's latent reward function (discretized)
        Returns the jacobian of the preference likelihood function"""
        ofbk = np.array(self.ordinal_fbk)
        b0 = ofbk[:,1]
        b1 = ofbk[:,2]
        idxs = ofbk[:,0].astype(int)
        zm1 = (b1 - r[idxs]) / self.ordinal_noise
        zm2 = (b0 - r[idxs]) / self.ordinal_noise
        zs = 1 / self.ordinal_noise * (self.dsigmoid(zm1) - self.dsigmoid(zm2)) / (self.sigmoid(zm1) - self.sigmoid(zm2) + 1e-8)
        grad = np.bincount(idxs, weights=zs, minlength=self.action_space.shape[0])
        return np.diag(grad)
        
    
    def overall_hessian(self, r: np.ndarray) -> np.ndarray:
        """Computes the jacobian of the overall likelihood function
        
        r: user's latent reward function (discretized)
        Returns the jacobian of the overall likelihood function"""
        hess = 0
        if self.preference_fbk and len(self.preference_fbk) > 0:
            hess += self.preference_hessian(r)
        if self.coactive_fbk and len(self.coactive_fbk) > 0:
            hess += self.coactive_hessian(r)
        if self.ordinal_fbk and len(self.ordinal_fbk) > 0:
            hess += self.ordinal_hessian(r)
        return hess
            
    def predict(self, r, a1, a2):
        """Predicts a user's preference given a latent reward function"""
        return r[self.get_idx(a1)] >= r[self.get_idx(a2)] # TODO: need to vectorize this
    
    def optimal_action(self, r):
        """Finds the optimal action given a latent reward function"""
        raise NotImplementedError() # need to get REVERSE index here
        return np.argmax(r)
    
    
    
# tissue engineering
# immunoengineering 
# biomaterials