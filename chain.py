import numpy as np
class Chain:
    def __init__(self, N, K):
        self.N = N
        self.K = K
        # Create a random chain of N variables, {X0, X1, ..., XN-1}
        # each variable can take on values {0, 1, ..., K-1}.
        # 
        # Potential functions defined between each pair of adjacent variables.
        # For example, consider a chain of three variables:
        #     X0 - X1 - X2
        # There are two potentials:
        #     P0(X0,X1) and P1(X1,X2)
        #
        # This can be concisely described as an N-1xKxK matrix.

        # So in this representation, phi[n,i,j] means the potential
        #  value associated for the case that X_n=i and X_{n+1}=j
        
        self.phi = np.random.random((N-1,K,K))

    # Load a chain from a given file.
    @staticmethod
    def load(filename):
        phi = np.load(filename)
        N = phi.shape[0]+1
        K = phi.shape[1]
        c = Chain(N,K)
        c.phi = phi
        return c

    # Save the given chain to a specific filename.
    # Note: numpy expects the filename to end in .npz
    def save(self, filename):
        np.save(filename, self.phi)

    # Utility method to print the chain potentials.
    def __str__(self):
        return f'{self.N} {self.K}\n{self.phi}'

    # Return an NxK matrix of exact probababilities.
    # Suppose c is any chain, and let p = c.exact().
    # Then p should be an NxK matrix with p[n,k] := Pr(Xn = k).
    # TODO: Implement this method.
    def exact(self):
        fwd_msg = np.ones((self.N, self.K))
        for i in range(1, self.N):
            fwd_msg[i] = np.sum(fwd_msg[i-1, :, None] * self.phi[i-1], axis=0)

        bwd_msg = np.ones((self.N, self.K))
        for i in range(self.N-2, -1, -1):
            bwd_msg[i] = np.sum(bwd_msg[i+1, None, :] * self.phi[i], axis=1)

        p = fwd_msg * bwd_msg
        p_sum = np.sum(p, axis=1)
        p /= p_sum[:, None]
        return p
            
    
    # Return an NxK matrix of sampled probababilities.
    # Suppose c is any chain, and let p = c.sample(steps).
    # Then p should be an NxK matrix with p[n,k] ~= Pr(Xn = k).
    # This method should implement a Gibbs sampler.
    # Be sure to describe how you calculate the proposal distribution in your writeup.
    # TODO: Implement this method.
    def sample(self, steps):
        state = np.random.randint(self.K, size=self.N)
        p = np.zeros((self.N, self.K))

        for _ in range(steps):
            for i in range(self.N):
                if i == 0:
                    conditional_p = self.phi[i, :, state[i+1]]
                elif i == self.N - 1:
                    conditional_p = self.phi[i-1, state[i-1], :]
                else:
                    conditional_p = self.phi[i-1, state[i-1], :] * self.phi[i, :, state[i+1]]

                conditional_p /= np.sum(conditional_p)
                state[i] = np.random.choice(self.K, p=conditional_p)
            p[np.arange(self.N), state] += 1

        p /= steps
        return p
    