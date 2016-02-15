import numpy as np

class PotentialEnergySurface(object):
    """Abstract class for potential energy surfaces

    Attributes
    ----------
    dynamics_level : integer (0, 1, or 2)
        Level required to simulate the system
    """
    def H(self, snapshot):
        return self.V(snapshot) + self.kinetic_energy(snapshot)

    def V(self, snapshot):
        raise NotImplementedError("Using generic PES object")

    def __call__(self, snapshot):
        return self.V(snapshot)

    def kinetic_energy(self, snapshot):
        return 0.5*np.dot(snapshot.velocities, snapshot.momenta)

    def T(self, snapshot):
        """T = L + V; such that L = T - V; for generic action integration"""
        return self.kinetic_energy(snapshot)

    def dHdq(self, snapshot):
        dHdq = np.zeros(self.n_spatial * self.n_atoms)
        self.set_dHdq(dHdq, snapshot)
        return dHdq

    def set_dHdq(self, dHdq, snapshot):
        raise NotImplementedError("Using generic PES object")

    def dHdp(self, snapshot):
        dHdp = np.zeros(self.n_spatial * self.n_atoms)
        self.set_dHdp(dHdp, snapshot)
        return dHdp

    def set_dHdp(self, dHdp, snapshot):
        np.copyto(dHdp, snapshot.velocities)

    def d2Hdq2(self, snapshot):
        n_dim = self.n_spatial * self.n_atoms
        d2Hdq2 = np.zeros(n_dim, n_dim)
        self.set_d2Hdq2(d2Hdq2, snapshot)
        return d2Hdq2

    def set_d2Hdq2(self, d2Hdq2, snapshot):
        raise NotImplementedError("Using generic PES object")

    def d2Hdp2(self, snapshot):
        n_dim = self.n_spatial * self.n_atoms
        d2Hdp2 = np.zeros(n_dim, n_dim)
        self.set_d2Hdp2(d2Hdp2, snapshot)

    def set_d2Hdp2(self, d2Hdp2, snapshot):
        return # default shouldn't even alloc these

    def d2Hdqdp(self, snapshot):
        n_dim = self.n_spatial * self.n_atoms
        d2Hdqdp = np.zeros(n_dim, n_dim)
        self.set_d2Hdqdp(d2Hdqdp, snapshot)

    def set_d2Hdqdp(self, d2Hdqdp, snapshot):
        return # default shouldn't even alloc these

    def d2Hdpdq(self, snapshot):
        n_dim = self.n_spatial * self.n_atoms
        d2Hdpdq = np.zeros(n_dim, n_dim)
        self.set_d2Hdpdq(d2Hdpdq, snapshot)

    def set_d2Hdpdq(self, d2Hdpdq, snapshot):
        raise NotImplementedError("Using generic PES object")


class OneDimensionalInteractionModel(PotentialEnergySurface):
    def __init__(self, interaction):
        super(OneDimensionalInteractionModel, self).__init__()
        self.n_atoms = 1
        self.n_spatial = 1
        self.dynamics_level = 0
        self.interaction = interaction

    def V(self, snapshot):
        return self.interaction.f(snapshot.coordinates[0])

    def set_dHdq(self, dHdq, snapshot):
        x = snapshot.coordinates[0]
        dHdq[0] = self.interaction.dfdx(x)

    def set_dHdp(self, dHdp, snapshot):
        dHdp[0] = snapshot.velocities[0]

    # TODO: the rest is only necessary for full semiclassical calculations
    def set_d2Hdq2(self, d2Hdq2, snapshot):
        x = snapshot.coordinates[0]
        d2Hdq2[(0,0)] = self.interaction.d2fdx2(x)
