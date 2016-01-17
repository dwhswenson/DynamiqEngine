class PotentialEnergySurface(object):
    """Abstract class for potential energy surfaces

    Attributes
    ----------
    derivatives_defined : integer (0, 1, or 2)
        Level required to simulate the system
    """
    def __init__(self, n_dim):
        self.derivatives_defined = 0

    def H(self, snapshot):
        raise NotImplementedError("Using generic PES object")

    def T(self, snapshot):
        pass

    def dHdq(self, snapshot):
        raise NotImplementedError("Using generic PES object")

    def dHdp(self, snapshot):
        raise NotImplementedError("Using generic PES object")

    def d2Hdq2(self, snapshot):
        raise NotImplementedError("Using generic PES object")

    def d2Hdp2(self, snapshot):
        raise NotImplementedError("Using generic PES object")

    def d2Hdqdp(self, snapshot):
        raise NotImplementedError("Using generic PES object")

    def d2Hdpdq(self, snapshot):
        raise NotImplementedError("Using generic PES object")

class OneDimensionalModel(PotentialEnergySurface):
    def __init__(self, interaction):
        self.n_atoms = 1
        self.n_spatial = 1
        self.derivatives_defined = 0
        self.interaction = interaction

    def H(self, snapshot):
        x = snapshot.positions[0]
        return self.kinetic_energy(snapshot) + self.interaction.f(x)

    def dHdq(self, snapshot):
        x = snapshot.positions[0]
        return np.array([self.interaction.dfdx(x)])

    def dHdp(self, snapshot):
        return np.array([snapshot.momenta[0]])

    # TODO: the rest is only necessary for full semiclassical calculations
