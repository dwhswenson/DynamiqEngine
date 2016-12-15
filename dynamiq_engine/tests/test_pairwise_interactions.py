import dynamiq_engine as dynq
from tools import *

from dynamiq_engine.potentials.pairwise_interactions import *
from example_systems import ho_2_1, anharmonic_morse

class testConstantInteraction(object):
    def setup(self):
        self.cst = ConstantInteraction(value=3.5)

    def test_f(self):
        tests = {
            0.0 : 3.5,
            1.0 : 3.5,
            0.5 : 3.5
        }
        check_function(self.cst.f, tests)

    def test_dfdx(self):
        tests = { 0.0 : 0.0, 1.0 : 0.0, 0.5 : 0.0 }
        check_function(self.cst.dfdx, tests)

    def test_d2fdx2(self):
        tests = { 0.0 : 0.0, 1.0 : 0.0, 0.5 : 0.0 }
        check_function(self.cst.d2fdx2, tests)

class testHarmonicOscillatorInteraction(object):
    def setup(self):
        self.ho = ho_2_1.potential
        self.ho_test_snaps = ho_2_1.snapshots

    def test_f(self):
        tests = {
            0.0 : 1.0, # = 0.5*2.0*(0.0-1.0)^2
            1.0 : 0.0,
            0.5 : 0.25, # = 0.5*2.0*(0.5-1.0)^2
            2.0 : 1.0
        }
        check_function(self.ho.f, tests)

    def test_dfdx(self):
        tests = {
            0.0 : -2.0,
            1.0 : 0.0,
            0.5 : -1.0,
            2.0 : 2.0
        }
        check_function(self.ho.dfdx, tests)

    def test_d2fdx2(self):
        tests = {
            0.0 : 2.0,
            1.0 : 2.0,
            0.5 : 2.0,
            2.0 : 2.0
        }
        check_function(self.ho.d2fdx2, tests)

    def test_H(self):
        tests = {
            self.ho_test_snaps[0] : 1.0,
            self.ho_test_snaps[1] : 4.25,
            self.ho_test_snaps[2] : 1.25
        }
        check_function(self.ho.H, tests)

    def test_T(self):
        # Definition of T is $L + V = p * dH/dp - H + V$; test this directly.
        # Use a lambda rather than nested def because nested def screws with
        # my in-editor test runner
        explicit_T = lambda pes, snap : (
            np.dot(snap.momenta, pes.dHdp(snap))
            - self.ho.H(snap) + self.ho.V(snap)
        )
        tests = {s : explicit_T(self.ho, s) for s in self.ho_test_snaps}
        check_function(self.ho.T, tests)

    def test_dHdq(self):
        tests = {
            self.ho_test_snaps[0] : np.array([0.0]),
            self.ho_test_snaps[1] : np.array([-1.0]),
            self.ho_test_snaps[2] : np.array([2.0])
        }
        check_function(self.ho.dHdq, tests)

    def test_dHdp(self):
        tests = {
            self.ho_test_snaps[0] : np.array([2.0]),
            self.ho_test_snaps[1] : np.array([4.0]),
            self.ho_test_snaps[2] : np.array([-1.0])
        }
        check_function(self.ho.dHdp, tests)

    def test_d2Hdq2(self):
        tests = {s : np.array([[2.0]]) for s in self.ho_test_snaps}
        check_function(self.ho.d2Hdq2, tests)

    def test_d2Hdpdq(self):
        tests = {s : np.array([[0.0]]) for s in self.ho_test_snaps}
        check_function(self.ho.d2Hdpdq, tests)

        local_d2Hdpdq = None
        for s in self.ho_test_snaps:
            self.ho.set_d2Hdpdq(local_d2Hdpdq, s)

    def test_d2Hdqdp(self):
        tests = {s : np.array([[0.0]]) for s in self.ho_test_snaps}
        check_function(self.ho.d2Hdqdp, tests)

        local_d2Hdqdp = None
        for s in self.ho_test_snaps:
            self.ho.set_d2Hdqdp(local_d2Hdqdp, s)

    def test_d2Hdp2(self):
        tests = {s : np.array([[2.0]]) for s in self.ho_test_snaps}
        check_function(self.ho.d2Hdp2, tests)


class testTanhInteraction(object):
    def setup(self):
        self.tanh = TanhInteraction(a=0.75, V0=0.1, R0=0.5)

    def test_f(self):
        tests = {
            0.0 : -0.0358357398350786,
            0.5 : 0.0,
            1.0 : 0.0358357398350786,
            2.0 : 0.0809301070201781
        }
        check_function(self.tanh.f, tests)

    def test_dfdx(self):
        tests = {
            0.0 : 0.0653684981285442,
            0.5 : 0.0750000000000000,
            1.0 : 0.0653684981285442,
            2.0 : 0.0258773833327689
        }
        check_function(self.tanh.dfdx, tests)

    def test_d2fdx2(self):
        tests = {
            0.0 : 0.0351379273851650,
            0.5 : 0.0,
            1.0 : -0.0351379273851650,
            2.0 : -0.0314138910378474
        }
        check_function(self.tanh.d2fdx2, tests)


class testMorseInteraction(object):
    def setup(self):
        #self.morse = MorseInteraction(D=30.0, beta=0.08, x0=0.5)
        self.morse = anharmonic_morse.potential
        self.morse_test_snaps = anharmonic_morse.snapshots

    def test_f(self):
        tests = {
            -1.0 : 0.487663414879601,
            0.0 : 0.0499655787054630,
            0.5 : 0.0,
            5.0 : 2.74198811453729
        }
        check_function(self.morse.f, tests)

    def test_dfdx(self):
        tests = {
            -1.0 : -0.690011033961740,
            0.0 : -0.203886208716337,
            0.5 : 0.0,
            5.0 : 1.01243553653309
        }
        check_function(self.morse.dfdx, tests)

    def testd2fdx2(self):
        tests = {
            -1.0 : 0.543360556440359,
            0.0 : 0.432293130684491,
            0.5 : 0.384000000000000,
            5.0 : 0.105918023365982
        }
        check_function(self.morse.d2fdx2, tests)

    def test_d2Hdq2(self):
        tests = {
            self.morse_test_snaps[0] : 0.432293130684491,
            self.morse_test_snaps[1] : 0.543360556440359,
            self.morse_test_snaps[2] : 0.105918023365982
        }
        check_function(self.morse.d2Hdq2, tests)

    def test_d2Hdp2(self):
        tests = {s : np.array([[5.0]]) for s in self.morse_test_snaps}
        check_function(self.morse.d2Hdp2, tests)

class testQuarticInteraction(object):
    def setup(self):
        self.quartic = QuarticInteraction(1.5, 1.25, 2.0, 1.0, 0.25, 0.5)

    def test_f(self):
        tests = {
            -1.0 : 6.625,
            0.0 : 0.1875,
            0.5 : 0.25,
            2.0 : 18.0625
        }
        check_function(self.quartic.f, tests)

    def test_dfdx(self):
        tests = {
            -1.0 : -16.8125,
            0.0 : -0.8125,
            0.5 : 1.0,
            2.0 : 35.6875
        }
        check_function(self.quartic.dfdx, tests)

    def test_d2fdx2(self):
        tests = {
            -1.0 : 33.25,
            0.0 : 4.75,
            0.5 : 4.0,
            2.0 : 55.75
        }
        check_function(self.quartic.d2fdx2, tests)

class testGaussianInteraction(object):
    def setup(self):
        self.gaussian = GaussianInteraction(A=2.0, alpha=0.25, x0=0.5)

    def test_f(self):
        tests = {
            0.0 : 1.87882612562695,
            0.5 : 2.0,
            1.0 : 1.87882612562695,
            2.0 : 1.13956564946185
        }
        check_function(self.gaussian.f, tests)

    def test_dfdx(self):
        tests = {
            0.0 : 0.469706531406738,
            0.5 : 0.0,
            1.0 : -0.469706531406738,
            2.0 : -0.854674237096384
        }
        check_function(self.gaussian.dfdx, tests)

    def test_d2fdx2(self):
        tests = {
            0.0 : -0.821986429961791,
            0.5 : -1.0,
            1.0 : -0.821986429961791,
            2.0 : 0.0712228530913653
        }
        check_function(self.gaussian.d2fdx2, tests)

class testLennardJonesInteraction(object):
    pass

class testWCAInteraction(object):
    pass

class testGeneralizedWCAInteraction(object):
    pass
