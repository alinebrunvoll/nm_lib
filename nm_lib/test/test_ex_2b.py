import numpy as np
from nm_lib import nm_lib as nm

def u_analytical(xx: np.ndarray, tt: np.ndarray, a: float = -1) -> np.ndarray:
    r"""
    Analytical solution to the advection equation.

    Requires
    ----------
    u(xx) : `function`

    Parameters
    ----------
    xx : `array`
         the x-axis.
    tt : `array`
         the time. 
    a : `float`
        the advection velocity. (-1 as default)

    Returns
    ------- 
    `array`
        the analytical solution over the times `tt`.
    """

    x0 = -2.6
    xf = 2.6

    u_an = np.zeros((len(xx), len(tt)))
    x_new = np.zeros((len(xx), len(tt)))

    for i in range(len(tt)):

        x_new[:, i] = ((xx - a * tt[i] ) - x0) % (xf - x0) + x0
        u_an[:, i] = u_initial(x_new[:, i], tt[i])
    
    return u_an


def u_initial(x: np.ndarray, t: float = 0) -> np.ndarray:
    r"""
    Initial condition for the advection equation.

    Parameters
    ----------
    x : `array`
        the x-axis.
    t : `float`
        the time.

    Returns
    -------
    `array`
        the initial condition.
    """
    return np.cos(6*np.pi*x / 5)**2 / np.cosh(5*x**2)



def test_ex_2b():
    r"""
    Test the numerical solution to the advection equation.

    Requires
    ----------
    u_analytical : `function`
    """

    x0 = -2.6
    xf = 2.6

    nt = 1000
    nx = 100 
    xx = np.linspace(x0, xf, nx) 

    a = -1

    t, unnt = nm.evolv_adv_burgers(xx, u_initial(xx), nt, a=a)
    u_an = u_analytical(xx, t)

    assert np.isclose(np.sum(unnt), np.sum(u_an), rtol=1e-6)


if __name__ == "__main__":
    test_ex_2b()








