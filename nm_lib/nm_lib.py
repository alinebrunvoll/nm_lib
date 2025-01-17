#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 02 10:25:17 2021.

@author: Juan Martinez Sykora
"""

# import builtin modules

# import external public "common" modules
import numpy as np


def deriv_dnw(xx, hh, **kwargs):
    """
    Returns the downwind 2nd order derivative of hh array respect to xx array.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.

    Returns
    -------
    `array`
        The downwind 2nd order derivative of hh respect to xx. Last
        grid point is ill (or missing) calculated.
    """

    # Last point is ill calculated

    # np.roll(hh, -1) is the same as hh[i+1]
    hp = (np.roll(hh, -1) - hh) / (np.roll(xx, -1) - xx)

    return hp

def order_conv(hh, hh2, hh4, **kwargs):
    """
    Computes the order of convergence of a derivative function.

    Parameters
    ----------
    hh : `array`
        Function that depends on xx.
    hh2 : `array`
        Function that depends on xx but with twice number of grid points than hh.
    hh4 : `array`
        Function that depends on xx but with twice number of grid points than hh2.
    Returns
    -------
    `array`
        The order of convergence.
    """

def deriv_4tho(xx, hh, **kwargs):
    """
    Returns the 4th order derivative of hh respect to xx.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.

    Returns
    -------
    `array`
        The centered 4th order derivative of hh respect to xx.
        Last and first two grid points are ill calculated.
    """

def step_adv_burgers(
    xx, hh, a, cfl_cut=0.98, ddx=lambda x, y: deriv_dnw(x, y), **kwargs
):
    r"""
    Right hand side of Burger's eq. where a can be a constant or a function
    that depends on xx.

    Requires
    ----------
    cfl_adv_burger function which computes np.min(dx/a)

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default clf_cut=0.98.
    ddx : `lambda function`
        Allows to select the type of spatial derivative.
        By default lambda x,y: deriv_dnw(x, y)

    Returns
    -------
    `array`
        Time interval.
        Right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x}
    """

    dt = cfl_adv_burger(a, xx) * cfl_cut

    rhs = - a * ddx(xx, hh)

    return dt, rhs

def cfl_adv_burger(a, x):
    """
    Computes the dt_fact, i.e., Courant, Fredrich, and Lewy condition for the
    advective term in the Burger's eq.

    Parameters
    ----------
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    x : `array`
        Spatial axis.

    Returns
    -------
    `float`
        min(dx/|a|)
    """

    dx = np.gradient(x)

    return np.min(dx / np.abs(a)) 

def evolv_adv_burgers(
    xx,
    hh,
    nt,
    a,
    cfl_cut=0.98,
    ddx=lambda x, y: deriv_dnw(x, y),
    bnd_type="wrap",
    bnd_limits=[0, 1],
    **kwargs
):
    r"""
    Advance nt time-steps in time the burger eq for a being a a fix constant or array.
    Requires
    ----------
    step_adv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y).
    bnd_type : `string`
        Allows to select the type of boundaries.
        By default 'wrap'.
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1].

    Returns
    -------
    t : `array`
        time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents 
        all the elements of the domain.
    """

    t = np.zeros((nt))
    unnt = np.zeros((len(xx), nt))
    unnt[:, 0] = hh

    for i in range(0, nt-1): 

        dt, rhs = step_adv_burgers(xx, unnt[:, i], a=a, cfl_cut=cfl_cut, ddx=ddx)

        # Compute next timestep
        u_next = unnt[:, i] + rhs * dt 
        
        # Fix boundaries 
        if bnd_limits[1] > 0: 
            u_next_temp = u_next[bnd_limits[0] : -bnd_limits[1]]  # dnw / central scheme
        else:
            u_next_temp = u_next[bnd_limits[0] :] # upw scheme

        unnt[:, i+1] = np.pad(u_next_temp, bnd_limits, bnd_type) 

        # Update time
        t[i+1] = t[i] + dt

    return t, unnt

def deriv_upw(xx, hh, **kwargs):
    r"""
    returns the upwind 2nd order derivative of hh respect to xx.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.

    Returns
    -------
    `array`
        The upwind 2nd order derivative of hh respect to xx. First
        grid point is ill calculated.
    """

    # Last point is ill calculated

    # np.roll(hh, 1) is the same as hh[i-1]
    hp = (hh - np.roll(hh, 1)) / (xx - np.roll(xx, 1))

    return hp

def deriv_cent(xx, hh, **kwargs):
    r"""
    returns the centered 2nd derivative of hh respect to xx.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.

    Returns
    -------
    `array`
        The centered 2nd order derivative of hh respect to xx. First
        and last grid points are ill calculated.
    """

    # First and last points are ill calculated

    # np.roll(hh, 1) for hh[i-1] and np.roll(hh, -1) for hh[i+1]
    hp = (np.roll(hh, -1) - np.roll(hh, 1)) / (np.roll(xx, -1) - np.roll(xx, 1))

    return hp

def evolv_uadv_burgers(
    xx,
    hh,
    nt,
    cfl_cut=0.98,
    ddx=lambda x, y: deriv_dnw(x, y),
    bnd_type="wrap",
    bnd_limits=[0, 1],
    **kwargs
):
    r"""
    Advance nt time-steps in time the burger eq for a being u.

    Requires
    --------
    step_uadv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    cfl_cut : `float`
        constant value to limit dt from cfl_adv_burger.
        By default 0.98.
    ddx : `lambda function`
        Allows to change the space derivative function.
    bnd_type : `string`
        It allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """

    t = np.zeros((nt))
    unnt = np.zeros((len(xx), nt))
    unnt[:, 0] = hh

    for i in range(0, nt-1): 

        dt, rhs = step_uadv_burgers(xx, unnt[:, i], cfl_cut=cfl_cut, ddx=ddx)

        # Compute next timestep
        u_next = unnt[:, i] + rhs * dt 
        
        # Fix boundaries 
        if bnd_limits[1] > 0: 
            u_next_temp = u_next[bnd_limits[0] : -bnd_limits[1]]  # dnw scheme
        else:
            u_next_temp = u_next[bnd_limits[0] :] # upw scheme

        unnt[:, i+1] = np.pad(u_next_temp, bnd_limits, bnd_type) 

        # Update time
        t[i+1] = t[i] + dt

    return t, unnt 

def evolv_Lax_uadv_burgers(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    cfl_cut: float = 0.98,
    ddx = lambda x, y: deriv_dnw(x, y),
    bnd_type: str = "wrap",
    bnd_limits: tuple = [0, 1],
    **kwargs
) -> tuple:
    r"""
    Advance nt time-steps in time the burger eq for a being u using the Lax
    method.

    Requires
    --------
    step_uadv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of time steps.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `array`
        Lambda function allows to change the space derivative function.
        By derault  lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """

    t = np.zeros((nt))
    unnt = np.zeros((len(xx), nt))
    unnt[:, 0] = hh

    for i in range(0, nt-1): 

        dt, rhs = step_uadv_burgers(xx, unnt[:, i], cfl_cut=cfl_cut, ddx=ddx)

        # Compute next timestep
        u_next = 0.5 * (np.roll(unnt[:, i], -1) + np.roll(unnt[:, i], 1)) \
            - unnt[:, i] * dt / (np.roll(xx, -1) - np.roll(xx, 1)) \
            * (np.roll(unnt[:, i], -1) - np.roll(unnt[:, i], 1))
        
        # Fix boundaries 
        if bnd_limits[1] > 0: 
            u_next_temp = u_next[bnd_limits[0] : -bnd_limits[1]]  # dnw scheme
        else:
            u_next_temp = u_next[bnd_limits[0] :] # upw scheme

        unnt[:, i+1] = np.pad(u_next_temp, bnd_limits, bnd_type) 

        # Update time
        t[i+1] = t[i] + dt

    return t, unnt

def evolv_Lax_adv_burgers(
    xx,
    hh,
    nt,
    a,
    cfl_cut=0.98,
    ddx=lambda x, y: deriv_dnw(x, y),
    bnd_type="wrap",
    bnd_limits=[0, 1],
    **kwargs
):
    r"""
    Advance nt time-steps in time the burger eq for a being a a fix constant or
    array.

    Requires
    --------
    step_adv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """

    t = np.zeros((nt))
    unnt = np.zeros((len(xx), nt))
    unnt[:, 0] = hh

    for i in range(0, nt-1): 

        dt, rhs = step_adv_burgers(xx, unnt[:, i], a=a, cfl_cut=cfl_cut, ddx=ddx) 

        # Compute next timestep
        u_next = 0.5 * (np.roll(unnt[:, i], -1) + np.roll(unnt[:, i], 1)) + rhs * dt 
        
        # Fix boundaries 
        if bnd_limits[1] > 0: 
            u_next_temp = u_next[bnd_limits[0] : -bnd_limits[1]]  # dnw scheme
        else:
            u_next_temp = u_next[bnd_limits[0] :] # upw scheme

        unnt[:, i+1] = np.pad(u_next_temp, bnd_limits, bnd_type) 

        # Update time
        t[i+1] = t[i] + dt

    return t, unnt

def step_uadv_burgers(xx, hh, cfl_cut=0.98, ddx=lambda x, y: deriv_dnw(x, y), **kwargs):
    r"""
    Right hand side of Burger's eq. where a is u, i.e hh.

    Requires
    --------
        cfl_adv_burger function which computes np.min(dx/a)

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to select the type of spatial derivative.
        By default lambda x,y: deriv_dnw(x, y)


    Returns
    -------
    dt : `array`
        time interval
    unnt : `array`
        right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x}
    """

    a = hh

    # Compute the time step
    # dt = cfl_adv_burger(a[:-1], xx)
    dt = cfl_adv_burger(a, xx)

    # Compute the right hand side
    rhs = -a * ddx(xx, hh)

    return dt, rhs

def cfl_diff_burger(a, x):
    r"""
    Computes the dt_fact, i.e., Courant, Fredrich, and Lewy condition for the
    diffusive term in the Burger's eq.

    Parameters
    ----------
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    x : `array`
        Spatial axis.

    Returns
    -------
    `float`
        min(dx/|a|)
    """

    dx = np.gradient(x)
    return np.min(dx**2 / (2 * np.abs(a))) 

def evolv_Rie_uadv_burgers(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    cfl_cut: float = 0.98,
    ddx = lambda x, y: deriv_upw(x, y),
    bnd_type: str = "wrap",
    bnd_limits: tuple = [1, 0],
    **kwargs
) -> tuple:
    r"""
    Advance the burger eq `nt` time-steps in time for `a` = `u`, using a Riemann solver.

    Requires
    --------
    cfl_diff_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_upw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [1,0]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), where j represents
        all the elements of the domain.
    """

    t = np.zeros((nt))
    unnt = np.zeros((len(xx), nt))
    unnt[:, 0] = hh

    dx = xx[1] - xx[0]

    for i in range(0, nt-1):

        # 1. Compute u_L and u_R    
        u_L = np.roll(unnt[:, i], 0) 
        u_R = np.roll(unnt[:, i], -1) 

        # 2. Compute corresponding fluxes
        F_L = 0.5 * u_L**2
        F_R = 0.5 * u_R**2 

        # 3. Compute the propagation speed
        v_a = np.maximum(np.abs(u_L), np.abs(u_R)) 

        # 4. Compute the interface fluxes (Rusanov)
        F_plus05 = 0.5 * (F_L + F_R) - 0.5 * v_a * (u_R - u_L) # [i+1/2]
        F_int = (F_plus05 - np.roll(F_plus05, 1)) / dx 
        
        # 5. Advance the solution in time
        dt = cfl_adv_burger(v_a, xx)
        u_next = unnt[:, i] - dt * F_int  

        # Boundary conditions 
        if bnd_limits[1] > 0: 
            u_next_temp = u_next[bnd_limits[0]:-bnd_limits[1]]  # dnw scheme
        else:
            u_next_temp = u_next[bnd_limits[0]:] # upw scheme

        # Update the solution
        unnt[:, i+1] = np.pad(u_next_temp, bnd_limits, bnd_type) 
        t[i+1] = t[i] + dt

    return t, unnt

def evolve_Lax_Rie_uadv_burgers(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    cfl_cut: float = 0.98,
    ddx = lambda x, y: deriv_upw(x, y),
    bnd_type: str = "wrap",
    bnd_limits: tuple = [1, 0],
    **kwargs
) -> tuple:
    r"""
    Advance burger eq. `nt` time-steps for `a` = `u`, by combining the Lax and Riemann methods.

    Requires
    --------
    cfl_diff_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_upw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [1,0]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), where j represents
        all the elements of the domain.
    """

    t = np.zeros((nt))
    unnt = np.zeros((len(xx), nt))
    unnt[:, 0] = hh

    dx = xx[1] - xx[0]

    def flux_limiter(r):
        thetas = np.array([1., 2.])
        mins = np.zeros((len(thetas), len(r)))
        for i, theta in enumerate(thetas):
            mins[i] = np.min(( np.min(theta * r), np.min((1. + r)/2.), theta ))
        phi = np.max((0, np.max(mins)))
        return phi

    for i in range(0, nt-1):

        # Compute u_L and u_R    
        u_L = np.roll(unnt[:, i], 0)  # u[i]
        u_R = np.roll(unnt[:, i], -1) # u[i+1]

        # Compute corresponding fluxes
        F_L = 0.5 * u_L**2
        F_R = 0.5 * u_R**2 

        # Compute the propagation speed
        v_a = np.maximum(np.abs(u_L), np.abs(u_R)) 
        dt = cfl_adv_burger(v_a, xx)

        # Compute the Riemann flux
        F_Rie = 0.5 * (F_L + F_R) - 0.5 * v_a * (u_R - u_L) # [i+1/2]

        # Compute the Lax flux
        unnt_Lax = 0.5 * (np.roll(unnt[:, i], -1) + np.roll(unnt[:, i], 1)) \
            - unnt[:, i] * dt / (np.roll(xx, -1) - np.roll(xx, 1)) \
            * (np.roll(unnt[:, i], -1) - np.roll(unnt[:, i], 1))
        F_Lax = unnt_Lax

        # Compute the Lax-Rie flux
        r = (u_L - np.roll(unnt[:, i], 1)) / (u_R + u_L)
        F_Lax_Rie = F_Rie + flux_limiter(r) * (F_Lax - F_Rie)

        # 5. Advance the solution in time
        u_next = unnt[:, i] - dt * (F_Lax_Rie - np.roll(F_Lax_Rie, 1)) / dx 

        # Boundary conditions 
        if bnd_limits[1] > 0: 
            u_next_temp = u_next[bnd_limits[0]:-bnd_limits[1]]  # dnw scheme
        else:
            u_next_temp = u_next[bnd_limits[0]:] # upw scheme


        # Update the solution
        unnt[:, i+1] = np.pad(u_next_temp, bnd_limits, bnd_type) 
        t[i+1] = t[i] + dt

    return t, unnt

def ops_Lax_LL_Add(
    xx,
    hh,
    nt,
    a,
    b,
    cfl_cut=0.98,
    ddx=lambda x, y: deriv_dnw(x, y),
    bnd_type="wrap",
    bnd_limits=[0, 1],
    **kwargs
):
    r"""
    Advance nt time-steps in time the burger eq for a being a and b a fix
    constant or array. Solving two advective terms separately with the Additive
    Operator Splitting scheme.  Both steps are with a Lax method.

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries
        By default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """
    t = np.zeros((nt))
    unnt = np.zeros((len(xx), nt))
    unnt[:, 0] = hh

    for i in range(0, nt-1): 

        dt_u, rhs_u = step_adv_burgers(xx, unnt[:, i], a, cfl_cut=cfl_cut, ddx=ddx)
        dt_v, rhs_v = step_adv_burgers(xx, unnt[:, i], b, cfl_cut=cfl_cut, ddx=ddx)
        
        # Calculate timestep
        dt = np.min([dt_v, dt_u]) * 0.5 # XXX ADD 0.5 HERE

        dx = xx[1] - xx[0]

        # Compute next timestep
        # XXX ADD RHS MANUALLY AND FIX IT ACCORDING TO WIKI
        unn = 0.5 * (np.roll(unnt[:, i], -1) + np.roll(unnt[:, i], 1)) - ((a*dt) / (2*dx) * (np.roll(unnt[:, i], -1) - np.roll(unnt[:, i], 1))) #+ rhs_u * dt
        vnn = 0.5 * (np.roll(unnt[:, i], -1) + np.roll(unnt[:, i], 1)) - ((b*dt) / (2*dx) * (np.roll(unnt[:, i], -1) - np.roll(unnt[:, i], 1))) #+ rhs_v * dt
        u_next = unn + vnn - (0.5 * np.roll(unnt[:, i], -1) + 0.5 * np.roll(unnt[:, i], 1)) # unnt[:, i] # MADE STABLE by taking the surrounding half steps
        
        # Fix boundaries 
        if bnd_limits[1] > 0: 
            u_next_temp = u_next[bnd_limits[0] : -bnd_limits[1]]  # dnw/centr scheme
        else:
            u_next_temp = u_next[bnd_limits[0] :] # upw scheme

        unnt[:, i+1] = np.pad(u_next_temp, bnd_limits, bnd_type) 

        # Update time
        t[i+1] = t[i] + dt

    return t, unnt

def ops_Lax_LL_Lie(
    xx,
    hh,
    nt,
    a,
    b,
    cfl_cut=0.98,
    ddx=lambda x, y: deriv_dnw(x, y),
    bnd_type="wrap",
    bnd_limits=[0, 1],
    **kwargs
):
    r"""
    Advance nt time-steps in time the burger eq for a being a and b a fix
    constant or array. Solving two advective terms separately with the Lie-
    Trotter Operator Splitting scheme.  Both steps are with a Lax method.

    Requires:
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """
    t = np.zeros((nt))
    unnt = np.zeros((len(xx), nt))
    unnt[:, 0] = hh
    vnnt = np.zeros((len(xx), nt))
    vnnt[:, 0] = hh

    for i in range(0, nt-1): 

        dt_u = cfl_adv_burger(a, xx) * cfl_cut
        dt_v = cfl_adv_burger(b, xx) * cfl_cut
        dt = np.min([dt_u, dt_v]) * 0.5 # XXX ADD 0.5 HERE

        dx = xx[1] - xx[0]
        
        # _, rhs_u = step_adv_burgers(xx, unnt, a=a, cfl_cut=cfl_cut, ddx=ddx)

        unnt[:, i] = 0.5 * (np.roll(vnnt[:, i], -1) + np.roll(vnnt[:, i], 1)) - ((a*dt) / (2*dx) * (np.roll(vnnt[:, i], -1) - np.roll(vnnt[:, i], 1))) # + rhs_u * dt 

        # _, rhs_v = step_adv_burgers(xx, unnt[:, i], a=b, cfl_cut=cfl_cut, ddx=ddx)
        
        vnnt[:, i] = 0.5 * (np.roll(unnt[:, i], -1) + np.roll(unnt[:, i], 1)) - ((b*dt) / (2*dx) * (np.roll(unnt[:, i], -1) - np.roll(unnt[:, i], 1))) # + rhs_v * dt
                
        u_next = vnnt[:, i] #unn + vnn - unnt[:, i]
        
        # Fix boundaries 
        if bnd_limits[1] > 0: 
            u_next_temp = u_next[bnd_limits[0] : -bnd_limits[1]]  # dnw/centr scheme
        else:
            u_next_temp = u_next[bnd_limits[0] :] # upw scheme

        vnnt[:, i+1] = np.pad(u_next_temp, bnd_limits, bnd_type) 

        # Update time
        t[i+1] = t[i] + dt

    return t, vnnt

def ops_Lax_LL_Strang(
    xx,
    hh,
    nt,
    a,
    b,
    cfl_cut=0.98,
    ddx=lambda x, y: deriv_dnw(x, y),
    bnd_type="wrap",
    bnd_limits=[0, 1],
    **kwargs
):
    r"""
    Advance nt time-steps in time the burger eq for a being a and b a fix
    constant or array. Solving two advective terms separately with the Lie-
    Trotter Operator Splitting scheme. Both steps are with a Lax method.

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger
    numpy.pad for boundaries.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        Allows to select the type of boundaries.
        By default `wrap`
    bnd_limits : `list(int)`
        The number of pixels that will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """
    t = np.zeros((nt))
    unnt = np.zeros((len(xx), nt))
    unnt[:, 0] = hh 
    vnnt = np.zeros((len(xx), nt))
    vnnt[:, 0] = hh
    wnnt = np.zeros((len(xx), nt))
    wnnt[:, 0] = hh
    
    for i in range(0, nt-1): 

        # Calculate timestep
        dt_u = cfl_adv_burger(a, xx) * cfl_cut
        dt_v = cfl_adv_burger(b, xx) * cfl_cut

        dt = np.min([dt_u, dt_v]) * 0.5 # XXX ADD 0.5 HERE
        dx = xx[1] - xx[0]

        # Advance half a timestep:
        unnt[:, i] = 0.5 * (np.roll(wnnt[:, i], -1) + np.roll(wnnt[:, i], 1)) - ((a*dt) / (4*dx) * (np.roll(wnnt[:, i], -1) - np.roll(wnnt[:, i], 1)))#+ rhs_u * dt * 0.5 # XXX w here
        
        # Advance half a timestep:
        vnnt[:, i] = 0.5 * (np.roll(unnt[:, i], -1) + np.roll(unnt[:, i], 1)) - ((b*dt) / (2*dx) * (np.roll(unnt[:, i], -1) - np.roll(unnt[:, i], 1))) #+ rhs_v * dt * 0.5 # XXX u at t+1/2

        # Advance half a timestep:
        wnnt[:, i] = 0.5 * (np.roll(vnnt[:, i], -1) + np.roll(vnnt[:, i], 1)) - ((a*dt) / (4*dx) * (np.roll(vnnt[:, i], -1) - np.roll(vnnt[:, i], 1)))#+ rhs_w * dt * 0.5 # XXX v here

        u_next = wnnt[:, i]
        
        # Fix boundaries 
        if bnd_limits[1] > 0: 
            u_next_temp = u_next[bnd_limits[0] : -bnd_limits[1]]  # dnw/centr scheme
        else:
            u_next_temp = u_next[bnd_limits[0] :] # upw scheme

        wnnt[:, i+1] = np.pad(u_next_temp, bnd_limits, bnd_type) 
        # Update time
        t[i+1] = t[i] + dt 

    return t, wnnt # return w

def ops_Lax_LH_Strang(
    xx,
    hh,
    nt,
    a,
    b,
    cfl_cut=0.98,
    ddx=lambda x, y: deriv_dnw(x, y),
    bnd_type="wrap",
    bnd_limits=[0, 1],
    **kwargs
):
    r"""
    Advance nt time-steps in time the burger eq for a being a and b a fix
    constant or array. Solving two advective terms separately with the Strang
    Operator Splitting scheme. One step is with a Lax method and the second
    step is the Hyman predictor-corrector scheme.

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
#         By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """
    t = np.zeros((nt))
    unnt = np.zeros((len(xx), nt))
    unnt[:, 0] = hh
    vnnt = np.zeros((len(xx), nt))
    vnnt[:, 0] = hh
    wnnt = np.zeros((len(xx), nt))
    wnnt[:, 0] = hh

    for i in range(0, nt-1): 

        # Calculate timestep
        dt_a = cfl_adv_burger(a, xx) * cfl_cut
        dt_b = cfl_adv_burger(b, xx) * cfl_cut
        dt = np.min([dt_a, dt_b]) * 0.5 # XXX ADD 0.5 HERE
        dx = xx[1] - xx[0]

        unnt[:, i] = 0.5 * (np.roll(wnnt[:, i], -1) + np.roll(wnnt[:, i], 1)) - ((a*dt) / (4*dx) * (np.roll(wnnt[:, i], -1) - np.roll(wnnt[:, i], 1)))
        vnnt[:, i] = 0.5 * (np.roll(unnt[:, i], -1) + np.roll(unnt[:, i], 1)) - ((b*dt) / (2*dx) * (np.roll(unnt[:, i], -1) - np.roll(unnt[:, i], 1)))

        # Using the Hyman predictor-corrector scheme
        if i == 0:
            vnnt[:, i], u_prev, dt_v = hyman(xx, unnt[:, i], dt/2, a=b, cfl_cut=cfl_cut, ddx=ddx,)
        else: 
            vnnt[:, i], u_prev, dt_v = hyman(xx, unnt[:, i], dt/2, a=b, cfl_cut=cfl_cut, ddx=ddx, fold=u_prev, dtold=dt_v)
        
        wnnt[:, i] = 0.5 * (np.roll(vnnt[:, i], -1) + np.roll(vnnt[:, i], 1)) - ((a*dt) / (4*dx) * (np.roll(vnnt[:, i], -1) - np.roll(vnnt[:, i], 1)))

        u_next = wnnt[:, i]
        
        # Fix boundaries 
        if bnd_limits[1] > 0: 
            u_next_temp = u_next[bnd_limits[0] : -bnd_limits[1]]  # dnw/centr scheme
        else:
            u_next_temp = u_next[bnd_limits[0] :] # upw scheme

        wnnt[:, i+1] = np.pad(u_next_temp, bnd_limits, bnd_type) 

        # Update time
        t[i+1] = t[i] + dt

    return t, wnnt




def step_diff_burgers(xx, hh, a, ddx=lambda x, y: deriv_cent(x, y), **kwargs):
    r"""
    Right hand side of the diffusive term of Burger's eq. where nu can be a
    constant or a function that depends on xx.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y)

    Returns
    -------
    `array`
        Right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x}
    """    
    # evolv func

    dx = xx[1] - xx[0]

    rhs = a * (np.roll(hh, -1) - 2 * hh + np.roll(hh, 1)) / dx**2 
    # rhs = 1st and 2nd derivative of hh

    return rhs

def evolve(
    xx: np.ndarray,
    hh: np.ndarray,
    a: float,
    nt: int,
    cfl_cut: float = 0.98,
    bnd_type: str = "wrap",
    bnd_limits: tuple = [1, 0],
    **kwargs
) -> tuple:
    r"""
    Advance burger eq. `nt` time-steps for `a` = `u` for the Newton-Rhapson method.

    Requires
    --------
    cfl_diff_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_upw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [1,0]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), where j represents
        all the elements of the domain.
    """

    t = np.zeros((nt))
    unnt = np.zeros((len(xx), nt))
    unnt[:, 0] = hh

    for i in range(0, nt-1): 

        dx = xx[1] - xx[0]
        rhs = step_diff_burgers(xx, unnt[:, i], a=a, cfl_cut=cfl_cut)
        dt = cfl_diff_burger(a, xx) * cfl_cut 

        # Compute next timestep
        u_next = unnt[:, i] + rhs * dt
        
        # Fix boundaries 
        if bnd_limits[1] > 0: 
            u_next_temp = u_next[bnd_limits[0] : -bnd_limits[1]]  # dnw / central scheme
        else:
            u_next_temp = u_next[bnd_limits[0] :] # upw scheme

        unnt[:, i+1] = np.pad(u_next_temp, bnd_limits, bnd_type) 

        # Update time
        t[i+1] = t[i] + dt

    return t, unnt

def NR_f(xx, un, uo, a, dt, **kwargs):
    r"""
    NR F function.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    un : `array`
        Function that depends on xx. (u^{n+1})
    uo : `array`
        Function that depends on xx. (u^{n})
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `float`
        Time interval

    Returns
    -------
    `array`
        function  u^{n+1}_{j}-u^{n}_{j} - a (u^{n+1}_{j+1} - 2 u^{n+1}_{j} -u^{n+1}_{j-1}) dt
    """

    dx = xx[1] - xx[0]

    F_j =  un - uo - a * (np.roll(un, -1) - 2 * un + np.roll(un, 1)) * dt / dx**2

    return F_j

    # return un - step_diff_burgers(xx, un, a) * dt - uo 

def jacobian(xx, un, a, dt, **kwargs):
    r"""
    Jacobian of the F function.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    un : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `float`
        Time interval

    Returns
    -------
    `array`
        Jacobian F_j'(u^{n+1}{k})
    """

    # Calculate jacobian
    # Matrix that is 0 or 1 depending on when the indx is j or k

    dx = xx[1] - xx[0]

    J = np.zeros((len(xx), len(xx)))

    for i in range(len(xx)):
        J[i, i] = 1 + dt * 2 * a / dx**2
        if i < len(xx) - 1:
            J[i, i+1] = -dt * a / dx**2
        if i > 1:
            J[i, i-1] = -dt * a / dx**2

    return J

def Newton_Raphson(
    xx, hh, a, dt, nt, toll=1e-5, ncount=2, bnd_type="wrap", bnd_limits=[1, 1], **kwargs
):
    r"""
    NR scheme for the burgers equation.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `float`
        Time interval
    nt : `int`
        Number of iterations
    toll : `float`
        Error limit.
        By default 1e-5
    ncount : `int`
        Maximum number of iterations.
        By default 2
    bnd_type : `string`
        Allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [1,1]

    Returns
    -------
    t : `array`
        Array of time.
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    errt : `array`
        Error for each timestep
    countt : `list(int)`
        number iterations for each timestep
    """
    err = 1.0
    unnt = np.zeros((np.size(xx), nt))
    errt = np.zeros((nt))
    countt = np.zeros((nt))
    unnt[:, 0] = hh
    t = np.zeros((nt))

    ## Looping over time
    for it in range(1, nt):
        uo = unnt[:, it - 1]
        ug = unnt[:, it - 1]
        count = 0
        # iteration to reduce the error.
        while (err >= toll) and (count < ncount):

            jac = jacobian(xx, ug, a, dt)  # Jacobian
            ff1 = NR_f(xx, ug, uo, a, dt)  # F
            # Inversion:
            un = ug - np.matmul(np.linalg.inv(jac), ff1)

            # error:
            err = np.max(np.abs(un - ug) / (np.abs(un) + toll))  # error
            # err = np.max(np.abs(un-ug))
            errt[it] = err

            # Number of iterations
            count += 1
            countt[it] = count

            # Boundaries
            if bnd_limits[1] > 0:
                u1_c = un[bnd_limits[0] : -bnd_limits[1]]
            else:
                u1_c = un[bnd_limits[0] :]
            un = np.pad(u1_c, bnd_limits, bnd_type)
            ug = un
        err = 1.0
        t[it] = t[it - 1] + dt
        unnt[:, it] = un

    return t, unnt, errt, countt

def NR_f_u(xx, un, uo, dt, **kwargs):
    r"""
    NR F function.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    un : `array`
        Function that depends on xx. 
    uo : `array`
        Function that depends on xx. 
    a : `float` and `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `int`
        Time interval

    Returns
    -------
    `array`
        function  u^{n+1}_{j}-u^{n}_{j} - u^{n+1}_j (u^{n+1}_{j+1} - 2 u^{n+1}_{j} -u^{n+1}_{j-1}) dt
    """

    dx = xx[1] - xx[0]
    return un - uo - un * (np.roll(un, -1) - 2 * un + np.roll(un, 1)) * dt / dx**2

def jacobian_u(xx, un, dt, **kwargs):
    """
    Jacobian of the F function.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    un : `array`
        Function that depends on xx.
    dt : `int`
        Time interval

    Returns
    -------
    `array`
        Jacobian F_j'(u^{n+1}{k})
    """

    dx = xx[1] - xx[0]

    J = np.zeros((len(xx), len(xx)))

    for i in range(len(xx)):
        J[i, i] = 1 + dt * 2 * un[i] / dx**2
        if i < len(xx) - 1:
            J[i, i+1] = -dt * un[i] / dx**2
        if i > 1:
            J[i, i-1] = -dt * un[i] / dx**2

    return J

def Newton_Raphson_u(
    xx, hh, dt, nt, toll=1e-5, ncount=2, bnd_type="wrap", bnd_limits=[1, 1], **kwargs
):
    """
    NR scheme for the burgers equation.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    dt : `float`
        Time interval
    nt : `int`
        Number of iterations
    toll : `float`
        Error limit.
        By default 1-5
    ncount : `int`
        Maximum number of iterations.
        By default 2
    bnd_type : `string`
        Allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [1,1]

    Returns
    -------
    t : `array`
        Time.
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    errt : `array`
        Error for each timestep
    countt : `array(int)`
        Number iterations for each timestep
    """
    
    err = 1.0
    unnt = np.zeros((np.size(xx), nt))
    errt = np.zeros((nt))
    countt = np.zeros((nt))
    unnt[:, 0] = hh
    t = np.zeros((nt))

    ## Looping over time
    for it in range(1, nt):
        uo = unnt[:, it - 1]
        ug = unnt[:, it - 1]
        count = 0
        # iteration to reduce the error.
        while (err >= toll) and (count < ncount):

            jac = jacobian_u(xx, ug, dt)  # Jacobian
            ff1 = NR_f_u(xx, ug, uo, dt)  # F
            # Inversion:
            un = ug - np.matmul(np.linalg.inv(jac), ff1)

            # error
            err = np.max(np.abs(un - ug) / (np.abs(un) + toll))
            errt[it] = err

            # Number of iterations
            count += 1
            countt[it] = count

            # Boundaries
            if bnd_limits[1] > 0:
                u1_c = un[bnd_limits[0] : -bnd_limits[1]]
            else:
                u1_c = un[bnd_limits[0] :]
            un = np.pad(u1_c, bnd_limits, bnd_type)
            ug = un
        err = 1.0
        t[it] = t[it - 1] + dt
        unnt[:, it] = un

    return t, unnt, errt, countt

def taui_sts(nu, niter, iiter):
    """
    STS parabolic scheme. [(nu -1)cos(pi (2 iiter - 1) / 2 niter) + nu + 1]^{-1}

    Parameters
    ----------
    nu : `float`
        Coefficient, between (0,1).
    niter : `int`
        Number of iterations
    iiter : `int`
        Iterations number

    Returns
    -------
    `float`
        [(nu -1)cos(pi (2 iiter - 1) / 2 niter) + nu + 1]^{-1}
    """

def evol_sts(
    xx,
    hh,
    nt,
    a,
    cfl_cut=0.45,
    ddx=lambda x, y: deriv_cent(x, y),
    bnd_type="wrap",
    bnd_limits=[0, 1],
    nu=0.9,
    n_sts=10,
):
    """
    Evolution of the STS method.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.45
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_cent(x, y)
    bnd_type : `string`
        Allows to select the type of boundaries
        by default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By defalt [0,1]
    nu : `float`
        STS nu coefficient between (0,1).
        By default 0.9
    n_sts : `int`
        Number of STS sub iterations.
        By default 10

    Returns
    -------
    t : `array`
        time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """

def hyman(
    xx,
    f,
    dth,
    a,
    fold=None,
    dtold=None,
    cfl_cut=0.8,
    ddx=lambda x, y: deriv_dnw(x, y),
    bnd_type="wrap",
    bnd_limits=[0, 1],
    **kwargs
):

    dt, u1_temp = step_adv_burgers(xx, f, a, ddx=ddx)

    if np.any(fold) == None:
        fold = np.copy(f)
        f = (np.roll(f, 1) + np.roll(f, -1)) / 2.0 + u1_temp * dth
        dtold = dth

    else:
        ratio = dth / dtold
        a1 = ratio**2
        b1 = dth * (1.0 + ratio)
        a2 = 2.0 * (1.0 + ratio) / (2.0 + 3.0 * ratio)
        b2 = dth * (1.0 + ratio**2) / (2.0 + 3.0 * ratio)
        c2 = dth * (1.0 + ratio) / (2.0 + 3.0 * ratio)

        f, fold, fsav = hyman_pred(f, fold, u1_temp, a1, b1, a2, b2)

        if bnd_limits[1] > 0:
            u1_c = f[bnd_limits[0] : -bnd_limits[1]]
        else:
            u1_c = f[bnd_limits[0] :]
        f = np.pad(u1_c, bnd_limits, bnd_type)

        dt, u1_temp = step_adv_burgers(xx, f, a, cfl_cut, ddx=ddx)

        f = hyman_corr(f, fsav, u1_temp, c2)

    if bnd_limits[1] > 0:
        u1_c = f[bnd_limits[0] : -bnd_limits[1]]
    else:
        u1_c = f[bnd_limits[0] :]
    f = np.pad(u1_c, bnd_limits, bnd_type)

    dtold = dth

    return f, fold, dtold

def hyman_corr(f, fsav, dfdt, c2):
    return fsav + c2 * dfdt

def hyman_pred(f, fold, dfdt, a1, b1, a2, b2):
    fsav = np.copy(f)
    tempvar = f + a1 * (fold - f) + b1 * dfdt
    fold = np.copy(fsav)
    fsav = tempvar + a2 * (fsav - tempvar) + b2 * dfdt
    f = tempvar

    return f, fold, fsav