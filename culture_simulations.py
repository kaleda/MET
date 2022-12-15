"""
Simulations for the paper 'Conditions that favor cumulative cultural
evolution' by Kaleda K. Denton, Yoav Ram, and Marcus W. Feldman.
"""

import numpy as np
import math
import pandas as pd
import itertools
from scipy import special as sp

"""
Functions used in models of non-cumulative culture.
"""

def specify_model():
    """
    The user specifies the type of model.
    """
    signal = -1
    while signal < 0:
        msg = ('Enter \n'
               ' 0 for non-cumulative culture or \n'
               ' 1 for cumulative culture: \n')
        cumulative = input('Please specify the type of model. ' + msg)
        while cumulative not in ['0', '1']:
            cumulative = input('The value must be 0 or 1. ' + msg)

        msg = ('For cultural transmission, please enter \n'
               ' 0 for conformity, \n'
               ' 1 for anticonformity, \n'
               ' 2 for unbiased transmission, or \n'
               ' 3 for success-biased transmission: \n')
        transmission = input(msg)
        while transmission not in ['0', '1', '2', '3']:
            transmission = input('The value must be 0, 1, 2, or 3. ' + msg)

        msg = ('For environmental change, please enter \n'
               ' 0 for deterministic rate γ, \n'
               ' 1 for the periodic model, or \n'
               ' 2 for the random model: \n')
        env_change = input(msg)
        while env_change not in ['0', '1', '2']:
            env_change = input('The value must be 0, 1, or 2. ' + msg)

        if cumulative == '0':  # For the model of non-cumulative culture
            msg = ('Please enter \n'
                   ' 0 if social learning is initially rare or \n'
                   ' 1 if social learning is initially common: \n')
        else:  # For the model of cumulative culture
            msg = ('Please enter \n 0 if cumulative culture is'
                   ' initially absent or \n'
                   ' 1 if cumulative culture is initially present: \n')
        init_frq = input(msg)
        while init_frq not in ['0', '1']:
            init_frq = input('The value must be 0 or 1. ' + msg)

        msg = ('Please enter \n 0 for an infinite population or \n'
               ' 1 for a finite population: \n')
        finite = input(msg)
        while finite not in ['0', '1']:
            finite = input('The value must be 0 or 1. ' + msg)

        if finite == '1':
            msg = ('Please enter the number of individuals'
                   ' in the finite population: \n')
            pop_size = input(msg)
            while pop_size.isdigit() == False or pop_size == '0':
                pop_size = input('The value must be a positive integer. ' + msg)
        else:
            pop_size = 'NA'

        msg = ('Please enter \n 0 if this is the last model or \n'
               ' 1 if you want to run more models after this one: \n')
        iter = input(msg)
        while iter not in ['0', '1']:
            iter = input('The value must be 0 or 1. ' + msg)

        msg = ('Please enter \n 0 to lock in your answers or \n'
               ' 1 if you want to change your answers: \n')
        end = input(msg)
        while end not in ['0', '1']:
            end = input('The value must be 0 or 1. ' + msg)
        if end == '0':
            signal = 1
    # Return a unique string that specifies the model
    # Note: the last entry (iter) will be deleted before the string is passed into some later functions
    return cumulative + transmission + env_change + init_frq + finite + pop_size + iter


def at_equilibrium(frqs1, frqs2, pr):
    """
    Determines whether two lists of frequencies, frqs1 and frqs2, are equal,
    up to the precision specified by pr. In practice, it is used to determine
    whether equilibrium is reached.
    """
    for i in range(len(frqs1)):
        if round(frqs1[i], pr) != round(frqs2[i], pr):
            return False  # Not at equilibrium
    return True


def finite_sample(N, frqs):
    """
    Gives phenotype frequencies in a finite population of size N, given the
    corresponding frequencies denoted by frqs in an infinite population.
    """
    draw = np.random.multinomial(N, frqs)
    return [i / N for i in draw]


def rounding(frqs):
    """
    The input to this function is frqs, a list of frequencies that all must
    be between 0 and 1 and sum to 1. The output is the list of frequencies
    without rounding error.
    """
    frqs_in_bounds = []
    for frq in frqs:  # Ensure that all frequencies are in [0,1]
        if frq < 0:
            frq = 0
        elif frq > 1:
            frq = 1
        frqs_in_bounds.append(frq)

    valid_frqs = []
    normalizer = sum(frqs_in_bounds)
    for frq in frqs_in_bounds:  # Ensure that all frequencies sum to 1
        frq /= normalizer
        valid_frqs.append(frq)
    return valid_frqs


def get_A(n, model):
    """
    Given the number of role models (n) and the type of model, outputs
    the list of conformity coefficients A(j) for j=k,...,n-1 (see Eq. (3)).
    """
    A_vec = []
    # Determine k, following the description above Eq. (3)
    if n % 2 == 0:
        k = int(n / 2 + 1)
    else:
        k = int((n + 1) / 2)

    if model[1] == '0':  # Conformity
        for j in range(k, n):
            A_vec.append(n - j - 0.1)
    elif model[1] == '1':  # Anti-conformity
        for j in range(k, n):
            A_vec.append(-1 * j + 0.1)
    elif model[1] == '2':  # Unbiased transmission
        for j in range(k, n):
            A_vec.append(0)
    else:
        return "Error"
    return A_vec


def frq_bias_dichotomous(n, q, A_vec):
    """
    The inputs to this function are:
        n: the number of role models.
        q: the frequency of the phenotype of interest.
        In practice, q is x1+y1 (skilled phenotype).
        A_vec: list of conformity coefficients [A(k),...,A(n-1)]; see Eq. (3).
    This function returns the right-hand side of Eq. (3).
    """
    # Determine k, following the description above Eq. (3)
    if n % 2 == 0:
        k = int(n / 2 + 1)
    else:
        k = int((n + 1) / 2)
    # Determine φ(q), stored as the variable φ, following Eq. (3)
    φ = q
    for j in range(k, n):
        # Note that A(k) is at index 0 in A_vec, so adjust index by -k
        φ += A_vec[j - k] / n * sp.binom(n, j) * (q ** j * (1 - q) ** (n - j) - q ** (n - j) * (1 - q) ** j)
    return φ


def initial_frq_social(model):
    """
    Returns the initial frequency of social learners in the population, given
    the type of model of non-cumulative culture.
    """
    if model[4] == '1':  # Finite population
        N = int(model[5:]) # Note that 'iter' was removed from the end of 'model' string (see def main)
        if model[3] == '0':  # Social learning is initially rare
            return 1 / N
        else:  # Social learning is initially common
            if N % 2 == 0:
                return (N / 2 + 1) / N
            else:
                return (N + 1) / (2 * N)
    else:  # Infinite population
        if model[3] == '0':  # Social learning is initially rare
            return 1e-7
        else: # Social learning is initially common
            return 0.5 + 1e-7


def non_cumulative(row):
    """
    Models of non-cumulative culture.
    """
    W0, γ, K, Cs, Ci, D, δ, n, model = row

    if model[0] != '0':
        print("Error")
        return None

    n = int(n)
    l = int(round(1 / γ, 0))  # Period, for periodic environmental change model
    if model[1] == '3':
        ρ = 1
    else:
        ρ = 0.4
    pr = 10  # Precision
    signal = -1  # This variable tells us when the simulation is over
    init_social_frq = initial_frq_social(model)

    # Specify initial frequencies in the order y_0, y_1, x_0, x_1
    # Note that all individuals are initially unskilled
    frqs = [1 - init_social_frq, 0, init_social_frq, 0]
    frqs = rounding(frqs)

    frqs_over_time = [frqs]
    social_over_time = [frqs[2] + frqs[3]]

    # Terms in Eqs. (1) that do not depend on frequencies or γ
    x0_const = (1 - δ) * (1 - K - Ci)
    x1_const_ind = δ * (1 - K - Ci + D)
    x1_const_soc = 1 - K - Cs + D
    y0_term = (1 - δ) * (1 - Ci)
    y1_term = δ * (1 - Ci + D)

    equil_frqs_social = []
    at_equil = False

    # Generations before we begin counting frequencies (in some models)
    prelim_gens = 50000

    # Generations in which we count frequencies (in some models)
    count_gens = 50000

    # For the random environmental change model
    env_change_list = np.random.binomial(1, γ, prelim_gens + count_gens)

    gen = 1  # Current generation
    while signal < 0:
        if model[2] == '1':  # Periodic environmental change
            if gen % l == 0:
                γ = 1
            else:
                γ = 0
        elif model[2] == '2':  # Random environmental change
            # Note that the first generation is 1 so we adjust by -1
            γ = env_change_list[gen - 1]

        # Define ξ = ρ * π(x1+y1) + (1-ρ) * φ(x1+y1)
        if ρ == 1:
            ξ = 1 - (frqs[0] + frqs[2]) ** n
        else:
            A_vec = get_A(n, model)
            ξ = (ρ * (1 - (frqs[0] + frqs[2]) ** n)
                 + (1 - ρ) * (frq_bias_dichotomous(int(n), frqs[1] + frqs[3], A_vec)))

        # Following Eqs. (1)
        x0_term = (γ + (1 - γ) * (1 - ξ)) * x0_const
        x1_term = (γ + (1 - γ) * (1 - ξ)) * x1_const_ind + (1 - γ) * ξ * x1_const_soc
        T = ((frqs[2] + frqs[3]) * (x0_term + x1_term)
             + (frqs[0] + frqs[1]) * (y0_term + y1_term))
        new_frqs = [(frqs[0] + frqs[1]) * y0_term / T,
                    (frqs[0] + frqs[1]) * y1_term / T,
                    (frqs[2] + frqs[3]) * x0_term / T,
                    (frqs[2] + frqs[3]) * x1_term / T]
        new_frqs = rounding(new_frqs)

        if model[4] == '1':  # Finite population
            N = int(model[5:]) # Note that 'iter' was removed from the end of 'model' string (see def main)
            new_frqs = finite_sample(N, new_frqs)

        if model[2] == '0':  # Deterministic environmental change with rate γ
            if at_equilibrium(frqs, new_frqs, pr):
                signal = 1  # End simulation if equilibrium is reached
        elif model[2] == '1' and model[4] == '0':  # Infinite, periodic model
            if gen >= 2 * l:
                at_equil = True  # This will become False if not at equilibrium
                equil_frqs_social = []
                for g in range(gen - l, gen):  # Generations in most recent period
                    if at_equil:
                        frqs1 = frqs_over_time[g]
                        equil_frqs_social.append(sum(frqs1[2:]))
                        # Generations from the previous period
                        frqs2 = frqs_over_time[g - l]
                        at_equil = at_equilibrium(frqs1, frqs2, pr)
                    else:
                        equil_frqs_social = []
            if at_equil:
                signal = 1  # End simulation
        elif gen == prelim_gens + count_gens - 1:  # For random environmental change or finite, periodic model
            signal = 1

        gen += 1
        frqs = new_frqs
        if model[2] == '1' and model[4] == '0':  # Infinite, periodic model
            frqs_over_time.append(frqs)
        elif model[2] != '0':  # Random environmental change model or finite, periodic model
            social_over_time.append(frqs[2] + frqs[3])

    if model[2] == '0':
        frq_social = sum(frqs[2:])
    elif model[2] == '1' and model[4] == '0':
        frq_social = sum(equil_frqs_social) / len(equil_frqs_social)
    else:
        subset = social_over_time[prelim_gens:]
        frq_social = sum(subset) / len(subset)

    new_row = list(row.copy())
    # Accounting for rounding error
    if frq_social > 1 - 1e-6:
        new_row.append("Fixation")
    elif frq_social > init_social_frq:
        new_row.append("Invasion")
    elif frq_social < 1e-6:
        new_row.append("Loss")
    else:
        new_row.append("No invasion")
    new_row.append(frq_social)
    return (new_row)


"""
Additional functions used in models of cumulative culture.
"""

def success_bias(n, j, x_vec):
    """
    The inputs to this function are:
        n: number of role models.
        j: the skill level of interest that an individual acquires under
        success-biased transmission, namely the highest skill level in the
        sample of role models.
        x_vec: the list of variant frequencies x0, x1, ...
    The output is the probability that j is the highest skill level in the sample
    of role models (which is the right-hand side of Eq. (G1) in Appendix G if j >= 1).
    """
    if j == 0:
        return x_vec[0] ** n  # All role models must have skill level zero
    else:
        x_below_and_equal = sum(x_vec[:j + 1])  # Include level j
        x_below = sum(x_vec[:j])  # Exclude level j
        # Return P(all less than or equal to j) - P(all less than j)
        return x_below_and_equal ** n - x_below ** n


def conform_given_z_all(n, z_i, z_all, model, pr):
    """
    The inputs to this function are:
        n: number of role models.
        z_i: the variant frequency of interest, equivalent to x_i in Eqs. (17)
        of Denton et al. (2022).
        z_all: a list of all variant frequencies, equivalent to bold x in
        Eqs. (17) of Denton et al. (2022).
        model: string whose second character is '0' if conformity,
        '1' if anti-conformity, '2' if unbiased transmission.
        pr: precision; used for rounding to the nearest pr decimal places.
    The output of the function is equivalent to g_i(bold x) * d(bold x) / n
    shown in Eqs. (17) of Denton et al. (2022), with their bold x given
    by z_all, their x_i given by z_i, and their d(bold x) given by A.
    """
    if model[1] == '2': return 0  # Unbiased transmission

    # See row 1 of Eq. (17b) in Denton et al. (2022)
    if round(z_i, pr) == 0 or round(z_i, pr) == n: return 0
    z_subset = [j for j in z_all if j != 0]
    z_subset.sort(reverse=True)
    z_avg = n / len(z_subset)
    if round(z_i, pr) == round(z_avg, pr): return 0

    index = 0
    denom_high = 0
    # Entry 0 is always greater than the average if z_i != z_avg
    while z_subset[index] > round(z_avg, pr):
        denom_high += z_subset[index]
        index += 1

    index = len(z_subset) - 1
    denom_low = 0
    while z_subset[index] < round(z_avg, pr):
        denom_low += 1 / (z_subset[index])
        index -= 1

    # See inequalities (18) in Denton et al. (2022)
    A_min = -1 * denom_high
    A_max = ((n / z_subset[0]) - 1) * denom_high

    if model[1] == '0':
        A = A_max - 0.1  # Conformity
    else:
        A = A_min + 0.1  # Anti-conformity
    if z_i > z_avg:
        return (z_i / denom_high) * (A / n)
    elif z_i < z_avg:
        return - (1 / z_i) * (1 / denom_low) * (A / n)


def get_states(n, m):
    """
    This function takes in n, the number of role models, and m, the number
    of phenotypes, and outputs all possible role model states.
    Credit to: Mark Tolonen, 2019, “Generate all possible lists of length N that sum to S in Python.”
    https://stackoverflow.com/questions/7748442/generate-all-possible-lists-of-length-n-that-sum-to-s-in-python
    """
    if m == 1:
        yield (n,)
    else:
        for elem in range(n + 1):
            for state in get_states(n - elem, m - 1):
                yield (elem,) + state


def frq_bias_polychotomous(n, i, q_all, model, pr):
    """
    The inputs to this function are:
        n: number of role models.
        i: the variant (skill level) of interest.
        q_all: vector of all variant frequencies in the population.
        model: string whose second character is '0' if conformity,
        '1' if anti-conformity, '2' if unbiased transmission.
        pr: precision, same as in conform_given_z_all function.
    The output is equivalent to the right-hand side of Eq. (19) in
    Denton et al. (2022), where their p is denoted by q and their
    bold x is denoted by z_all.
    """
    states = list(get_states(n, len(q_all)))  # All possible role model states
    q_i_prime = q_all[i]
    for z_all in states:
        numerator = math.factorial(n)
        denominator = 1
        for l in range(len(z_all)):
            numerator *= q_all[l] ** (z_all[l])
            denominator *= math.factorial(z_all[l])
        q_i_prime += (conform_given_z_all(n, z_all[i], z_all, model, pr)
                      * numerator / denominator)
    return q_i_prime


def cumulative(row):
    """
    Models of cumulative culture.
    """
    # Get parameters from row. Note, m gives total number of possible social
    # learner phenotypes, including skill levels 1,2,... and unskilled state
    W0, γ, K, Cs, Ci, D, δ, n, m, model = row

    if model[0] != '1':
        print("Error")
        return None

    n, m = int(n), int(m)
    pr = 10  # Precision
    signal = -1  # This variable tells us when the simulation is over

    if model[3] == '0':  # Cumulative culture is initially absent
        x_vec = [0] * m
        x_vec[0] = 1
    else:  # Cumulative culture is initially present
        x_vec = [0] * m
        x_vec[0] = 1 - (0.1)
        x_vec[3] = 0.1
    x_vec = rounding(x_vec)

    if model[1] == '3':
        ρ = 1
    else:
        ρ = 0.4

    I, d_factor, u = 0.08, 0.1, 0.001  # Fixed parameters; d_factor is beta in the paper
    γ_ = γ  # Preserve the original value as the variable γ may change

    D_vec = []  # This will store conformity coefficients
    D_vec.append(0)
    for i in range(0, m - 1):  # Index 1 corresponds to D1
        D_vec.append(D + d_factor * i)

    if model[2] != '2':
        maximum_gen = int(round(1 / γ, 0))
    else:  # Random environmental change
        maximum_gen = np.random.geometric(p=γ_)

    # Terms in Eqs. (8) that do not depend on frequencies or γ
    const_ind = 1 - K - Ci - I
    const_soc = 1 - K - Cs - I
    x0_const = (1 - δ) * (1 - K - Ci)
    x1_const_ind = (1 - u) * δ * (const_ind + D_vec[1])
    x1_const_soc = (1 - u) * (const_soc + D_vec[1])
    x2_const_ind = u * δ * (const_ind + D_vec[2])
    x2_const_soc1 = u * (const_soc + D_vec[2])
    x2_const_soc2 = (1 - u) * (const_soc + D_vec[2])

    gen = 0
    if model[2] != '0': γ = 0

    while signal < 0:
        if maximum_gen > 0:
            π_vec = []
            for i in range(m):
                π = success_bias(n, i, x_vec)
                π_vec.append(π)
            π_vec = rounding(π_vec)

            if ρ == 1:
                ξ_vec = π_vec
            else:
                φ_vec = []
                for i in range(m):
                    φ = frq_bias_polychotomous(n, i, x_vec, model, pr)
                    φ_vec.append(φ)
                φ_vec = rounding(φ_vec)

                ξ_vec = []
                for i in range(m):
                    ξ_vec.append(ρ * π_vec[i] + (1 - ρ) * φ_vec[i])
                ξ_vec = rounding(ξ_vec)

            x_terms_vec = []  # Following Eqs. (8)
            x_terms_vec.append((γ + ξ_vec[0] * (1 - γ)) * x0_const)
            x_terms_vec.append(((γ + ξ_vec[0] * (1 - γ)) * x1_const_ind
                                + (1 - γ) * ξ_vec[1] * x1_const_soc))
            x_terms_vec.append(((γ + ξ_vec[0] * (1 - γ)) * x2_const_ind
                                + (1 - γ) * (ξ_vec[1] * x2_const_soc1
                                             + ξ_vec[2] * x2_const_soc2)))
            for i in range(3, m - 1):
                x_terms_vec.append(((1 - γ) * (ξ_vec[i - 1] * u
                                               + ξ_vec[i] * (1 - u))
                                    * (const_soc + D_vec[i])))
            x_terms_vec.append(((1 - γ) * (ξ_vec[m - 2] * u * (const_soc + D_vec[m - 1])
                                           + ξ_vec[m - 1] * (1 - K - Cs + D_vec[m - 1]))))
            normalizer = sum(x_terms_vec)
            x_prime_vec = []
            for i in range(m):
                x_prime_vec.append(x_terms_vec[i] / normalizer)
            x_prime_vec = rounding(x_prime_vec)

            if model[4] == '1':  # Finite population
                N = int(model[5:])
                x_vec = finite_sample(N, x_prime_vec)
            else:  # Infinite population
                x_vec = x_prime_vec
            x_vec = rounding(x_vec)

            if gen == maximum_gen - 1:
                signal = 1
            gen += 1
        else:
            signal = 1

    mean_level_culture = 0
    for i in range(1, m):
        mean_level_culture += i * (x_vec[i])
    return ([W0, γ_, K, Cs, Ci, D, δ, n, m, model, mean_level_culture])

def main():
    """
    Running the models.
    """
    # Parameter values
    W0 = [1] # Baseline fitness
    Ci_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.49] # Fitness costs of individual learning
    K_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.49] # Fitness costs of capacity for social learning (neural circuitry)
    Cs_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.49] # Fitness costs of social learning (i.e., copying others)
    D_values = [0.1, 0.3, 0.5, 0.7] # Fitness benefits of having the skill
    δ_values = [0.1, 0.3, 0.5, 0.7] # Probabilities of learning the skill individually
    γ_values = [(1/16), (1/8), (1/4)] # Probabilities of environmental change 
    m = 6  # Total number of phenotypes: r=5 skill levels plus 1 unskilled state

    signal = -1
    count = 0
    while signal < 0:
        model = specify_model()
        if model[len(model)-1] == '0':
            signal = 1 # This will be the last iteration
        model = model[:-1]

        if model[0] == '1':  # Cumulative culture
            env_change = model[2]
            if env_change == '2':
                env_change = '1' # For the burn-in model
            burn_in_model = '0' + model[1] + env_change + '1' + '0' + model[5:]
            nc_data = {'c': [burn_in_model[0]], 'tr': [burn_in_model[1]],
                    'env': [burn_in_model[2]], 'frq': [burn_in_model[3]],
                    'fin': [burn_in_model[4]], '3': ['NA'], '4': ['NA'],
                    '5': ['NA'], '6': ['NA'], '7': ['NA'], '8': ['NA'], '9': ['NA'],
                    '10': ['NA']}
            # In next step, multiply by 4 because the burn-in period can be very slow, so it
            # is efficient to run 4 versions of each cumulative model with the same burn-in step
            # at once (namely, finite/infinite and cumulative culture initially present/absent)
            c_data = {'c': [model[0]] * 4, 'tr': [model[1]] * 4, 'env': [model[2]] * 4, 'frq': [model[3]] * 4,
                      'fin': [model[4]] * 4, '3': ['NA'] * 4, '4': ['NA'] * 4, '5': ['NA'] * 4, '6': ['NA'] * 4,
                      '7': ['NA'] * 4, '8': ['NA'] * 4, '9': ['NA'] * 4, '10': ['NA'] * 4}
            add_c_results = pd.DataFrame(c_data)
        else:
            nc_data = {'c': [model[0]], 'tr': [model[1]], 'env': [model[2]], 'frq': [model[3]],
                    'fin': [model[4]], '3': ['NA'], '4': ['NA'], '5': ['NA'], '6': ['NA'],
                    '7': ['NA'], '8': ['NA'], '9': ['NA'], '10': ['NA']}
        add_nc_results = pd.DataFrame(nc_data)
        for n in range(3, 11):
            n_values = [n]

            # Get all combinations of parameter values for non-cumulative model
            # (for the cumulative model, this is needed for the 'burn-in' period)
            v = [W0, γ_values, K_values, Cs_values, Ci_values, D_values, δ_values, n_values]
            values = pd.DataFrame(list(itertools.product(*v)), columns=['W0', 'γ', 'K', 'Cs',
                                                                        'Ci', 'D', 'δ', 'n'])
            values = values[values['K'] + values['Cs'] > values['Ci']]  # See Appendix E
            values = values[values['K'] + values['Cs'] < values['D']]  # See Appendix E

            if model[0] == '1': # Cumulative culture
                values['Model'] = burn_in_model
            else:
                values['Model'] = model

            results_df = pd.DataFrame.from_records(values.apply(non_cumulative, 1))
            results_df.columns = ['W0', 'γ', 'K', 'Cs', 'Ci', 'D',
                                  'δ', 'n', 'Model', 'Invasion', 'Freq']
            results = (results_df[results_df['Invasion'] == 'Fixation'].shape[0]
                       + results_df[results_df['Invasion'] == 'Invasion'].shape[0])
            results /= results_df.shape[0]

            if model[0] == '0':
                print("Role models:", n, "Cumulative: " + model[0],
                      "Transmission: " + model[1], "Env change: " + model[2],
                      "Frequency: " + model[3], "Finite: " + model[4], "Result:", results)
            else:
                print("Role models:", n, "Cumulative: " + burn_in_model[0],
                      "Transmission: " + burn_in_model[1], "Env change: " + burn_in_model[2],
                      "Frequency: " + burn_in_model[3], "Finite: " + burn_in_model[4], "Result:", results)

            add_nc_results[str(n)] = results

            if model[0] == '1':
                # Select parameter values that led to fixation of social learning at equilibrium
                values = results_df[results_df['Invasion'] == 'Fixation']
                values = values.drop('Invasion', axis=1)
                values = values.drop('Freq', axis=1)
                values = values.drop('Model', axis=1)
                values['m'] = m
                values['Model'] = model

                results_df = pd.DataFrame.from_records(values.apply(cumulative, 1))
                # Below there is a typo where Cl should have been Ci, but this does not affect the results
                results_df.columns = ['W0', 'γ', 'K', 'Cs', 'Cl', 'D', 'δ', 'n', 'm', 'Model', 'Mean Level']
                results = results_df['Mean Level'].sum() / results_df.shape[0]
                print("Role models:", n, "Cumulative: " + model[0], "Transmission: " + model[1],
                      "Env change: " + model[2],
                      "Frequency: " + model[3], "Finite: " + model[4], "Result:", results)

                add_c_results[str(n)][0] = results

                # To save time, we can run multiple models here as follows
                if model[4] == '0':  # If the population size were originally infinite, switch to finite (or vice versa)
                    # Choose population size N = 100
                    new_model = model[:4] + '1' + '100'
                    add_c_results['fin'][1] = '1'
                    add_c_results['fin'][3] = '1' # To match 'new_model3' introduced later
                else:
                    new_model = model[:4] + '0' + 'NA'
                    add_c_results['fin'][1] = '0'
                    add_c_results['fin'][3] = '0' # To match 'new_model3' introduced later
                values['Model'] = new_model
                results_df = pd.DataFrame.from_records(values.apply(cumulative, 1))
                results_df.columns = ['W0', 'γ', 'K', 'Cs', 'Cl', 'D', 'δ', 'n', 'm', 'Model', 'Mean Level']
                results = results_df['Mean Level'].sum() / results_df.shape[0]
                print("Role models:", n, "Cumulative: " + new_model[0], "Transmission: " + new_model[1],
                      "Env change: " + new_model[2],
                      "Frequency: " + new_model[3], "Finite: " + new_model[4], "Result:", results)
                add_c_results[str(n)][1] = results

                if model[3] == '0':  # If cumulative culture were originally absent, switch to present (or vice versa)
                    new_model2 = model[:3] + '1' + model[4:]
                    add_c_results['frq'][2] = '1'
                    add_c_results['frq'][3] = '1' # To match 'new_model3' introduced later
                else:
                    new_model2 = model[:3] + '0' + model[4:]
                    add_c_results['frq'][2] = '0'
                    add_c_results['frq'][3] = '0' # To match 'new_model3' introduced later
                values['Model'] = new_model2
                results_df = pd.DataFrame.from_records(values.apply(cumulative, 1))
                results_df.columns = ['W0', 'γ', 'K', 'Cs', 'Cl', 'D', 'δ', 'n', 'm', 'Model', 'Mean Level']
                results = results_df['Mean Level'].sum() / results_df.shape[0]
                print("Role models:", n, "Cumulative: " + new_model2[0], "Transmission: " + new_model2[1],
                      "Env change: " + new_model2[2],
                      "Frequency: " + new_model2[3], "Finite: " + new_model2[4], "Result:", results)
                add_c_results[str(n)][2] = results

                # Combine the two different types
                new_model3 = new_model2[:4] + new_model[4:]
                values['Model'] = new_model3
                results_df = pd.DataFrame.from_records(values.apply(cumulative, 1))
                results_df.columns = ['W0', 'γ', 'K', 'Cs', 'Cl', 'D', 'δ', 'n', 'm', 'Model', 'Mean Level']
                results = results_df['Mean Level'].sum() / results_df.shape[0]
                print("Role models:", n, "Cumulative: " + new_model3[0], "Transmission: " + new_model3[1],
                      "Env change: " + new_model3[2],
                      "Frequency: " + new_model3[3], "Finite: " + new_model3[4], "Result:", results)
                add_c_results[str(n)][3] = results

        if count == 0:
            add_nc_results.to_csv('results-file.csv')
        else:
            add_nc_results.to_csv('results-file.csv', mode = 'a', header=False)

        if model[0] == '1':
            add_c_results.to_csv('results-file.csv', mode='a', header=False)

        count += 1

if __name__ == '__main__':
    main()
