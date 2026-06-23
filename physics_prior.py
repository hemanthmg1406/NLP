"""Static physics symbol prior for quantum-physics papers.

Used as a last-resort fallback in find_symbol_definitions (Level 3) when both
local context and the paper-level scan return no definition.

Design principle: only retain entries whose meaning is THE SAME across every
quantum physics subfield (quantum optics, condensed matter, quantum information,
open quantum systems). Any symbol whose meaning shifts between subfields has
been removed — an empty definition is better than a wrong one.

Removed: I, S, T, H, E, N, k, phi, Phi, psi, Psi, sigma, rho (generic), mu,
lambda, Lambda, Omega, alpha, eta, theta, Theta, xi, Xi, chi, zeta, Pi, Delta,
delta, epsilon, varepsilon, Gamma, tau, mathcal_E, mathcal_S, mathcal_T,
mathcal_C, mathcal_P, mathcal_V, mathcal_U, mathcal_F, mathcal_N, mathcal_Z,
mathcal_H, hat_O.

Kept: physical constants (hbar, k_B), Pauli matrices (labelled subscripts),
bosonic ladder operators (hat_a/b/c, hat_n), density operator (hat_rho),
Hamiltonian operator (hat_H), unitary (hat_U), momentum/position (hat_p/x),
Lindblad/dissipator (mathcal_L, mathcal_D), Hilbert space (mathbb_H),
symmetry-classified sign-subscripted operators (Xi_+/-etc.),
and stable subscripted forms (omega_0, kappa_0, g_0, etc.).

Non-generative: this is a static lookup table, not a model output.
Logged in audit trail as symbol_def_source='physics_prior'.
"""

# Keys use the same normalized form as symbols_extract.normalize_identifier:
# no backslash, decorator prefix joined with underscore (e.g. hat_H, mathcal_L).
PHYSICS_PRIOR = {
    # ------------------------------------------------------------------
    # Physical constants — universal, no subfield ambiguity
    # ------------------------------------------------------------------
    "hbar":         "reduced Planck constant (h / 2pi)",
    "k_B":          "Boltzmann constant",

    # ------------------------------------------------------------------
    # Pauli matrices — labelled subscripts make them unambiguous
    # ------------------------------------------------------------------
    "sigma_x":      "Pauli X matrix",
    "sigma_y":      "Pauli Y matrix",
    "sigma_z":      "Pauli Z matrix",

    # ------------------------------------------------------------------
    # Bosonic ladder operators — standard across quantum optics and
    # condensed matter; hat notation is the universal convention
    # ------------------------------------------------------------------
    "hat_H":        "Hamiltonian operator",
    "hat_a":        "bosonic annihilation operator",
    "hat_b":        "bosonic annihilation operator (second mode)",
    "hat_c":        "bosonic annihilation operator (third mode)",
    "hat_n":        "number operator",
    "hat_U":        "unitary evolution operator",
    "hat_rho":      "density operator",
    "hat_p":        "momentum operator",
    "hat_x":        "position operator",

    # ------------------------------------------------------------------
    # Superoperators — stable within open quantum systems context
    # ------------------------------------------------------------------
    "mathcal_L":    "Lindblad superoperator or Lagrangian",
    "mathcal_D":    "dissipator superoperator",

    # ------------------------------------------------------------------
    # Hilbert space
    # ------------------------------------------------------------------
    "mathbb_H":     "Hilbert space",

    # ------------------------------------------------------------------
    # Sign-subscripted symmetry operators (non-Hermitian / topological physics).
    # These are always classified this way when the subscript is + or -.
    # _plus / _minus because normalize_identifier maps _+ → _plus.
    # ------------------------------------------------------------------
    "Xi_plus":          "particle-hole symmetry operator (positive sign)",
    "Xi_minus":         "particle-hole symmetry operator (negative sign)",
    "mathcal_T_plus":   "time-reversal symmetry operator (positive variant)",
    "mathcal_T_minus":  "time-reversal symmetry operator (negative variant)",
    "Gamma_plus":       "chiral symmetry operator (positive variant)",
    "Gamma_minus":      "chiral symmetry operator (negative variant)",

    # ------------------------------------------------------------------
    # Subscripted forms — specific enough to be unambiguous
    # ------------------------------------------------------------------
    "rho_0":        "initial or equilibrium density matrix",
    "omega_0":      "resonance frequency",
    "omega_c":      "cutoff frequency or cavity frequency",
    "gamma_0":      "bare decay rate",
    "kappa_0":      "intrinsic decay rate",
    "phi_0":        "magnetic flux quantum or equilibrium phase",
    "H_0":          "unperturbed Hamiltonian",
    "E_n":          "n-th energy level",
    "E_0":          "ground-state energy",
    "g_0":          "single-photon coupling rate",
    "kappa":        "decay rate / coupling strength",
}
