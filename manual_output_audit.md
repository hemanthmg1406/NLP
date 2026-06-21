# Manual Audit of `output.json`

## 2505.20373 - Magnetic field / SPION imaging

### Line 1: PARTIAL

- Equation and meaning are accurate.
- `symbols` is empty but should define `B(r)`, `B_k(r)`, `r`, and `k`.
- Relation to line 2 being `none` is reasonable.

### Line 2: PARTIAL

- Equation is faithfully extracted.
- Meaning begins with irrelevant text: `"sample wide basis at best"`.
- Missing definitions: `m_C`, `M_C`, `m_P`, and `k_B`.
- `H = magnetization field` follows the paper, though "applied magnetic field" would be clearer.
- The important interpretation is absent: at magnetic saturation, `L -> 1`, hence `m_C ≈ N m_P`.

**Paper quality:** Equations good; meaning/symbol completeness poor.

## 2503.12870 - Clifford hierarchy and hypergraph states

### Line 1: PARTIAL

- Clifford-hierarchy equation is correct.
- `E = subsets in V` is wrong contamination from the hypergraph discussion. Here `E` is a Pauli operator in `P_n`.
- `P = qubit Pauli operator` confuses `P` with the Pauli group `\mathcal P_n`.
- Meaning is grammatically incomplete.

### Line 2: PARTIAL

- Controlled-Z action is correct.
- `"Z is order multiple controlled"` is malformed and incomplete.
- Missing `x`, `i_j`, `k`, and the computational-basis interpretation.
- Strong relation to line 3 is conceptually plausible; relation to line 1 is weaker than reported.

### Line 3: PARTIAL

- Hypergraph-state equation is correct.
- `A = the first` is plainly wrong.
- `A` is a hyperedge, `E` the hyperedge set, and `P_ψ` the corresponding Boolean polynomial.
- `"Introduced in the context of controlled-controlled-Z"` is too narrow; it defines a general hypergraph state.

### Line 4: PARTIAL

- Noise-tailoring derivation is correctly copied.
- Meaning is too generic and misses the main result: twirling removes off-diagonal terms in the `|φ_b>` basis and produces `ρ_p`.
- `Z` does not occur in the extracted equation and should not be defined here.
- Missing `U_t`, `X_φ^a`, `φ_b`, `p_a`, and `ρ_p`.

### Line 5: PARTIAL

- Convolution equation and theorem title are correct.
- `p = first is that the support size` is wrong.
- `p_a` is the diagonal noise distribution; `μ_u` is its self-convolution.
- `Z` is unrelated contamination.
- Potential relation to line 7 is reasonable but incomplete because lines 6-7 derive the theorem.

### Line 6: PARTIAL

- Long CNOT derivation is faithful.
- Meaning captures the operation but not the purpose: it lowers the Boolean-polynomial degree and prepares the convolution protocol.
- Symbol definitions for `p` and `Z` are wrong.
- Potential relation to line 7 should be a strong continuation relation.

### Line 7: FAIL

- Starts with `=`, so it is not a standalone equation.
- The source equation group begins with `Prob(u') = ...`; that beginning was dropped.
- Meaning is locally relevant, but symbols are contaminated.
- This should be merged with the preceding probability-expression row or retain its left-hand side.

**Paper quality:** Equations mostly faithful, but symbol extraction is unreliable and line 7 is structurally broken.

## 2406.10156 - VQLS / Poisson equation

### Line 1: PARTIAL

- Equation is correct.
- Meaning discusses `b` and `U` rather than explaining `A = Σ_l c_l A_l`.
- `A = the DPEM` is too specific here; this is the general VQLS matrix decomposition.
- The `equivalent` relation to line 2 is wrong: they encode different inputs.

### Line 2: PARTIAL

- Faithfully reproduces the figure's `b = Σ_l U_l`.
- The paper prose actually describes a preparation circuit `U` composed of sub-unitaries, so the figure equation is dimensionally/semantically suspicious.
- This should be flagged as a source-level anomaly rather than treated as an ordinary vector decomposition.

### Line 3: PARTIAL

- Discrete Poisson equation matrix is correct.
- Meaning `"The Poisson"` is truncated and poor.
- Should identify `A` as the 1D discrete Poisson equation matrix with diagonal `2` and nearest-neighbor `-1`.

### Line 4: PARTIAL

- Four-dimensional Pauli decomposition is correct.
- Meaning is vague.
- Missing definition of `Y`; `I` and `X` definitions are incomplete.

### Line 5: PARTIAL

- Eight-dimensional Pauli decomposition is correct.
- Same missing/weak symbol definitions as line 4.
- Relation between lines 4 and 5 should be "same decomposition method at different system sizes," not merely similarity scores.

### Line 6: PARTIAL

- HED linear combination is correct.
- Meaning should explain that it uses four constant-count submatrices.
- Missing definitions of `L_1`, `L_2`, and `L_3`.

### Line 7: GOOD/PARTIAL

- Equation and local meaning are correct.
- `I = The matrix` and `X = gate acting` are low-quality definitions.
- Should say `I` acts on the other qubits and `X` acts on the least-significant qubit.

**Paper quality:** Equation fidelity is good; generated meanings are consistently weak. Lines 1-2 are also duplicated from a figure despite fuller equations/text elsewhere.

## 2505.04321 - Gaussian states and quantum metrology

### Line 1a: PARTIAL

- Commutation relation is copied correctly.
- Missing definitions for `b_i`, `b_j`, and `Ω_ij`.
- The source convention itself appears unusual and should be preserved but optionally flagged.

### Line 1b: PARTIAL

- Symplectic matrix is correct.
- Relation to 1a is complementary/definition, not `equivalent`.

### Line 2a: FAIL

- The Wigner-function equation is followed by unrelated equation labels and symbols: `ξ∈R² ρ W(x) ρ θ ρ_θ x̄_θ σ_θ`.
- This is a DOM equation-group boundary failure.
- `R` and `θ` definitions do not belong to this equation.
- The source formula itself also looks malformed for a two-mode Wigner transform, so fidelity and source validity should be reported separately.

### Line 2b: PARTIAL

- Uncertainty condition is correct.
- `sigma = definite covariance matrix` is ungrammatical.
- Missing `Ω`; meaning should state that the condition enforces physicality.

### Line 2c: PARTIAL

- Gaussian Wigner-function form is copied correctly.
- Relation to 2a as a special case is correct.
- Missing definitions for `x`, `x̄`, and `σ`.
- The `(2π)^4` normalization is source-faithful but potentially conventionally questionable.

### Line 3a: FAIL

- Three separate subequations were concatenated with artifacts such as `U→`.
- `S = symplectic matrix` is wrong here: `\mathcal S(R)` is the two-mode squeezing unitary.
- The meaning discusses generic Gaussian unitaries rather than the actual state construction.

### Line 3b: PARTIAL

- Thermal-state equation is correct.
- Missing `n`, `\bar n_i`, and the Fock-state probability interpretation.
- Strong reference relation to 3a is correct.

**Paper quality:** Only 1a, 1b, 2b, 2c, and 3b have structurally usable equations. Lines 2a and 3a are major extraction failures.
