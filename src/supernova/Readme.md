# SuperNova Description

This document explains from a high-level how the SuperNova protocol was implemented in Arecibo.
We aim to provide a mathematical description of the protocol, as it is implemented, and highlight the differences with the original paper.

## Terminology and Concept Clarifications

Before delving into the specifics of the implementation, it's crucial to define and clarify some key terms and concepts used throughout this document:

- **Recursive SNARK**: A Recursive SNARK is a type of succinct non-interactive argument of knowledge for a circuit $F$ which can be composed with itself as $z\_{i+1} \gets F(z_i)$.
Each iteration proves the verification of a proof for $z_i$ and the correctness of $z\_{i+1}$, ensuring the proving of each step remains constant.
- **Augmented Circuit**: In the context of the SuperNova protocol, an augmented circuit refers to a circuit $F'$ composing $F$ with a circuit which partially verifies the validity of the previous output $z_i$ before running $F(z_i)$.
- **NIFS Verifier**: A non-interactive folding scheme is a protocol for efficiently updating a proof $\pi_i$ about an iterated function $z\_{i+1} \gets F(z_i)$ into a new proof $\pi\_{i+1}$, through a process referred to as "folding".
By splitting the proof into an instance/witness pair $(u,w) = \pi$, the folding verifier describes an algorithm for verifying that the $u$ component was properly updated.

## SuperNova vs. Nova

The main improvement of SuperNova, is to allow each iteration to apply one of several functions to the previous output, whereas Nova only supported the iteration of a single function.

Let $F_0, \ldots, F\_{\ell-1}$ be folding circuits with the same arity $a$.
In the context of SuperNova, this means that each $F_j$ takes $a$ inputs from the previous iteration, and returns $a$ outputs.
These circuits implement the `circuit_supernova::StepCircuit` trait, where the main differences with the existing `StepCircuit` trait are
- The circuit $F_j$ provides its `circuit_index` $j$
- The `synthesize` function upon input $z_i$ returns the next `program_counter` $\mathsf{pc}\_{i+1}$ alongside the output $z\_{i+1}$. It also accepts the (optional) input program counter $\mathsf{pc}_i$, which can be `None` when $\ell=1$. During circuit synthesis, a constraint enforces $\mathsf{pc}_i \equiv j$. In contrast to the paper, the _predicate_ function $\varphi$ is built into the circuit itself. In other words, we have the signature $(\mathsf{pc}\_{i+1}, z\_{i+1}) \gets F\_{j}(\mathsf{pc}\_{i}, z\_{i})$.

The goal is to efficiently prove the following computation:
```ignore
pc_i = pc_0
z_i = z_0
for i in 0..num_steps
	(pc_i, z_i) = F_{pc_i}(z_i)
return z_i
```

## Cycles of Curves

"Cycles of Curves" describes a technique for more efficiently verifying the output $z_i$ of the previous circuit iteration, by running the verification on a curve whose base/scalar fields are inverted.
The result is that the elliptic curve scalar multiplications in the algorithm can be computed in the "native field" of the circuit, minimizing the need for expensive non-native field arithmetic.

While the original Nova implementation allows computation to be done on both curves, the SuperNova implementation only uses the cycle curve to verify the computation performed on the primary curve.

## Prover state

The prover needs to store data about the previous function iteration. It is defined by the `supernova::RecursiveSNARK` struct. It contains:

- $i$: the number of iterations performed.
Note that the `new` constructor actually performs the first iteration, and the first call to `prove_step` simply sets the counter to 1.
- Primary curve:
	- $(\mathsf{pc}_i, z_0, z_i)$: current program counter and inputs for the primary circuit
	- $U[\ ],W[\ ]$: List of relaxed instance/witness pairs for all the circuits on the primary curve.
	These can be `None` when the circuit for that pair has not yet been executed.
	The last updated entry is the result of having folded a proof for the correctness of $z_i$.
- Secondary curve
	- $(z_0', z_i')$: Inputs for the single circuit on the secondary curve.
	- $u',w'$: Proof for the correctness of the circuit that produced $z_i'$
	- $U', W'$: Relaxed instance/witness pair into which $(u', w')$ will be folded into in the next iteration.

Due to the particularities of the cycles of curves implementation, the outputs of the circuits producing $(z_i, z_i')$ are encoded in public outputs of the proof $u'$.


## Prove Step

At each step, the prover needs to:
- Create a proof $T'$ for folding $u'$ into $U'$, producing $U'\_{next}$.
- Create a proof $(u,w)$ on the primary curve for the statements:
	- $(\mathsf{pc}\_{i+1}, z\_{i+1}) \gets F_\mathsf{pc_i}(z_i)$
	- Verifying the folding of $u'$ into $U'$ with $T'$
- Create a proof $T$ for folding $u$ into $U[\mathsf{pc}_i]$, producing $U\_{next}$
- Create a proof $(u'\_{next}, w'\_{next})$ for the verification on the secondary curve
	- Verifying the folding of $u$ into $U[\mathsf{pc}_i]$ with $T$
- Update the state of the claims
	- $U' = U'\_{next}$, $W' = W'\_{next}$
	- $U[\mathsf{pc}_i] = U\_{next}$, $W[\mathsf{pc}_i] = W\_{next}$
	- $(u',w') = (u'\_{next}, w'\_{next})$
- Save $z\_{i+1},z'\_{i+1}, \mathsf{pc}\_{i+1}$ as inputs for the next iteration.

In pseudocode, `prove_step` looks something like:

```text
if i = 0 {
	U[] = [ø;l]

	// Create a proof for the first iteration of F on the primary curve
	(pc_1, z_1), (u_1, w_1) <- Prove(F_{pc0},
		i=0,
		pc_0,
		z_0,
		_,   // z_i : z_0 is the input used
		_,   // U' : Existing accumulator is empty
		_,   // u' : No proof of the secondary curve to verify
		_,   // T' : Nothing to fold
		0,   // index of u' in U'
	)
	// The circuit output is [ vk, i=1, pc_1, z_0, z_1, U'=ø ]
	// Update state to catch up with verifier
	z_i    = z_1
	pc_i   = pc_1
	U'     = ø
	W'     = ø

	// Create proof on secondary curve
	// verifying the validity of the first proof
	z'_1, (u'_1, w'_1) <- Prove(F',
		i,
		0,      // pc is always 0 on secondary curve
		z'_0,
		_,      // z'_i : z'_0 is the input used
		_,      // U[]: all accumulators on primary curve are empty
		u_0,    // proof for z1
		_,      // T: u_0 is directly included into U[pc0]
		pc_1,   // index of u_0 in U[]
	)
	// The circuit outputs [ vk, i=1, z'_0, z'_1, U_next[] ]
	// Update state to catch up with verifier
	z_i'    = z_1'
	U[pc_1] = u_1
	W[pc_1] = w_1

	// Save the proof of F' to be folded into U' in the next iteration
	u'     = u'_1
	w'     = w'_1
} else {
	// Create folding proof for u' into U', producing U'_next
	(U'_next, W'_next), T' <- NIFS.Prove(U', W', u', w')

	// Create a proof for the next iteration of F on the primary curve
	(pc_next, z_next), (u_new, w_new) <- Prove(F_{pc_i},
		i,
		pc_i,
		z_0,
		z_i,
		[U'],
		u',
		T',
		0,     // index of u' in [U'] is always 0
	)
	// The circuit outputs [ vk, i+1, pc_next, z_0, z_next, U'_next ]
	// Update state to catch up with verifier
	z_i  = z_next
	pc_i = pc_next
	U'   = U'_next
	W'   = W'_next

	// Create folding proof for u_new into U[pci], producing U_next
	(U_next, W_next), T <- NIFS.Prove(U[pci], W[pci], u_new, w_new)

	// Create proof on secondary curve
	// verifying the folding of u_next into
	z'_next, (u'_next, w'_next) <- Prove(F',
		i,
		0,     // pc is always 0 on secondary curve
		z_0',
		z_i',
		U[],
		u_new,
		T,
		pc_i,  // Index of u_new in U[]
	)
	// The circuit outputs [ vk, i+1, z'_0, z'_next, U_next[] ]
	// Update state to catch up with verifier
	z_i'       = z'_next
	U[pc_next] = U_next
	W[pc_next] = W_next

	// Save the proof of F' to be folded into U' in the next iteration
	u'         = u'_next
	w'         = w'_next
}
```

Each iteration stops when the prover has produced a valid R1CS instance $(u',w')$ for the secondary circuit, just before folding it back into its accumulator $(U',W')$ in the next iteration.
This allows us to access the public outputs of the secondary circuit in the next iteration, or when verifying the IVC chain.

## Augmented Circuit

During each proof iteration, the circuits evaluated and proved by the prover need to be *augmented* to include additional constraints which verify that the previous iteration was correctly accumulated.

To minimize code duplication, there is only a single version of the recursive verification circuit. The circuit is customized depending on whether it is synthesized on the primary/secondary curve.

### Input Allocation

The inputs of provided to the augmented step circuit $F'_j$ are:

**Inputs for step circuit**
- $\mathsf{vk}$: a digest of the verification key for the final compressing SNARK (which includes all public parameters of all circuits)
- $i \in \mathbb{Z}\_{\geq 0}$: the number of iteration of the functions before running $F$
- $\mathsf{pc}_i \in [\ell]$: index of the current function being executed
	- **Primary**: The program counter $\mathsf{pc}_i$ must always be `Some`, and through the `EnforcingStepCircuit` trait, we enforce $\mathsf{pc}_i \equiv j$.
	- **Secondary**: Always `None`, and interpreted as $\mathsf{pc}_i \equiv 0$, since there is only a single circuit.
- $z_0 \in \mathbb{F}^a$: inputs for the first iteration of $F$
- $z_i \in \mathbb{F}^a$: inputs for the current iteration of $F$
	- **Base case**: Set to `None`, in which case it is allocated as $\[0\]$, and $z_0$ is used as $z_i$.
- $U_i[\ ] \in \mathbb{R}'^\ell$: list of relaxed R1CS instances on the other curve
	- **Primary**: Since there is only a single circuit on the secondary curve, we have $\ell = 0$ and therefore $U_i[\ ]$ only contains a single `RelaxedR1CSInstance`.
	- **Secondary**: The list of input relaxed instances $U_i[\ ]$ is initialized by passing a slice `[Option<RelaxedR1CSInstance<G>>]`, one for each circuit on the primary curve.
	Since some of these instances do not exist yet (i.e. for circuits which have not been executed yet), the `None` entries are allocated as a default instance.

To minimize the cost related to handling public inputs/outputs of the circuit, these values are hashed as $H(\mathsf{vk}, i, \mathsf{pc}_i, z_0, z_i, U_i[\ ])$.
In the first iteration though, the hash comparison is skipped, and the optional values are conditionally replaced with constrained default values.

**Auxiliary inputs for recursive verification of other the curve's circuit**
- $u \in \mathbb{R}'$: fresh R1CS instance for the previous iteration on the other curve
	- Contains the public outputs of the 2 previous circuits on the different curves.
	- **Base case -- Primary**: Set to `None`, since there is no proof of the secondary curve to fold
- $T \in \mathbb{G}'$: Proof for folding $u$ into $U[\mathsf{pc}']$.
	- **Base case -- Primary**: Set to `None`, since there is no proof of the secondary curve to fold
- $\mathsf{pc}' \in [\ell]$: index of the previously executed function on the other curve.
	- **Primary**: Always 0 since the program counter on the secondary curve is always 0
	- **Secondary**: Equal to the program counter of the last function proved on the primary curve.

These non-deterministic inputs are used to compute the circuit's outputs.
When they are empty, we allocate checked default values instead.
We also check that the computed hash of the inputs matches the hash of the output of the previous iteration contained in $u$.

**Outputs**
- $\mathsf{vk}$: passed along as-is
- $i+1 \in \mathbb{Z}\_{\geq 0}$: the incremented number of iterations
- $\mathsf{pc}\_{i+1} \in [\ell]$: index of next function to execute
- $z_0 \in \mathbb{F}^a$: passed along as-is
- $z\_{i+1} \in \mathbb{F}^a$: output of the execution $F\_{\mathsf{pc}_i}$
- $U\_{i+1}[\ ] \in \mathbb{R}'^\ell$: Updated list of Relaxed R1CS instances, reflecting the folding of $u$ into $U_i[\mathsf{pc}']$
	- **Primary**: Since no input proof was provided, we set $U_1$ to the default initial instance.

All these values should be computed deterministically from the inputs described above (even if just passed along as-is).
The actual public output is the hash of these values, to be consistent with the encoding of the inputs.

### Constraints

The circuit has a branching depending on whether it is verifying the first iteration of the IVC chain. Each branch computes the next list of instances $U\_{i+1}[\ ]$.

#### Branch: i>0 `synthesize_non_base_case`

The verification circuit first checks that the public output $u.X_0$ is equal to the hash of all outputs of the previous circuit iteration.
Note that this value is defined as a public output of the proof $u$ on the other curve.
It was simply passed along unverified by the cycle circuit to link the two circuits from the same curve.
Since the base case does not have any previous input, we only check the hash if $i>0$.
The circuit produces a bit corresponding to:

$$b\_{X_0} \gets X_0 \stackrel{?}{=} H(\mathsf{vk}, i, \mathsf{pc}_i, z_0, z_i, U_i[\ ])$$

This bit is checked later on.

The circuit extracts $U_i[\mathsf{pc}']$ by using conditional selection on $U_i[\ ]$.
This is done by computing a selector vector $s \in \{0,1\}^\ell$ such that $s\_{\mathsf{pc}'}=1$ and all other entries are 0.

The instance new folding instance $U\_{i+1}[\mathsf{pc}']$ is produced by running the NIFS folding verifier:

$$
U\_{i+1}[\mathsf{pc}'] \gets \mathsf{NIFS}.\mathsf{Verify}(\mathsf{vk}, u, U[\mathsf{pc}'], T)
$$

A new list of accumulators $U\_{i+1}[\ ]$ is then obtained using conditional selection.
This branch returns $U\_{i+1}[\ ]$, $b\_{X_0}$ as well as the selector $s$.

#### Branch: i=0 (`synthesize_base_case`)

If $i \equiv 0$, then the verification circuit must instantiate the inputs as their defaults.
Namely, it initializes a list $U_0[\ ]$ (different from the input list which is given to the previous branch) with "empty instances" (all group elements are set to the identity).

The output list of instances $U_1[\ ]$ is
- **Primary curve**: the incoming proof $u$ is trivial, so the result of folding two trivial instances is defined as the trivial relaxed instance.
- **Secondary curve**: the instance $U_0[\mathsf{pc}']$ is simply replaced with the relaxation of $u$ using conditional selection.

This branch returns $U_1[\ ]$.

#### Remaining constraints

Having run both branches, the circuit has computed
- $U\_{i+1}[\ ], b\_{X_0}, s$ from the first branch
- $U_1[\ ]$ from the second branch

- Using the bit $b\_{i=0} \gets i \stackrel{?}{=} 0$, it needs to conditionally select which list of instance to return.
	- $U\_{i+1} \gets b\_{i=0} \ \ ?\ \ U\_{1}[\ ] \ \  :\ \  U\_{i+1}[\ ]$
- Check that $(i\neq 0) \implies b\_{X_0}$, enforcing that the hash is correct when not handling the base case
	- $b\_{i=0} \lor b\_{X_0}$
- Select
	- $z_i \gets b\_{i=0} \ \ ?\ \ z_0 \ \  :\ \  z_i$
- Enforce circuit selection
	- $\mathsf{pc}\_{i} \equiv j$
- Compute next output
	- $(\mathsf{pc}\_{i+1}, z\_{i+1}) \gets F_j(z_i)$


### Public Outputs

The output at this point would be

$$
\Big (i+1, \mathsf{pc}\_{i+1}, z_0, z\_{i+1}, U\_{i+1}\Big)
$$

To keep the number of public outputs small, the outputs of the circuit are hashed into a single field element. We create this hash as $H\_{out} = H\big (\mathsf{vk}, i+1, \mathsf{pc}\_{i+1}, z_0, z\_{i+1}, U\_{next}\big)$.

We also return the hash resulting from the output on the other curve, $u.X_1$. It will be unpacked at the start of the next iteration of the circuit on the cycle curve, so we swap it and place it first. The actual public output is then.

$$
[u.X_1, H\_{out}]
$$

We can view output as the shared state between the circuits on the two curve. The list of two elements is a queue, where the last inserted element is popped out to be consumed by the verification circuit, and the resulting output is added to the end of the queue.

## Verification

After any number of iterations of `prove_step`, we can check that the current prover state is correct. In particular, we want to ensure that $(z_i, z'_i)$ are the correct outputs after having run $i$ iterations of the folding prover.

To verify that $(z_i, z'_i)$ are correct, the verifier needs to recompute the public outputs of the latest proof $u'$. Since this is the output on the secondary curve, the first entry $u'.X_0$ will be the output of the primary curve circuit producing $(\mathsf{pc}_i, z_i)$ and the accumulator $U'$ in which we will fold $u'$. The second entry $u'.X_1$ is the output of the last circuit on the secondary curve, which will have folded the proof for $(\mathsf{pc}_i, z_i)$ into $U[\ ]$.

- $u'.X_0 \stackrel{?}{=} H(\mathsf{vk}, i, \mathsf{pc}_i, z_0, z_i, U')$
- $u'.X_1 \stackrel{?}{=} H'(\mathsf{vk}, i, z'_0, z'_i, U[\ ])$

We then verify that $(u',w')$ is a satisfying circuit, which proves that all relaxed instances $U[\ ], U'$ were correctly updated through by folding proof.

We then need to verify that all accumulators $(U[\ ], W[\ ])$ and $(U', W')$ are correct by checking the circuit satisfiability.