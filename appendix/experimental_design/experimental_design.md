# Experimental Design for Left Ventricular Biomechanics

To ensure thorough coverage of the input parameter space, it is important to design a framework capable of exploring every possible combination as effectively as possible. Experimental design refers to the systematic planning of physical experiments to efficiently explore a parameter space and extract meaningful insights while minimizing both the number of simulations and time. Basically, a good experimental design should aim to minimize the number of runs needed to acquire as much information as possible (Fang et al., 2005). To achieve this, various techniques have been developed and applied, four of which were initially considered and are briefly described below.

---

## Rectangular Grid

Rectangular grid sampling involves discretizing each dimension of the input space into evenly spaced intervals and evaluating all possible combinations of these values across dimensions (Young, 1991).  

For a \(d\)-dimensional parameter space with \(n\) discretization points per dimension, the total number of sample points is given by \(n^d\).  
Let each parameter $\(q_i \in [a_i, b_i]\)$, for \(i = 1, \dots, d\). Then, the values for \(q_i\) are computed as:

$$
q_i^{(j)} = a_i + \frac{j - 1}{n - 1}(b_i - a_i), \quad j = 1, \dots, n.
$$

The Cartesian product of these values across all dimensions forms the complete grid of input vectors.  
While straightforward, this method presents significant limitations for high-dimensional parameter spaces due to the curse of dimensionality.  
For a 4-dimensional parameter space, achieving reasonable resolution requires \(n^4\) sample points.  
Even modest discretization (e.g., \(n = 20\)) yields 160,000 required samples, making the approach computationally prohibitive for **ABAQUS** simulations.  
Because we defined our number of simulations at 10,000, rectangular grid sampling is impractical for this study's objectives.

---

## Uniform Distribution

Uniform random sampling, also known as **Monte Carlo sampling**, consists of independently drawing each parameter from a uniform distribution over its domain:

$$
q_i^{(j)} \sim \mathcal{U}(a_i, b_i), \quad j = 1, \dots, N.
$$

This results in \(N\) independent samples:

$$
\mathbf{q}^{(j)} = (q_1^{(j)}, q_2^{(j)}, \dots, q_d^{(j)}).
$$

Although conceptually simple, this approach lacks the stratification mechanisms of more advanced sampling methods that are capable of better capturing nonlinear relationships between parameters.

---

## Sobol Sequence

**Sobol sequences** are quasi-random low-discrepancy sequences designed to generate points that uniformly fill a multidimensional space (Lemieux, 2009).  

Mathematically, a Sobol sequence generates a sequence of points:

$$
\mathbf{x}_i = (x_i^{(1)}, x_i^{(2)}, \dots, x_i^{(d)}), \quad i = 1, 2, \dots, N,
$$

where each component \(x_i^{(j)} \in [0, 1]\) is constructed using direction numbers and bitwise operations involving the binary representation of the index \(i\).  
The goal is to minimize the discrepancy \(D_N\) of the point set, quantifying how uniformly samples cover the domain:

$$
D_N = \sup_{B \subset [0,1]^d} \left| \frac{A(B; N)}{N} - \lambda(B) \right|.
$$

Despite their theoretical advantages, Sobol samples are less effective here, as our physiological parameter space involves complex dependencies.  
For locally nonlinear or abrupt biomechanical responses, Sobol points may fail to allocate sufficient resolution to critical regions.  
However, other studies have achieved satisfactory results using Sobol sampling (Noe et al., 2019).

---

## Latin Hypercube Sampling

**Latin hypercube sampling (LHS)** is a stratified sampling technique introduced by McKay et al. (1979) to generate diverse, well-distributed parameter combinations across the input space of reduced material parameters  
\(\mathbf{q} = (q_1, q_2, q_3, q_4) \in \mathbb{R}^4.\)

Let the parameter space be defined as a \(d\)-dimensional hypercube, with each parameter \(q_i \in [a_i, b_i]\), for \(i = 1, \ldots, d\).  
Suppose we wish to generate \(N\) samples.  
In LHS, each interval \([a_i, b_i]\) is divided into \(N\) equally probable, non-overlapping intervals.  
Then, one value is randomly sampled from each interval without replacement.  

Formally:

$$
P_{ij} = [a_i + \frac{j-1}{N}(b_i - a_i),\, a_i + \frac{j}{N}(b_i - a_i)], \quad j = 1, \dots, N.
$$

For each \(i\), randomly permute the indices \(\{1, 2, \dots, N\}\) to obtain \(\pi_i\).  
Then sample \(x_i^{(j)} \in P_{i,\pi_i(j)}\) uniformly and construct:

$$
\mathbf{q}^{(j)} = (x_1^{(j)}, x_2^{(j)}, \dots, x_d^{(j)}), \quad j = 1, \dots, N.
$$

This ensures each marginal distribution is uniformly sampled while maintaining better global coverage than standard Monte Carlo sampling.  
For correlated variables, however, LHS may show limited advantage (Press et al., 1992).  
Owen (1997) proved that the variance of an \(n_t\)-point LHS sample, \(\sigma^2_{\text{LHS}}\), is related to the variance of a traditional \(n_t\)-point Monte Carlo sample, \(\sigma^2_{\text{MC}}\), by:

$$
\sigma^2_{\text{LHS}} \leq \frac{n_t}{n_t -1} \sigma^2_{\text{MC}}, \quad \text{for } n_t > 1.
$$

Thus, an LHS with \(n_t\) points never leads to a variance greater than that of an MC sample with \(n_t-1\) points.

---

![A comparison of different design choices for the training inputs. The plots show 100 points \(q_i\) in the two-dimensional space \([0, 1]^2\) using different design choices: (a) regular grid, (b) sampled from a uniform distribution, (c) Latin hypercube design, and (d) Sobol sequence.](images/exp_des.png)

---

Given these considerations, the input parameter vector \(\mathbf{q} = (q_1, q_2, q_3, q_4)\) was sampled from the uniform domain \([0.1, 5]^4\) using Latin hypercube sampling with \(N = 10{,}000\) samples.  
Parameter bounds were adopted from previous literature (Noe et al., 2019; Davies et al., 2019), where they were shown to capture physiologically realistic variations in myocardial material properties.  
Latin hypercube sampling ensures that our surrogate models are trained on a dataset broadly covering the biomechanical response space of the left ventricle.
