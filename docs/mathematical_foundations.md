# ALPS-4B: Mathematical Foundations and Proofs

This document presents the detailed mathematical foundations, formal proofs, and statistical formulations underlying the **ALPS-4B** multi-scale Joint-Embedding Predictive Architecture.

---

## 1. Banach Contraction Refinement Convergence Proof

To coordinate slow conceptual strategic guidance $c_T$ with mid-frequency tactical representations $h_T$, ALPS-4B implements an iterative **Checker-Refinement** loop inside the Banach checker. We model the refinement operator $\mathcal{R}$ on a Banach space $\mathcal{X}$ (a complete normed vector space) equipped with metric $d(u, v) = \|u - v\|_2$.

### Definition 1.1 (Contraction Mapping)
An operator $\mathcal{R}: \mathcal{X} \rightarrow \mathcal{X}$ is a *contraction mapping* on a metric space $(\mathcal{X}, d)$ if there exists a Lipschitz constant $L \in [0, 1)$ such that for all $u, v \in \mathcal{X}$:
$$d(\mathcal{R}(u), \mathcal{R}(v)) \le L \cdot d(u, v)$$

### Theorem 1.2 (Banach Fixed-Point Theorem)
Let $(\mathcal{X}, d)$ be a non-empty complete metric space with a contraction mapping $\mathcal{R}: \mathcal{X} \rightarrow \mathcal{X}$. Then:
1. $\mathcal{R}$ admits a unique fixed point $z^* \in \mathcal{X}$ such that $\mathcal{R}(z^*) = z^*$.
2. For any initial guess $z^{(0)} \in \mathcal{X}$, the sequence $z^{(n+1)} = \mathcal{R}(z^{(n)})$ converges to $z^*$.
3. The convergence rate is geometric, satisfying:
   $$d(z^{(n)}, z^*) \le \frac{L^n}{1 - L} d(z^{(1)}, z^{(0)})$$

### Proof of Existence and Uniqueness
Choose an arbitrary initial element $z^{(0)} \in \mathcal{X}$. Define the sequence $z^{(n)}$ by $z^{(n)} = \mathcal{R}^n(z^{(0)})$.
For any $n \ge 1$:
$$d(z^{(n+1)}, z^{(n)}) = d(\mathcal{R}(z^{(n)}), \mathcal{R}(z^{(n-1)})) \le L \cdot d(z^{(n)}, z^{(n-1)})$$
By induction:
$$d(z^{(n+1)}, z^{(n)}) \le L^n \cdot d(z^{(1)}, z^{(0)})$$

To prove that the sequence $z^{(n)}$ is Cauchy, choose $m > n \ge 1$:
$$d(z^{(m)}, z^{(n)}) \le \sum_{i=n}^{m-1} d(z^{(i+1)}, z^{(i)}) \le \sum_{i=n}^{m-1} L^i \cdot d(z^{(1)}, z^{(0)}) \le d(z^{(1)}, z^{(0)}) \frac{L^n}{1 - L}$$
Since $L < 1$, as $n \rightarrow \infty$, $L^n \rightarrow 0$. Therefore, $d(z^{(m)}, z^{(n)}) \rightarrow 0$, which proves that $z^{(n)}$ is a Cauchy sequence. 
Since $(\mathcal{X}, d)$ is complete, the sequence must converge to some limit $z^* \in \mathcal{X}$:
$$\lim_{n \rightarrow \infty} z^{(n)} = z^*$$

By the continuity of the contraction mapping $\mathcal{R}$:
$$\mathcal{R}(z^*) = \mathcal{R}(\lim_{n \rightarrow \infty} z^{(n)}) = \lim_{n \rightarrow \infty} \mathcal{R}(z^{(n)}) = \lim_{n \rightarrow \infty} z^{(n+1)} = z^*$$
This proves that $z^*$ is a fixed point.

To prove uniqueness, assume there exists another fixed point $y^* \in \mathcal{X}$ such that $\mathcal{R}(y^*) = y^*$.
$$d(z^*, y^*) = d(\mathcal{R}(z^*), \mathcal{R}(y^*)) \le L \cdot d(z^*, y^*)$$
Since $L < 1$, this inequality can only hold if $d(z^*, y^*) = 0$, implying $z^* = y^*$. $\blacksquare$

---

## 2. Sliced Isotropic Gaussian Regularization (SIGReg)

### 2.1 The Cramér-Wold Theorem
Direct density estimation or distribution alignment in high-dimensional spaces $\mathbb{R}^D$ is computationally intractable and suffers from the curse of dimensionality. To stabilize JEPA without momentum teachers, ALPS-4B leverages the **Cramér-Wold Theorem**.

#### Theorem 2.1 (Cramér-Wold)
A multivariate probability distribution $\mathbb{P}$ on $\mathbb{R}^D$ is uniquely and completely determined by the family of its 1D projections:
$$\{\mathbb{P}_a : \mathbb{P}_a(x) = \mathbb{P}(\{\mathbf{z} \in \mathbb{R}^D : \mathbf{z} \cdot a \le x\}), \; a \in \mathbb{S}^{D-1}\}$$
where $\mathbb{S}^{D-1}$ is the unit hypersphere in $\mathbb{R}^D$.

Therefore, to enforce that the latent representation matrix $\mathbf{z}$ follows an isotropic Gaussian distribution $\mathcal{N}(0, I_D)$, it is mathematically sufficient to draw random unit projection vectors $a \in \mathbb{S}^{D-1}$ and verify that the 1D projected variables $y = \mathbf{z}a$ follow a standard normal distribution $\mathcal{N}(0, 1)$.

### 2.2 The Epps-Pulley Statistic Formulation
To differentiably measure the discrepancy between the empirical distribution of $y$ and the standard normal $\mathcal{N}(0, 1)$, we employ the Epps-Pulley test statistic. This statistic measures the weighted $L^2$-distance between the empirical characteristic function (ECF) and the Gaussian characteristic function.

#### Definition 2.2
Let $Y_1, \dots, Y_n$ be a sample of standardized projected variables. The empirical characteristic function is:
$$\psi_n(t) = \frac{1}{n} \sum_{j=1}^n e^{i t Y_j}$$
The characteristic function of the standard normal $\mathcal{N}(0, 1)$ is:
$$\psi_0(t) = e^{-t^2/2}$$
Using a Gaussian weight density $\varphi_\beta(t) = \frac{1}{\beta\sqrt{2\pi}} \exp\left(-\frac{t^2}{2\beta^2}\right)$, the Epps-Pulley statistic $T_{n,\beta}$ is defined as:
$$T_{n,\beta} = n \int_{-\infty}^{\infty} |\psi_n(t) - e^{-t^2/2}|^2 \varphi_\beta(t)\, dt$$

#### Derivation of the Analytical Form
Expanding the integrand:
$$|\psi_n(t) - e^{-t^2/2}|^2 = \left( \psi_n(t) - e^{-t^2/2} \right) \left( \overline{\psi}_n(t) - e^{-t^2/2} \right) = \psi_n(t)\overline{\psi}_n(t) - e^{-t^2/2}(\psi_n(t) + \overline{\psi}_n(t)) + e^{-t^2}$$
Substituting the ECF:
$$\psi_n(t)\overline{\psi}_n(t) = \frac{1}{n^2} \sum_{j=1}^n \sum_{k=1}^n e^{i t (Y_j - Y_k)}$$
$$\psi_n(t) + \overline{\psi}_n(t) = \frac{2}{n} \sum_{j=1}^n \cos(t Y_j)$$

Integrating each term against the Gaussian weight function:
1. **Term 1**:
   $$\int_{-\infty}^{\infty} \frac{1}{n^2} \sum_{j=1}^n \sum_{k=1}^n e^{i t (Y_j - Y_k)} \varphi_\beta(t)\, dt = \frac{1}{n^2} \sum_{j=1}^n \sum_{k=1}^n \exp\left( -\frac{\beta^2}{2}(Y_j - Y_k)^2 \right)$$
2. **Term 2**:
   $$\int_{-\infty}^{\infty} -\frac{2}{n} \sum_{j=1}^n \cos(t Y_j) e^{-t^2/2} \varphi_\beta(t)\, dt = -2 (1 + \beta^2)^{-1/2} \frac{1}{n} \sum_{j=1}^n \exp\left( -\frac{\beta^2 Y_j^2}{2(1 + \beta^2)} \right)$$
3. **Term 3**:
   $$\int_{-\infty}^{\infty} e^{-t^2} \varphi_\beta(t)\, dt = (1 + 2\beta^2)^{-1/2}$$

Multiplying by $n$ yields the final closed-form Epps-Pulley statistic:
$$T_{n,\beta} = \frac{1}{n} \sum_{j=1}^n \sum_{k=1}^n \exp\left(-\frac{\beta^2}{2}(Y_j-Y_k)^2\right) - 2 \left(1 + \beta^2\right)^{-1/2} \sum_{j=1}^n \exp\left(-\frac{\beta^2 Y_j^2}{2(1+\beta^2)}\right) + \frac{n}{\sqrt{1+2\beta^2}}$$
This closed-form formulation is fully differentiable and has linear complexity $\mathcal{O}(NM)$ per batch, serving as a highly stable stabilizer.

---

## 3. Energy-Based Model Landscape

ALPS-4B binds the predictions of its hierarchical layers under a unified energy landscape. Let $x$ be the raw visual sensory stream and $a$ be the immediate actions.
The per-layer energy terms correspond to the prediction mean-squared errors:
- **Operative Energy**:
  $$E_O(z_t, h_T, a_t) = \| \text{Pred}_O(z_t, \text{stop\_grad}(h_T)) - z_{t+1} \|_2^2$$
- **Tactical Energy**:
  $$E_T(h_T, c_T) = \| \text{Pred}_T(h_T, \text{stop\_grad}(c_T)) - h_{T+1} \|_2^2$$
- **Strategic Energy**:
  $$E_S(c_T) = \| \text{Pred}_S(c_T) - c_{T+1} \|_2^2$$

The unified total energy is the weighted sum of these per-layer energies:
$$E_{\text{total}}(x, a) = \alpha \cdot E_S + \beta \cdot E_T + \gamma \cdot E_O$$
The optimal plan $a^*$ is obtained by finding the action trajectory that minimizes the total energy landscape:
$$a^* = \arg\min_{a} E_{\text{total}}(x, a)$$
This provides a rigorous framework for unified, multi-scale planning in latent space.
