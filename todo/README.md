# Full-Covariance Gaussian Posterior
## Correlation estimation

Instead of using a Neural Net for estimating the covariance directly, I propose to obtain the estimation of the correlation matrix using the neural network $f(.; \theta_{Cor})$.

- Input: $z \in \mathcal{Z}$
- 

$f(z; \theta_{corr}): Z \rightarrow C$ where $C \in \mathbb{R}^{N \times T}$

$corr \in \{c_1, c_2, c_3, c_5, \dots, c_T \}$

$$
       corr^{low} =   \begin{bmatrix}
                    -   & 0   & 0   & 0 \\
                    c_1 & -   & 0   & 0 \\
                    c_2 & c_3 & -   &  0\\
                    c_4 & c_5 & c_6 & - \\
                \end{bmatrix}
$$