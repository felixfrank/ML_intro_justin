# Conditional Random Variables via Neural Networks

## Why?

Often we want to specify the data we observe as a directed graphical model to model a joint distribution, i.e.

$$
\p{\bobs} = \prod_i \pp{\obs_i}{\parents{\obs_i}},
$$
where $\parents{\obs_i}$ are the "incoming" variables of $\obs_i$, i.e. $\p{\bobs_i, \parents{\bobs_i}} \neq \p{\bobs_i} \p{\parents{\bobs_i}}$.

This is quite present in current applications of neural networks, even though it is not necessarily made explicit. Of course, there are also other uses.

The question is then how we can represent the factors. Neural networks offer a convenient way to do so.

### Expressibility of the model

Note that factorizing the joint distribution with a directed graphical model does not imply that $x_k$ depends only on $\parents{\bobs_k}$. The conditional distribution of $\bobs_k$, given all the other members of $\bobs$ is not given by
$$
\pp{\bobs_k}{\bobs_\bar{k}} \ne \pp{\bobs_k}{\parents{\bobs_k}}.
$$
Rather the conditional distribution is correctly expressed with
$$
\pp{\bobs_k}{\bobs_\bar{k}} = \frac{\p{\bobs}}{\p{\bobs_\bar{k}}} = \frac{\prod_i \pp{\obs_i}{\parents{\obs_i}}}{\int \prod_i \pp{\obs_i}{\parents{\obs_i}} \mathrm{d}\obs_k}.
$$

## How?

### Representation through a neural network

Assume that we have a random variable that is specified by a set of finite parameter or sufficient statistics.

| Type      			  | Parameters               		| Set |
| ----------------------- | ------------------------------- | ------------------------------- |
| Bernoulli 			  | Rate                           | $(0, 1)$              |
| Gaussian  			  | Mean, Standard Deviation       | $\RR$, $\RR^+$ |
| Multivariate Gaussian | Mean vector, Covariance Matrix | $\RR^d, \RR^{d \times d}$ (Positive definite) |

Further assume that we have observation $\bobs$ and condition $\bcond$. We are interested in modelling $\pp{\bobs}{\bcond}$ where $\bobs \sim \bernoulli{r}$ with some rate $r$. Since $\bobs$ depends on $\bcond$ and $r$ is sufficient to represent $\bobs$, $r$ has to be a function of $\bcond$. We assume that this function can be represented through a neural network with parameters $\pars$. Hence we write:
$$
\pp{\bobs}{\bcond} = \bernoulli{r_\pars(\bcond)}.
$$
### Learning with maximum likelihood and stochastic gradient-descent

Suppose we are given a data set $\dataset = \{ (\bcond_n, \bobs_n)\}_{n=1}^N$ from which we want to estimate the conditional. We can do so by assuming aforementioned representation and then estimate $\theta$ via maximum likelihood:
$$
\begin{align}
&\quad\argmax_\theta \prod_i \ppp{\bobs_i}{\bcond_i}{\theta} \\
=&\quad\argmax_\theta \sum_i \log \ppp{\bobs_i}{\bcond_i}{\theta}.
\end{align}
$$
We have silently assumed that all samples are independently here.

If we want to employ gradient-based optimisation, we require that  $\pd{\log \ppp{\bobs}{\bcond}{\theta}}{\theta}​$  is "tractable". By that we mean that we can either

1. Evaluate it efficiently, or
2. Have an efficient unbiased estimator.

The most popular method is stochastic gradient-descent and variants thereof. Here we use the insight that
$$
\expcc{\sum_{i \in \minibatch} \pd{\log  \ppp{\bobs_i}{\bcond_i}{\theta}}{\theta}}{\minibatch}
= \sum_i \pd{\log \ppp{\bobs_i}{\bcond_i}{\theta}}{\theta},
$$
where $\minibatch​$ consists of samples drawn from $\{1, 2, \dots, N \}​$ uniformly with replacement.



-----
$$
\newcommand{\argmax}{\text{argmax}}
\newcommand{\dataset}{\mathcal{D}}
\newcommand{\minibatch}{\mathcal{B}}
\newcommand{\pars}{\theta}
%
\newcommand{\bernoulli}[1]{\text{Bernoulli}\left( #1 \right)}
\newcommand{\parents}[1]{\text{parents}\left( #1 \right)}
%
\newcommand{\p}[1]{p \left ( #1 \right)}
\newcommand{\pp}[2]{p \left(#1~|~#2 \right)}
\newcommand{\ppp}[3]{p_{#3} \left(#1~|~#2 \right)}
%
\newcommand{\pd}[2]{\frac{\partial #1}{\partial #2}}
%
\newcommand{\expc}[1]{\mathbb{E}\left [ #1 \right]}
\newcommand{\expcc}[2]{\mathbb{E}_{#2}\left [ #1 \right]}
%
\newcommand{\cond}{y}
\newcommand{\bcond}{\mathbf{\cond}}
\newcommand{\state}{s}
\newcommand{\bstate}{\mathbf{\state}}
\newcommand{\obs}{x}
\newcommand{\bobs}{\mathbf{\obs}}
\newcommand{\control}{u}
\newcommand{\bcontrol}{\mathbf{\control}}
\newcommand{\statediff}{y}
\newcommand{\bstatediff}{\mathbf{\statediff}}
\newcommand{\controldiff}{v}
\newcommand{\bcontroldiff}{\mathbf{\controldiff}}
\newcommand{\stateforward}{\mathbf{A}}
\newcommand{\controlforward}{\mathbf{B}}
\newcommand{\statecost}{\mathbf{Q}}
\newcommand{\controlcost}{\mathbf{R}}
%
\newcommand{\bSigma}{\boldsymbol{\Sigma}}
\newcommand{\bepsilon}{\boldsymbol{\epsilon}}
%
\newcommand{\mcN}{\mathcal{N}}
\newcommand{\RR}{\mathbb{R}}
$$
