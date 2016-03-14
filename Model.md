# Modeling Peer-Evaluation process

## Peergrade model

In this model there are $H$ handins and $G$ graders.
A handin $h \in H$ have an observed score $S_{hg}$ defined by a grader $g \in G$ who have graded the ha. 

I model that a handin score $S_{hg}$ is defined by a latent true score $T_{h}$ of the handin and a bias $B_{g}$ of the grader

\begin{align}
S_{hg} = T_{h} + B{g} + \epsilon
\end{align}

An unexplained iid residue $\epsilon$ is also added to model.

This model assumes that there is an underling true score for all handins and a grading bias of all graders.

With this the true score $S_{hg}$ can be assumed to be drawn from a single normal distribution

\begin{align}
S_{hg} &\sim N(T_{h}+B_{g}, \epsilon)
\end{align}

Defining the model as Bayesian hierarchical model I assume that the true score of a handin $h$ and bias of grader $g$ is drawn from two independent normal distributions. $\epsilon$ can be assumed to be drawn from a Gamma distribution

\begin{align} 
T_h &\sim N(T_h|\mu_h, \tau_h) \\
B_g &\sim N(B_g|\mu_g, \tau_g) \\
\epsilon &\sim \Gamma(\epsilon| \alpha_{\epsilon},\beta_{\epsilon}) 
\end{align}

The parameters $\mu_h$, $\mu_g$, $\tau_h$ and $\tau_g$ are unknown in the model so a new layer is added to the hierarchical model where the means $\mu$ and precisions $\tau$ are drawn from a two normal-gamma distribution:

\begin{align}
\mu_h, \tau_h &\sim N(\mu_h|\gamma_h, \tau_h*\lambda_h)\Gamma(\tau_h|\alpha_h, \beta_h) \\
\mu_g, \tau_g &\sim N(\mu_g|\gamma_g, \tau_g*\lambda_g)\Gamma(\tau_g|\alpha_g, \beta_g)
\end{align}

Where the parameters $\gamma_h$, $\tau_h$, $\lambda_h$, $\alpha_h$ $\beta_h$, $\gamma_g$, $\tau_g$, $\lambda_g$, $\alpha_g$, $\beta_g$, $\alpha_{\epsilon}$ and $\beta_{\epsilon}$ are hyper-parameters in the model. It does not make sense to add more complexity to the mode so the values of these hyper-parameters are set to values the are reasonable values according to the observed data 

##Markov Chain Monte Carlo

An Markov Chain Monte Carlo (MCMC) algorithm can be used to find the posterior distribution. In this case the latent score of a handing and bias of a grader. There exist multiple algorithms for solving such a problem such. The thesis will only look into random walk Monte Carlo methods (why?). 

###Gibbs Sampling

For the case of hierarchical model with multiple dimension of parameters the most efficient random walk method are are Gibbs sampling as it does not require any 'tuning' and sampling each variable in turn compared to other methods. Additionally it can Incorporate other methods in the sampling process. The main problem with the Gibbs sampling methods is that it requires the conditional probability of the target distributions. These probability can be hard, even impossible, to calculate.

Finding the conditional probability can be done in different ways but the easiest part is to find the full conditional probability of all the parameters. From here, each conditional probability can be found.

The full conditional probability:

\begin{align}
\begin{split}
p(T_{h},B_{g},\epsilon,\mu_g,\tau_g,\mu_h,\tau_h|S_{hg},...) \propto& \prod_{g \in \mathcal{G}(h)}[N(S_{hg}|T_{h}+B_{g},\epsilon)] \\
& \times \prod_{h \in \mathcal{H}(g)}[N(S_{hg}|T_{h}+B_{g},\epsilon)] \\
& \times \Gamma(\epsilon|\alpha_{\epsilon},\beta_{\epsilon}) \\
& \times N(\mu_g|\gamma_g, \tau_g*\lambda_g) \Gamma(\tau_g|\alpha_g, \beta_g) \\
& \times N(\mu_h|\gamma_h, \tau_h,\lambda_h) \Gamma(\tau_h|\alpha_h, \beta_h)
\end{split}
\end{align}

where $\mathcal{G}(h)$ are the set of graders who have graded the handin $h$ and $\mathcal{H}(g)$ are the handins that grader g have graded

We can then find the conditional probability for the different parameters.

For $T_{h}$ we have

\begin{align}
\begin{split}
p(T_h|S_{hg},...) \propto& \prod_{g \in \mathcal{G}(h)}[N(S_{hg}|T_h+B_g,\epsilon] \\
& \times N(T_h|\mu_h, \tau_h)
\end{split}
\end{align}

For $B_{g}$ we have

\begin{align}
\begin{split}
p(B_g|S_{hg},...) \propto& \prod_{h \in \mathcal{H}(g)}[N(S_{hg}|T_h+B_g,\epsilon] \\
& \times N(B_g|\mu_g, \tau_g)
\end{split}
\end{align}


For $\epsilon$:

\begin{align}
\begin{split}
p(\epsilon|S_{hg},...) \propto& \prod_{h \in \mathcal{H}(g)}\prod_{g \in \mathcal{G}(h)}[N(S_{hg}|T_h+B_g,\epsilon] \\
& \times \Gamma(\epsilon|\alpha_{\epsilon}, \beta_{\epsilon})
\end{split}
\end{align}

For $\mu_h$ and $\tau_h$ we have

\begin{align} \label{eq:conh}
\begin{split}
p(\mu_h,\tau_h|S_{h},...) \propto& N(T_{h}|\mu_h,\tau_h) \\
& \times N(\mu_h|\gamma_h, \tau_h*\lambda_h) \Gamma(\tau_h|\alpha_h, \beta_h)
\end{split}
\end{align}

and $\mu_g$ and $\tau_g$ we have

\begin{align} \label{eq:cong}
\begin{split}
p(\mu_h,\tau_h|S_{h},...) \propto& N(B_{g}|\mu_g,\tau_g) \\
& \times N(\mu_g|\gamma_g, \tau_g*\lambda_g) \Gamma(\tau_g|\alpha_g, \beta_g)
\end{split}
\end{align}

These conditional probability can be reduced by using the probability density function (PDF of the distributions that the probability consist of. The aim is to reduce it down to a form that defines a single distribution.

The normal distribution have the PDF:

\begin{align}
f(x|\mu,\tau) =& \frac{\sqrt{\tau}}{\sqrt{2\pi}}e^{-\tau\frac{(x-\mu)^2}{2}}
\end{align}

Gamma distribution:

\begin{align}
f(x|\alpha,\beta) =& \frac{\beta^\alpha}{\Gamma(\alpha)}\tau^{\alpha-\frac{1}{2}}e^{-\beta\tau}
\end{align}

and the normal-gamma distributions PDF is:

\begin{align}
f(\mu,\tau|\gamma_g, \lambda_g, \alpha_g, \beta_g) =& \frac{\beta^\alpha\sqrt{\lambda}}{\Gamma(\alpha)\sqrt{2\pi}}\tau^{\alpha-\frac{1}{2}}e^{-\beta\tau}e^{\frac{-\lambda\tau(\mu-\gamma)^2}{2}}
\end{align}