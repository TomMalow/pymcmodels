#Modeling Peer-Evaluation process

## Peergrade model

In this model there are $H$ handins and $G$ graders.
A handin $h \in H$ have an observed score $S_{hg}$ defined by a grader $g \in G$ who have graded the ha. 

I model that a handin score $S_{hg}$ is defined by a latent true score $T_{h}$ of the handin and a bias $B_{g}$ of the grader

\begin{align}
S_{hg} = T_{h} + B{g}
\end{align}

This model assumes that there is an underling true score for all handins and a grading bias of all graders.

Defining the model as Bayesian hierarchical model I assume that the true score of a handin $h$ and bias of grader $g$ is drawn from two independent normal distributions.

\begin{align} 
T_h &\sim N(\mu_h, \tau_h) \\
B_g &\sim N(\mu_g, \tau_g)
\end{align}

With this the true score $S_{hg}$ can be assumed to be drawn from a single normal distribution instead of the sum of two normal distributions.

\begin{align}
S_{hg} &\sim N(\mu_h+\mu_g, \tau_h+\tau_g)
\end{align}

The parameters $\mu_h$, $\mu_g$, $\tau_h$ and $\tau_g$ are unknown in the model so a new layer is added to the hierarchical model where the means $\mu$ and precisions $\tau$ are drawn from a two normal-gamma distribution:

\begin{align}
\mu_h, \sigma_h &\sim N(\phi_h, \sigma_h*\lambda_h)\Gamma(\alpha_h, \beta_h) \\
\mu_g, \sigma_g &\sim N(\phi_g, \sigma_g*\lambda_g)\Gamma(\alpha_g, \beta_g)
\end{align}

Where the parameters $\phi_h$, $\sigma_h$, $\lambda_h$, $\alpha_h$ $\beta_h$, $\phi_g$, $\sigma_g$, $\lambda_g$, $\alpha_g$ and $\beta_g$ are hyper-parameters in the model. It does not make sense to add more complexity to the mode so the values of these hyper-parameters are set to values the are reasonable values according to the observed data 

## Markov Chain Monte Carlo

An Markov Chain Monte Carlo (MCMC) algorithm can be used to find the posterior distribution. In this case the latent score of a handing and bias of a grader. There exist multiple algorithms for solving such a problem such. The thesis will only look into random walk Monte Carlo methods (why?). 

## Gibbs Sampling

For the case of hierarchical model with multiple dimension of parameters the most efficient random walk method are are Gibbs sampling as it does not require any 'tuning' and sampling each variable in turn compared to other methods. Additionally it can Incorporate other methods in the sampling process. The main problem with the Gibbs sampling methods is that it requires the conditional probability of the target distributions. These probability can be hard, even impossible, to calculate.

Finding the conditional probability can be done in different ways but the easiest part is to find the full conditional probability of all the parameters. From here, each conditional probability can be found.

The full conditional probability:

\begin{align}
\begin{split}
p(\mu_{\mathcal{G}(h)},\tau_{\mathcal{G}(h)},\mu_h,\tau_h|S_{h},...) \propto& \prod_{g \in \mathcal{G}(h)}[N(S_{h}|\mu_g+\mu_h,\tau_g+\tau_h) \\
& \times N(\mu_g,\tau_g|\phi_g, \sigma_g*\lambda_g) \Gamma(\sigma_g|\alpha_g, \beta_g)] \\
& \times N(\mu_h,\tau_h|\phi_h, \sigma_h*\lambda_h) \Gamma(\sigma_h|\alpha_h, \beta_h)
\end{split}
\end{align}

where $\mathcal{G}(h)$ are the set of graders who have graded the handin $h$

We can then find the conditional probability for the different parameters.

For $\mu_h$ and $\tau_h$ we have

\begin{align} \label{eq:conh}
\begin{split}
p(\mu_h,\tau_h|S_{h},...) \propto& \prod_{g \in \mathcal{G}(h)}[N(S_{h}|\mu_g+\mu_h,\tau_g+\tau_h)] \\
& \times N(\mu_h,\tau_h|\phi_h, \sigma_h*\lambda_h) \Gamma(\sigma_h|\alpha_h, \beta_h)
\end{split}
\end{align}

For a given grader $g$ with $\mu_g$ and $\tau_g$ who have graded a handin $h$ we have the conditional probability

\begin{align} \label{eq:cong}
\begin{split}
p(\mu_g,\tau_g|S_{h},...) \propto& N(S_{h}|\mu_g+\mu_h,\tau_g+\tau_h) \\
& \times N(\mu_g,\tau_g|\phi_g, \sigma_g*\lambda_g) \Gamma(\sigma_g|\alpha_g, \beta_g)
\end{split}
\end{align}

These conditional probability can be reduced by using the probability density function (PDF of the distributions that the probability consist of. The aim is to reduce it down to a form that defines a single distribution.

The normal distribution have the PDF:

\begin{align}
f(x|\mu,\tau) =& \frac{\sqrt{\tau}}{\sqrt{2\pi}}e^{-\tau\frac{(x-\mu)^2}{2}}
\end{align}

and the normal-gamma distributions PDF is:

\begin{align}
f(\mu,\tau|\phi_g, \lambda_g, \alpha_g, \beta_g) =& \frac{\beta^\alpha\sqrt{\lambda}}{\Gamma(\alpha)\sqrt{2\pi}}\tau^{\alpha-\frac{1}{2}}e^{-\beta\tau}e^{\frac{-\lambda\tau(\mu-\phi)^2}{2}}
\end{align}

lets start with \ref{eq:cong} as it is the easiest

\begin{align}
\begin{split}
p(\mu_g,\tau_g|S_{h},...) \propto& \frac{\sqrt{\tau_g+\tau_h}}{\sqrt{2\pi}}e^{-(\tau_g+\tau_h)\frac{(x-(\mu_g+\mu_h))^2}{2}} \\
& \times \frac{\beta_g^{\alpha_g}\sqrt{\lambda_g}}{\Gamma(\alpha_g)\sqrt{2\pi}}\tau_g^{\alpha_g-\frac{1}{2}}e^{-\beta_g\tau_g}e^{\frac{-\lambda_g\tau_g(\mu_g-\phi_g)^2}{2}}
\end{split}
\end{align}

We only want the probability in proportion to distributions. Constants can then be removed from the equation.

\begin{align}
\begin{split}
p(\mu_g,\tau_g|S_{h},...) \propto& \sqrt{\tau_g+\tau_h}e^{-(\tau_g+\tau_h)\frac{(x-(\mu_g+\mu_h))^2}{2}} \\
& \times \tau_g^{\alpha_g-\frac{1}{2}}e^{-\beta_g\tau_g}e^{\frac{-\lambda_g\tau_g(\mu_g-\phi_g)^2}{2}}
\end{split}
\end{align}

...

Next we can do the same with \ref{eq:conh}

\begin{align}
\begin{split}
p(\mu_h,\tau_h|S_{h},...) \propto& \prod_{g \in \mathcal{G}(h)}[\sqrt{\tau_g+\tau_h}e^{-(\tau_g+\tau_h)\frac{(x-(\mu_g+\mu_h))^2}{2}}] \\
& \times \tau_h^{\alpha_h-\frac{1}{2}}e^{-\beta_h\tau_h}e^{\frac{-\lambda_h\tau_h(\mu_h-\phi_h)^2}{2}}
\end{split}
\end{align}

The end result should be two new Normal-Gamma distribution. If these are obtained, the Gibbs Sampling method can be used to drasticly improve the current performance of MCMC. If not, then the Metrpoloish hastings method can be used instead but will add some drawbacks performance wise.

Example of a Metropolish Hasting combined with Gibbs can be seen [here](http://sites.stat.psu.edu/~mharan/MCMCtut/fullcond.pdf)