\name{ennet.train}
\alias{ennet.train}
\title{
Train ENNET predictor
}
\description{
This function trains a regression model for a given \code{X.train} feature matrix, \code{Y.train} response vector, and working parameters. A model returned by this function can be used to predict response for unseen data with \code{\link{ennet.test}} function.
}
\usage{
ennet.train(X.train, Y.train, s_f = 0.3, s_s = 1, M.train = 5000, nu = 0.001)
}
\arguments{
  \item{X.train}{
Input S-by-P feature matrix of training samples. Columns correspond to features, rows correspond to samples.
}
  \item{Y.train}{
Input S-element response vector of training samples.
}
  \item{s_f}{
Sampling rate of features, 0<s_f<=1. Fraction of columns from X.train, which will be sampled without replacement to calculate each extesion in boosting model. By default it's 0.3.
}
  \item{s_s}{
Sampling rate of samples, 0<s_s<=1. Fraction of rows from X.train, which will be sampled with replacement to calculate each extension in boosting model. By default it's 1.
}
  \item{M.train}{
Number of extensions in boosting model, e.g. number of iterations of the main loop of ENNET algorithm. By default it's 5000.
}
  \item{nu}{
Shrinkage factor, learning rate, 0<nu<=1. Each extension to boosting model will be multiplied by the learning rate. By default it's 0.001.
}
}
\details{
}
\value{
Regression model is a structure containing all the information needed to predict response for unseen data
}
\references{
}
\author{
Janusz Slawek <janusz.slawek@gmail.com>
}
\note{
}

\seealso{
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
