\name{ennet.test}
\alias{ennet.test}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Test ENNET predictor
}
\description{
This function tests a regression model for a given \code{X.test} feature matrix, \code{Y.test} response vector, and working parameters.
}
\usage{
ennet.test(model, X.test, Y.test, M.test)
}
\arguments{
  \item{model}{
Model returned by \code{\link{ennet.train}} function.
}
  \item{X.test}{
Input S-by-P feature matrix of unseen samples. Columns correspond to features, rows correspond to samples.
}
  \item{Y.test}{
Input S-element response vector of unseen samples.
}
  \item{M.test}{
Number of extensions of boosting model to take when predicting response. Must be not greater than \code{M.train} used when training boosting model.
}
}
\details{
}
\value{
Result of regression
}
\references{
}
\author{
Janusz Slawek <janusz.slawek@gmail.com>
}
\note{
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
