\name{v2l}
\alias{v2l}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Convert adjacency matrix to a list of edges
}
\description{
This function converts adjacency matrix V to a sorted list of edges, e.g. a list in which edges are sorted by decreasing confidence.
}
\usage{
v2l(V, max = 1e+05)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{V}{
Input adjacency matrix.
}
  \item{max}{
Maximal length of the resulting list. This number may be lower than the number of all the edges from adjacency matrix. Then only top \code{max} edges will be returned.
}
}
\details{
}
\value{
A data frame of sorted edges: (1) list of sources (2) list of destinations (3) list of confidences. Elements in all the lists correspond to each other.
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
