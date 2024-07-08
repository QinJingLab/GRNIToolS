\name{ennet}
\alias{ennet}
\title{
Calculate Gene Regulatory Network from Expression data
}
\description{
This function calculates a P-by-P adjacency matrix V from S-by-P expression matrix E. E is expected to be given as input. E is assumed to have P columns corresponding to genes and S rows corresponding to different experiments. Additionally, ennet function takes matrix of initial perturbations of genes K of the same size as E, and other parameters. As a result, ennet returns a squared matrix V of edge confidences of size P-by-P. A subset of known transcription factors can be defined as a subset of all P genes.
}
\usage{
ennet(E = matrix(rnorm(10000), 100, 100), K = matrix(0, nrow(E), ncol(E)), Tf = 1:ncol(E), s_s = 1, s_f = 0.3, M = 5000, nu = 0.001, scale = TRUE, center = TRUE, optimization.stage = 2)
}
\arguments{
  \item{E}{
S-by-P expression matrix. Columns correspond to genes, rows correspond to experiments. E is expected to be already normalized using standard methods, for example RMA.
}
  \item{K}{
S-by-P initial perturbation matrix. It directly corresponds to E matrix, e.g. if K[i,j] is equal to 1, it means that gene j was knocked-out in experiment i. Single gene knock-out experiments are rows of K with only one value 1. By default it's a matrix of zeros of the same size as E, e.g. unknown initial perturbation state of genes.
}
  \item{Tf}{
Vector of known transcription factors indicating which columns of E are known transcription factors. If it is defined, the resulting adjacency matrix V will still be P-by-P matrix, but only rows corresponding to Tf will be non-zero. By default it's a vector 1:P, e.g. every gene is a known transcription factor.
}
  \item{s_s}{
Sampling rate of experiments, 0<s_s<=1. Fraction of rows of E, which will be sampled with replacement to calculate each extension in boosting model. By default it's 1.
}
  \item{s_f}{
Sampling rate of transcription factors, 0<s_f<=1. Fraction of transcription factors from E, as indicated by \code{Tf} vector, which will be sampled without replacement to calculate each extesion in boosting model. By default it's 0.3.
}
  \item{M}{
Number of extensions in boosting model, e.g. number of iterations of the main loop of ENNET algorithm. By default it's 5000.
}
  \item{nu}{
Shrinkage factor, learning rate, 0<nu<=1. Each extension to boosting model will be multiplied by the learning rate. By default it's 0.001.
}
  \item{scale}{
Logical flag indicating if each column of E should be scaled to be unit standard deviation. By default it's TRUE.
}
  \item{center}{
Logical flag indicating if each column of E should be scaled to be zero mean. By default it's TRUE.
}
  \item{optimization.stage}{
Numerical flag indicating if re-evaluation of edge confidences should be applied after calculating initial V, optimization.stage=\{0,1,2\}. If optimization.stage=0, no re-evaluation will be applied. If optimization.stage=1, variance-based optimization will be applied. If optimization.stage=2, variance-based and z-score based optimizations will be applied.
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
  \item{V}{
Gene Regulatory Network in form of a P-by-P adjacency matrix.
}
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
Janusz Slawek <janusz.slawek@gmail.com>
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
\code{\link{ennet.train}}, \code{\link{ennet.test}}, \code{\link{v2l}}
}
\examples{
# load ennet library
library("ennet")
# this step is optional, it helps speed up calculations, by running the algorithm in parallel on 8 cores
library("doMC")
registerDoMC(cores=8)
# run network inference on a 100-by-100 dummy expression data.
V = ennet()
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
