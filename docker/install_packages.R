#set mirror from download R packages
#options(repos = c(CRAN = "https://mirrors.ustc.edu.cn/CRAN/"))
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
#options(BioC_mirror="https://mirrors.tuna.tsinghua.edu.cn/bioconductor")

install.packages(c("testthat","knitr","rematch2","waldo","testthat","roxygen2"),dependencies=TRUE)
install.packages('Marix',version = '1.6.0',dependencies=TRUE)
pkg <- c('foreach','plyr','Rcpp','usethis', 'miniUI', 'pkgdown','rcmdcheck', 'roxygen2', 'rversions', 'urlchecker','curl','gert','stringi','BiocManager',
'lars','ROCR','progress','readr','igraph','networkD3','ggsci','ggthemes','parmigene','corpcor','fdrtool','bc3net','c3net')
install.packages(pkg,dependencies=TRUE)
install.packages('devtools',dependencies=TRUE)
devtools::install_url('https://cran.r-project.org/src/contrib/Archive/foba/foba_0.1.tar.gz')
devtools::install_url('https://cran.r-project.org/src/contrib/Archive/htmlTable/htmlTable_2.0.0.tar.gz')
devtools::install_url('https://cran.r-project.org/src/contrib/Archive/Hmisc/Hmisc_4.5-0.tar.gz')
devtools::install_url('https://cran.r-project.org/src/contrib/Archive/PCIT/PCIT_1.5-3.tar.gz')
BiocManager::install('GO.db')
bio_pkg <- c('AnnotationDbi', 'GO.db', 'preprocessCore', 'impute', 'graph', 'GENIE3', 'minet','WGCNA','doRNG','GeneNet')
BiocManager::install(bio_pkg)
