#1. Preparing data
library(GRNIToolS)
ntf <- 30
ngene <- 100
nsample <- 10
set.seed(123)
A <- matrix(rnorm(ntf * nsample,0,1),ntf,nsample)
TF <- paste0('G',1:ntf)
B <- matrix(rnorm(ngene * nsample,0,1),ngene,nsample)
X <- matrix(0,ngene,ntf)
s <- c(1:10)
gene <- paste0('G',1:ngene)
HGSmat <- matrix(0,ngene,ntf)
HGSmat[sample(ngene * ntf,0.2 * ngene * ntf)] <- 1
colnames(HGSmat) <- TF
rownames(HGSmat) <- gene


#2. Network inference
res <- demo_reg(A, B, s, X, 'ADMM',max.steps = 200,TF, gene, cl.cores = 2,file=NULL,verbose=TRUE)
grn <- res$grn 
grn[1:5,1:5]
res <- demo_other(B,'Pearson',TF=TF,gene=gene,file=NULL,verbose=TRUE)grn <- res$grn 
grn[1:5,1:5]

#3. Evaluation
dat <- calpl(HGSmat = HGSmat,grn = grn,sym = F)
confu <- confusion(dat = dat,threshold = 0.5)
print(confu)
auc <- calauc(dat = dat)
print(auc)

#4. ROC and PR curve
p1 <- draw_plot(dat,method = 'ROC',algname = 'Pearson')
p2 <- draw_plot(dat,method = 'PR',algname = 'Pearson')
print(p1)

#5. Network diagram
net <- network(grn,TF = TF,gene = gene)
#print(net)
grn <- out_grn(grn,TF = TF,gene = gene)
head(grn)

# Pipeline
algorithm <- c('Pearson','parmigene_ARACNE.a','minet_MRNET','PCIT','ADMM')
conlist <- grn_main(A=A,B=B,X=X,seq = s,gene=gene,TF=TF,algorithm = algorithm,
                    HGSmat = HGSmat,file=TRUE, verbose=FALSE,bootstrap_num = 5,sym = FALSE,threshold = 0.5)
print(conlist[1:10,])
dat <- data.frame(alg = conlist$Algorithm,auc = conlist$AUROC)
p <- draw_box(dat,'AUROC')
print(p)