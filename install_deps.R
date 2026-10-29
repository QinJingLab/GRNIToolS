## Install the R dependencies that conda does not carry, then GRNIToolS itself.
## Run from an activated environment:  Rscript install_deps.R

options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 4, timeout = 3600)

## Paths come from the script location, so the repository can be moved freely.
script_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[startsWith(a, "--file=")])
  if (length(f) == 1) dirname(normalizePath(f)) else normalizePath(".")
}
HERE <- script_dir()
PKG  <- Sys.getenv("GRN_PKG",  file.path(HERE, "GRNIToolS", "GRNIToolS"))
WORK <- Sys.getenv("GRN_WORK", file.path(tempdir(), "grn_src"))
dir.create(WORK, showWarnings = FALSE, recursive = TRUE)

## Compiling needs $CONDA_PREFIX/bin on PATH for the conda C/C++ wrappers.
PREFIX <- Sys.getenv("CONDA_PREFIX", "")
if (nzchar(PREFIX)) {
  Sys.setenv(PATH = paste(file.path(PREFIX, "bin"), Sys.getenv("PATH"), sep = .Platform$path.sep))
} else {
  cat("WARNING: CONDA_PREFIX is empty; activate the GRN environment first.\n")
}

cat("R      :", R.version.string, "\n")
cat("prefix :", PREFIX, "\n")
cat("libpath:", .libPaths()[1], "\n")
cat("CC     :", Sys.which("x86_64-conda-linux-gnu-cc"), "\n\n")

## Unpack with the system tar: GitHub tarballs carry a pax header that R's own untar skips.
install_tarball <- function(path) {
  exdir <- file.path(WORK, paste0("extract_", tools::file_path_sans_ext(basename(path))))
  unlink(exdir, recursive = TRUE)
  dir.create(exdir, recursive = TRUE)
  st <- system2("tar", c("xzf", shQuote(path), "-C", shQuote(exdir)))
  if (st != 0) stop("tar failed on ", path)
  desc <- list.files(exdir, pattern = "^DESCRIPTION$", recursive = TRUE, full.names = TRUE)
  if (length(desc) == 0) stop("no DESCRIPTION found in ", path)
  pkgdir <- dirname(desc[1])
  cat("  pkgdir:", pkgdir, "\n")
  install.packages(pkgdir, repos = NULL, type = "source")
}

## doRNG and graph are only Suggests of GENIE3 and GeneNet, but the default paths need them.
NEED <- c("foba", "WGCNA", "parmigene", "PCIT", "GeneNet", "c3net", "bc3net", "ennet",
          "doRNG", "graph")
have <- function(p) requireNamespace(p, quietly = TRUE)
miss <- function() NEED[!vapply(NEED, have, TRUE)]

dl <- function(url, dest) {
  if (file.exists(dest) && file.size(dest) > 0) { cat("  cached:", dest, "\n"); return(dest) }
  cat("  GET:", url, "\n")
  utils::download.file(url, dest, mode = "wb", quiet = TRUE)
  stopifnot(file.size(dest) > 0)
  dest
}

## ---- 1) Bioconductor ------------------------------------------------------
bioc <- intersect(miss(), c("parmigene", "GeneNet", "graph"))
if (length(bioc)) {
  cat("== Bioconductor:", paste(bioc, collapse = ", "), "==\n")
  BiocManager::install(bioc, ask = FALSE, update = FALSE)
}

## ---- 2) CRAN --------------------------------------------------------------
cran <- intersect(miss(), c("WGCNA", "c3net", "bc3net", "doRNG"))
if (length(cran)) {
  cat("== CRAN:", paste(cran, collapse = ", "), "==\n")
  install.packages(cran)
}
## plyr is not imported by GRNIToolS, but ennet Depends on it and source installs skip deps.
if (!have("plyr")) {
  cat("== CRAN: plyr (ennet dependency) ==\n")
  install.packages("plyr")
}

## ---- 3) Archived on CRAN --------------------------------------------------
## foba: dropped from CRAN after 2013, last archive version is 0.1.
if ("foba" %in% miss()) {
  cat("== foba (CRAN Archive) ==\n")
  f <- dl("https://cran.r-project.org/src/contrib/Archive/foba/foba_0.1.tar.gz",
          file.path(WORK, "foba_0.1.tar.gz"))
  install.packages(f, repos = NULL, type = "source")
}

## PCIT: archived in 2020 and removed from Bioconductor; GRNIToolS uses pcit() and idx() only.
if ("PCIT" %in% miss()) {
  cat("== PCIT (CRAN Archive) ==\n")
  f <- dl("https://cran.r-project.org/src/contrib/Archive/PCIT/PCIT_1.5-3.tar.gz",
          file.path(WORK, "PCIT_1.5-3.tar.gz"))
  install.packages(f, repos = NULL, type = "source")
}

## ---- 4) GitHub only -------------------------------------------------------
## ennet is not on CRAN/Bioconductor/conda, and CRAN has an unrelated package of that name.
if ("ennet" %in% miss()) {
  cat("== ennet (GitHub slawekj/ennet @ master) ==\n")
  f <- dl("https://codeload.github.com/slawekj/ennet/tar.gz/refs/heads/master",
          file.path(WORK, "ennet-master.tar.gz"))
  install_tarball(f)
}

## ---- 5) GRNIToolS itself --------------------------------------------------
ok_pkg <- FALSE
if (!file.exists(file.path(PKG, "DESCRIPTION"))) {
  cat("== GRNIToolS: source directory not found at ", PKG, " ==\n", sep = "")
} else {
  cat("== GRNIToolS (local source) ==\n")
  install.packages(PKG, repos = NULL, type = "source")
  ok_pkg <- TRUE
}

## ---- Result ---------------------------------------------------------------
cat("\n================ RESULT ================\n")
ok <- TRUE
for (p in c(NEED, "plyr", "GRNIToolS")) {
  if (have(p)) {
    cat(sprintf("  OK      %-12s %s\n", p, as.character(packageVersion(p))))
  } else {
    ok <- FALSE
    cat(sprintf("  MISSING %-12s\n", p))
  }
}

## The CRAN package called ennet does not export ennet(), so check for it.
if (have("ennet")) {
  if ("ennet" %in% getNamespaceExports("ennet")) {
    cat("  OK      ennet exports ennet()\n")
  } else {
    ok <- FALSE
    cat("  WRONG   ennet installed but does NOT export ennet() (the CRAN package)\n")
  }
}
if (!ok_pkg) ok <- FALSE

## Resolving every Import says more about real usability than a per-package list.
if (have("GRNIToolS")) {
  imp <- unlist(strsplit(gsub("\\s*\\(.*?\\)", "",
           paste(read.dcf(file.path(PKG, "DESCRIPTION"), "Imports"), collapse = ",")), ","))
  imp <- trimws(imp[nzchar(imp)])
  bad <- imp[!vapply(imp, have, TRUE)]
  if (length(imp) && !length(bad)) {
    cat(sprintf("  OK      all %d GRNIToolS Imports resolve\n", length(imp)))
  } else {
    ok <- FALSE
    cat(sprintf("  MISSING GRNIToolS Imports: %s\n", paste(bad, collapse = ", ")))
  }
}

cat(if (ok) "\nALL DEPENDENCIES OK\n" else "\nSOME DEPENDENCIES MISSING\n")
