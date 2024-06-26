#!/usr/bin/env Rscript

# Required for reticulate to work properly
Sys.setenv(MKL_THREADING_LAYER = "GNU")

install.packages('optparse', repos='http://cran.us.r-project.org')
install.packages('reticulate', repos='http://cran.us.r-project.org')
