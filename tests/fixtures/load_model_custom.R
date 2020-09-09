init <- function(...) {
    library(brnn)
    library(glmnet)
}

load_model <- function(input_dir) {
    file.path(input_dir, "r_bin.rds")
    readRDS(file.path(input_dir, "r_bin.rds"))
}

transform <- function(data, model) {
    data[is.na(data)] <- 0
    data
}