fit <- function(X, y, output_dir, class_order=NULL, row_weights=NULL, ...){
  if (class(X) != "dgTMatrix") {
    stop("X matrix is not sparse when it should be")
  }
  if (is.null(colnames(X))) {
    stop("X colnames are null when they should be populated")
  }

  # Save model
  model_path <- file.path(output_dir, 'artifact.rds')
  saveRDS(c("dummy", "artifact"), file = model_path)
}

score <- function(X, model, ...) {
   # return a data.frame with a 'Predictions' column with a single value of 42
   data.frame('Predictions'=rep(42, nrow(X)))
}