score <- function(data, model, ...) {
    predictions = data.frame(rbind(c(0.3, 0.7), 1)[rep(1, nrow(data)), ])
    names(predictions) <- as.integer(c(0, 1))
    predictions
}

load_model <- function(input_dir) {
    "dummy"
}