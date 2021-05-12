# this score rturns predictions which fails a check that probabilities must add up to one
score <- function(data, model, ...) {
    predictions = data.frame(rbind(c(0.2, 0.7), 1)[rep(1, nrow(data)), ])
    names(predictions) <- c("yes", "no")
    predictions
}

load_model <- function(input_dir) {
    "dummy"
}