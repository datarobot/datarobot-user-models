prediction_value <- NaN


init <- function(...) {
    prediction_value <<- 1
    library(stringi)
}

read_input_data <- function(input_binary_data) {
    prediction_value <<- prediction_value + 1
    input_text_data <- stri_conv(input_binary_data, "utf8")
    read.csv(text=gsub("\r","", input_text_data, fixed=TRUE))
}

load_model <- function(input_dir) {
    prediction_value <<- prediction_value + 1
    "dummy"
}

transform <- function(data, model) {
    prediction_value <<- prediction_value + 1
    data
}

score <- function(data, model, ...) {
    prediction_value <<- prediction_value + 1
    predictions = data.frame(matrix(prediction_value, ncol = 1, nrow = nrow(data)))
    names(predictions) <- c("Predictions")
    predictions
}

post_process <- function(predictions, model) {
    predictions + 1
}
