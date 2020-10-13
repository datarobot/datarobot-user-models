# Needed libraries
library(caret)
library(devtools)

init_hook <- FALSE
read_input_data_hook <- FALSE
load_model_hook <- FALSE
transform_hook <- FALSE
score_hook <- FALSE
score_unstructured_hook <- FALSE
post_process_hook <- FALSE

REGRESSION_PRED_COLUMN_NAME <- "Predictions"
CUSTOM_MODEL_FILE_EXTENSION <- ".rds"
RUNNING_LANG_MSG <- "Running environment language: R."

init <- function(code_dir, target_type) {
    custom_path <- file.path(code_dir, "custom.R")
    custom_loaded <- import(custom_path)

    if (isTRUE(custom_loaded)) {
        if (target_type == TargetType$UNSTRUCTURED) {
            init_hook <<- getHookMethod("init")
            load_model_hook <<- getHookMethod("load_model")
            score_unstructured_hook <<- getHookMethod("score_unstructured")
        } else {
            init_hook <<- getHookMethod("init")
            read_input_data_hook <<- getHookMethod("read_input_data")
            load_model_hook <<- getHookMethod("load_model")
            transform_hook <<- getHookMethod("transform")
            score_hook <<- getHookMethod("score")
            post_process_hook <<- getHookMethod("post_process")
        }
    }

    if (!isFALSE(init_hook)) {
        init_hook(code_dir=code_dir)
    }
}

#' Load a serialized model.  The model should have the extension .rds
#'
#' @return the deserialized model
#' @export
#'
#' @examples
load_serialized_model <- function(model_dir) {
    model <- NULL
    if (!isFALSE(load_model_hook)) {
        model <- load_model_hook(model_dir)
    }
    if (is.null(model)) {
        file_names <- dir(model_dir, pattern = CUSTOM_MODEL_FILE_EXTENSION)
        if (length(file_names) == 0) {
            stop("\n\n", RUNNING_LANG_MSG, "\nCould not find a serialized model artifact with ",
                 CUSTOM_MODEL_FILE_EXTENSION,
                 " extension, supported by default R predictor. ",
                 "If your artifact is not supported by default predictor, implement custom.load_model hook."
                )
        } else if (length(file_names) > 1) {
            stop("\n\n", RUNNING_LANG_MSG, "\n",
            "Multiple serialized model artifacts found: [", paste(file_names, collapse = ' '),
            "] in ", model_dir,
            ". Remove extra artifacts or overwrite custom.load_model")
        }
        model_artifact <- file.path(model_dir, file_names[1])
        if (is.na(model_artifact)) {
            stop(sprintf("\n\n", RUNNING_LANG_MSG, "\n",
                         "Could not find serialized model artifact. Serialized model file name should have the extension %s",
                         CUSTOM_MODEL_FILE_EXTENSION
            ))
        }

        tryCatch(
            {
                model <- readRDS(model_artifact)
            },
            error = function(err) {
                stop("\n\n", RUNNING_LANG_MSG, "\n",
                  "Could not load searialized model artifact: ", model_artifact
                )
            }
        )
    }
    model
}

.predict_regression <- function(data, model, ...) {
    predictions <- data.frame(stats::predict(model, data))
    names(predictions) <- c(REGRESSION_PRED_COLUMN_NAME)
    predictions
}

.predict_binary <- function(data, model, ...) {
    kwargs <- list(...)
    positive_class_label<-kwargs$positive_class_label
    negative_class_label<-kwargs$negative_class_label
    predictions <- data.frame(stats::predict(model, data, type = "prob"))
    labels <- names(predictions)
    provided_labels <- c(positive_class_label, negative_class_label)
    provided_labels_sanitized <- make.names(provided_labels)
    labels_to_use <- NULL
    # check labels and provided_labels contain the same elements, order doesn't matter
    if (setequal(labels, provided_labels)) {
        labels_to_use <- provided_labels
    } else if (setequal(labels, provided_labels_sanitized)) {
        labels_to_use <- provided_labels_sanitized
    } else {
        stop("Wrong class labels. Use class labels according to your dataset")
    }
    # if labels are not on the same order, switch columns
    if (!identical(labels, labels_to_use)) {
        predictions <- predictions[, c(2, 1)]
    }
    names(predictions) <- provided_labels
    predictions
}

.predictors<-list()
.predictors[[TargetType$BINARY]] <- .predict_binary
.predictors[[TargetType$REGRESSION]] <- .predict_regression
.predictors[[TargetType$ANOMALY]] <- .predict_regression

#' Internal prediction method that makes predictions against the model, and returns a data.frame
#'
#' If the model is a regression model, the data.frame will have a single column "Predictions"
#' If the model is a classification model, the data.frame will have a column for each class label
#'     with their respective probabilities
#'
#' @param data data.frame to make predictions against
#' @param model to use to make predictions
#' @param positive_class_label character or NULL, The positive class label if this is a binary classification prediction request
#' @param negative_class_label character or NULL, The negative class label if this is a binary classification prediction request
#'
#' @return data.frame of predictions
#' @export
#'
#' @examples
model_predict <- function(...) {
    kwargs <- list(...)
    target_type<-kwargs$target_type
    kwargs$target_type <- NULL

    if (!exists(target_type, .predictors)) {
        stop(sprintf("Target type '%s' is not supported by R predictor", target_type))
    }
    do.call(.predictors[[target_type]], kwargs)
}

#' Makes predictions against the model using the custom predict
#' method and returns a data.frame
#'
#' If the model is a regression model, the data.frame will have a single column "Predictions"
#' If the model is a classification model, the data.frame will have a column for each class label
#'     with their respective probabilities
#'
#' @param data data.frame to make predictions against
#' @param model to use to make predictions
#' @param positive_class_label character or NULL, The positive class label if this is a binary classification prediction request
#' @param negative_class_label character or NULL, The negative class label if this is a binary classification prediction request
#'
#' @return data.frame of predictions
#' @export
#'
#' @examples
outer_predict <- function(input_filename, target_type, model=NULL, positive_class_label=NULL, negative_class_label=NULL){
    .validate_data <- function(to_validate) {
        if (!is.data.frame(to_validate)) {
            stop(sprintf("predictions must be of a data.frame type, received %s", typeof(to_validate)))
        }
    }

    .validate_binary_predictions <- function(to_validate) {
        .validate_data(to_validate)
        if (!identical(sort(names(to_validate)), sort(c(positive_class_label, negative_class_label)))) {
            stop(
              sprintf(
                "Expected predictions to have columns [%s], but encountered [%s]",
                paste(c(positive_class_label, negative_class_label), collapse=", "),
                paste(names(to_validate), collapse=", ")
              )
            )
        }
    }

    .validate_regression_predictions <- function(to_validate) {
        .validate_data(to_validate)
         if (!identical(names(to_validate), c(REGRESSION_PRED_COLUMN_NAME))) {
            stop(
                sprintf(
                    "Expected predictions to have columns [%s], but encountered [%s]",
                    paste(c(REGRESSION_PRED_COLUMN_NAME), collapse=", "),
                    paste(names(to_validate), collapse=", ")
                )
            )
        }
    }

    if (!isFALSE(read_input_data_hook)) {
        data <- read_input_data_hook(input_filename)
    } else {
        tmp = readChar(input_filename, file.info(input_filename)$size)
        data <- read.csv(text=gsub("\r","", tmp, fixed=TRUE))
    }
    if (is.null(model)) {
        model <- load_serialized_model()
    }

    if (!isFALSE(transform_hook)) {
        data <- transform_hook(data, model)
    }

    kwargs <- list(positive_class_label=positive_class_label,
                   negative_class_label=negative_class_label)
    if (!isFALSE(score_hook)) {
        predictions <- do.call(score_hook, list(data, model, kwargs))
    } else {
        kwargs <- append(kwargs, list(data, model, target_type=target_type), after=0)
        predictions <- do.call(model_predict, kwargs)
    }

    if (!isFALSE(post_process_hook)) {
        predictions <- post_process_hook(predictions, model)
    }

    if (target_type == TargetType$BINARY) {
        .validate_binary_predictions(predictions)
    } else if (target_type == TargetType$REGRESSION) {
        .validate_regression_predictions(predictions)
    } else {
        stop(sprintf("Unsupported target type %s", target_type))
    }
    predictions
}

predict_unstructured <- function(model=NULL, data, ...) {
    .validate_unstructured_predictions <- function(to_validate) {
        single_input_value <- FALSE
        if (!is.list(to_validate)) {
            single_input_value <- TRUE
            to_validate <- list(to_validate, NULL)
        }

        if (length(to_validate) != 2) {
            stop(sprintf("In unstructured mode predictions of type list must have length = 2, but received length = %s", length(to_validate)))
        }

        data <- to_validate[[1]]
        kwargs <- to_validate[[2]]

        if (!any(is.character(data), is.raw(data), is.null(data)) || !any(is.list(kwargs), is.null(kwargs))) {
            if (single_input_value) {
                error_msg <- sprintf("In unstructured mode single return value can be of type character/raw, but received %s", typeof(data))
            } else {
                error_msg <- sprintf("In unstructured mode list return value must be of type (character/raw, list) but received (%s, %s)", typeof(data), typeof(kwargs))
            }
            stop(error_msg)
        }
        to_validate
    }
    kwargs_list = list(...)
    kwargs_list <- append(kwargs_list, list(model, data), after=0)
    predictions <- do.call(score_unstructured_hook, kwargs_list)
    validated_pred_list = .validate_unstructured_predictions(predictions)
    validated_pred_list
}