# Needed libraries
library(caret)
library(devtools)

init_hook <- FALSE
load_model_hook <- FALSE
transform_hook <- FALSE
score_hook <- FALSE
post_process_hook <- FALSE

REGRESSION_PRED_COLUMN_NAME <- "Predictions"
CUSTOM_MODEL_FILE_EXTENSION <- ".rds"

#' Import R source files as a named package
#'
#' @param srcFiles character, file paths to load as the package
#' @param pkgName character or NULL, name to give the package
#'
#' @return bool, TRUE if the package was succussfully loaded, FALSE otherwise
#' @export
#'
#' @examples import("~/Documents/R/custom.R", "myPackage")
import <- function(srcFiles, pkgName = "custom") {
    dd <- tempdir()
    on.exit(unlink(file.path(dd, pkgName), recursive=TRUE))
    tryCatch(
        {
            package.skeleton(name=pkgName, path = dd, code_files=srcFiles)
            load_all(file.path(dd, pkgName))
            return(TRUE)
        },
        error = function(cond) {
            message(c(cond, "\n"))
            return(FALSE)
        }
    )
}

#' Get a method from a package
#'
#' @param name character, the name of the method to retrieve
#' @param pkgName character or NULL, the package to look in for the method
#'
#' @return function if the method is found or FALSE
#' @export
#'
#' @examples getHookMethod("foo", "myPackage")
getHookMethod <- function(name, pkgName = "custom") {
    tryCatch(
        {
            hook = getExportedValue(pkgName, name)
            if (is.function(hook)) {
                return(hook)
            } else {
                message(name, " is not a method")
                return(FALSE)
            }
        },
        error = function(cond) {
            message(c(cond, "\n"))
            return(FALSE)
        }
    )
}

init <- function(code_dir) {
    custom_path <- file.path(code_dir, "custom.R")
    custom_loaded <- import(custom_path)
    if (isTRUE(custom_loaded)) {
        init_hook <<- getHookMethod("init")
        load_model_hook <<- getHookMethod("load_model")
        transform_hook <<- getHookMethod("transform")
        score_hook <<- getHookMethod("score")
        post_process_hook <<- getHookMethod("post_process")
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
            stop("Could not find model artifact, with ", CUSTOM_MODEL_FILE_EXTENSION,
                 " extension, supported by default R predictor. ",
                 "If your artifact is not supported by default predictor, implement custom.load_model hook."
                )
        } else if (length(file_names) > 0) {
            stop("Multiple serialized model files found. Remove extra artifacts or overwrite custom.load_model")
        }
        model_artifact <- file.path(model_dir, file_names[1])
        if (is.na(model_artifact)) {
            stop(sprintf("Could not find serialized model file. Serialized model file name should have the extension %s",
                         CUSTOM_MODEL_FILE_EXTENSION
            ))
        }
        model <- readRDS(model_artifact)
    }
    model
}

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
model_predict <- function(data, model, positive_class_label=NULL, negative_class_label=NULL) {
    if (!is.null(positive_class_label) & !is.null(negative_class_label)) {
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
    } else {
        predictions <- data.frame(stats::predict(model, data))
        names(predictions) <- c(REGRESSION_PRED_COLUMN_NAME)
    }
    predictions
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
outer_predict <- function(data, model=NULL, positive_class_label=NULL, negative_class_label=NULL){
    .validate_data <- function(to_validate, hook) {
        if (!is.data.frame(to_validate)) {
            stop(sprintf("%s must return a data.frame", hook))
        }
    }
    
    .validate_predictions <- function(to_validate, hook) {
        .validate_data(to_validate, hook)
        if (!is.null(positive_class_label) & !is.null(negative_class_label)) {
            if (!identical(sort(names(to_validate)), sort(c(positive_class_label, negative_class_label)))) {
                stop(
                    sprintf(
                        "Expected %s predictions to have columns [%s], but encountered [%s]",
                        hook,
                        paste(c(positive_class_label, negative_class_label), collapse=", "),
                        paste(names(to_validate), collapse=", ")
                    )
                )
            }
        } else if (!identical(names(to_validate), c(REGRESSION_PRED_COLUMN_NAME))) {
            stop(
                sprintf(
                    "Expected %s predictions to have columns [%s], but encountered [%s]",
                    hook,
                    paste(c(REGRESSION_PRED_COLUMN_NAME), collapse=", "),
                    paste(names(to_validate), collapse=", ")
                )
            )
        }
    }
    
    if (is.null(model)) {
        model <- load_serialized_model()
    }
    
    if (!isFALSE(transform_hook)) {
        data <- transform_hook(data, model)
        .validate_data(data, "transform")
    }

    if (!isFALSE(score_hook)) {
        kwargs <- list()
        if (!is.null(positive_class_label) & !is.null(negative_class_label)) {
            kwargs <- append(kwargs, list(positive_class_label=positive_class_label,
                                          negative_class_label=negative_class_label))
        }
        predictions <- do.call(score_hook, list(data, model, kwargs))
        .validate_predictions(predictions, "score")
    } else {
        predictions <- model_predict(data, model, positive_class_label, negative_class_label)
    }
    
    if (!isFALSE(post_process_hook)) {
        predictions <- post_process_hook(predictions, model)
        .validate_predictions(predictions, "post_process")
    }
    
    predictions
}
