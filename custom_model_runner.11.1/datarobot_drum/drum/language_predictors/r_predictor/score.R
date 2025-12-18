# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
#
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
# Needed libraries
library(caret)
library(recipes)
library(devtools)
library(stringi)
library(Matrix)

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
    custom_loaded_R <- import(custom_path)

    if (isFALSE(custom_loaded_R)) {
        custom_path <- file.path(code_dir, "custom.r")
        custom_loaded_R <- import(custom_path)
    }

    if (isTRUE(custom_loaded_R)) {
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

has_read_input_data_hook <- function() {
    !isFALSE(read_input_data_hook)
}

#' Load a serialized model.  The model should have the extension .rds
#'
#' @return the deserialized model
#' @export
#'
#' @examples
load_serialized_model <- function(model_dir, target_type) {
    model <- NULL
    if (!isFALSE(load_model_hook)) {
        model <- load_model_hook(model_dir)
    }
    if (is.null(model)) {
        file_names <- dir(model_dir, pattern = CUSTOM_MODEL_FILE_EXTENSION, ignore.case = TRUE)
        if (length(file_names) == 0) {
            # Allow no serialized models if it is a transform
            if (target_type == TargetType$TRANSFORM) {
                return(NULL)
            }
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
            # Allow no serialized models if it is a transform
            if (target_type == TargetType$TRANSFORM) {
                return(NULL)
            }
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

.load_data <- function(binary_data, mimetype=NULL, use_hook=TRUE, sparse_colnames=NULL) {
    if (use_hook && !isFALSE(read_input_data_hook)) {
        data <- read_input_data_hook(binary_data)
    } else if (!is.null(mimetype) && mimetype == "text/mtx") {
        tmp_file_name <- tempfile()
        f <- file(tmp_file_name, "w+b")
        writeBin(binary_data, f)
        flush(f)
        data <- readMM(tmp_file_name)
        close(f)
        unlink(tmp_file_name)
        if(!is.null(sparse_colnames)) {
            colnames(data) <- sparse_colnames
        }
    } else {
        tmp <- stri_conv(binary_data, "utf8")
        text <- gsub("\r","", tmp, fixed=TRUE)
        data <- read.csv(text=text, check.names = FALSE)
        if (ncol(data) == 1) {
            data <- read.csv(text=text, check.names = FALSE, blank.lines.skip = FALSE)
        }
    }
    data
}

.predict_regression <- function(data, model, ...) {
    predictions <- data.frame(stats::predict(model, data))
    names(predictions) <- c(REGRESSION_PRED_COLUMN_NAME)
    predictions
}

.predict_classification <- function(data, model, ...) {
    data.frame(stats::predict(model, data, type = "prob"))
}

.predictors<-list()
.predictors[[TargetType$BINARY]] <- .predict_classification
.predictors[[TargetType$MULTICLASS]] <- .predict_classification
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
outer_predict <- function(target_type, binary_data=NULL, mimetype=NULL, model=NULL, positive_class_label=NULL, negative_class_label=NULL, class_labels=NULL, sparse_colnames=NULL){
    .validate_data <- function(to_validate) {
        if (!is.data.frame(to_validate)) {
            stop(sprintf("predictions must be of a data.frame type, received %s", typeof(to_validate)))
        }
    }
    data <- .load_data(binary_data, mimetype, sparse_colnames = sparse_colnames)
    if (!isFALSE(transform_hook)) {
        data <- transform_hook(data, model)
    }

    kwargs <- list(positive_class_label=positive_class_label,
                   negative_class_label=negative_class_label,
                   class_labels=class_labels)
    if (!isFALSE(score_hook)) {
        predictions <- do.call(score_hook, list(data, model, kwargs))
    } else {
        kwargs <- append(kwargs, list(data, model, target_type=target_type), after=0)
        predictions <- do.call(model_predict, kwargs)
    }

    if (!isFALSE(post_process_hook)) {
        predictions <- post_process_hook(predictions, model)
    }

    .validate_data(predictions)
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

#' Makes transforms against the transformer or by using the custom transform
#' method and returns a list containing the transformed X and optionally y
#'
#'
#' @param binary_data, Binary data containing X
#' @param target_binary_data, Optional binary data containing y
#' @param mimetype character, The file type of the binary data
#' @param transformer to use to make transformations
#'
#' @return list, Two-element list containing transformed X (data.frame or sparseMatrix) and y (vector or NULL)
#'
outer_transform <- function(binary_data=NULL, target_binary_data=NULL, mimetype=NULL, transformer=NULL, sparse_colnames=NULL){
    data <- .load_data(binary_data, mimetype=mimetype, sparse_colnames=sparse_colnames)
    target_data <- NULL
    if (!is.null(target_binary_data)) {
        target_data <- .load_data(target_binary_data, use_hook=FALSE)
    }

    if (!isFALSE(transform_hook)) {
        output_data <- transform_hook(data, transformer, target_data)
        if (is.data.frame(output_data)) {
            output_data <- list(output_data, NULL, NULL)
        } else if (is(output_data, 'sparseMatrix')) {
            output_data <- list(output_data, NULL, colnames(output_data))
        } else {
            output_data <- append(output_data, list(NULL))
        }
    } else {
        output_data <- list(bake(transformer, data), NULL, NULL)
    }

    if (!(is.data.frame(output_data[[1]]) || is(output_data[[1]], 'sparseMatrix'))) {
        stop(sprintf("Transformed X must be of a data.frame type, received %s", typeof(output_data)))
    }

    input_rows <- nrow(data)
    output_rows <- nrow(output_data[[1]])
    output_cols <- ncol(output_data[[1]])
    if (input_rows != output_rows) {
        stop(sprintf("Number of input rows (%d) does not match number of transform output rows (%d)", input_rows, output_rows))
    }
    if (output_cols <= 0) {
        stop("Number of transform output cols is 0. Must be > 0.")
    }

    # If the output data is sparse, convert it to a dataframe containing its summary. It will contain three columns
    # i, j, x where i is the row, j is the col, and x is the value of the sparse matrix. Set colnames to have a
    # special DataRobot magic value so we know the dataframe is actually a sparse matrix. In addition, add a row
    # to the end which contains [num_rows, num_cols, NaN]
    # TODO: [RAPTOR-6209] propagate column names when R output data is sparse
    if (is(output_data[[1]], 'sparseMatrix')) {
        summary_info_row = c(output_rows, output_cols, NaN)
        output_data_summary <- rbind(summary(output_data[[1]]), summary_info_row)

        output_data[[1]] <- data.frame(output_data_summary)
        colnames(output_data[[1]]) <- c("__DR__i", "__DR__j", "__DR__x")
    }

    if (!is.null(output_data[[2]])) {
        stop(sprintf("Transformation of the target variable is not supported by DRUM."))
    }

    output_data
}
