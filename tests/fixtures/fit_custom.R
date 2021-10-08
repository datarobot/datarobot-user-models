init <- function(...) {
  # custom init function to load required libraries
  library(tidyverse)
  library(caret)
  library(recipes)
  library(e1071)
  library(gbm)
}

fit <- function(X, y, output_dir, class_order=NULL, row_weights=NULL, ...){
    #' User-provided fit method, required for custom training
    #'
    #' Trains a regression or classification model using gbm (via caret)
    #' @param X data.frame - training data to perform fit on
    #' @param y data.frame column or array - target data to perform fit on
    #' @param output_dir the path to write output. This is the path provided in '--output' parameter of the
    #' 'drum fit' command.
    #' @param class_order : A two element long list dictating the order of classes which should be used for
    #'  modeling. Class order will always be passed to fit by DataRobot for classification tasks,
    #'  and never otherwise. When models predict, they output a likelihood of one class, with a
    #'  value from 0 to 1. The likelihood of the other class is 1 - this likelihood. Class order
    #'  dictates that the first element in the list will be the 0 class, and the second will be the
    #'  1 class.
    #' @param row_weights: An array of non-negative numeric values which can be used to dictate how important
    #'  a row is. Row weights is only optionally used, and there will be no filtering for which
    #'  custom models support this. There are two situations when values will be passed into
    #'  row_weights, during smart downsampling and when weights are explicitly provided by the user
    #' @param ...: Added for forwards compatibility
    #' @return Nothing

  # set up dataframe for modeling
  train_df <- X
  train_df$target <- unlist(y)

  if (!is.null(class_order)){
    # maybe add more processing to y if needed? R might not care
    outfile <- 'r_classif.rds'
  }else{
    outfile <-'r_reg.rds'
  }

  # set up the modeling pipeline
  model_recipe <- recipe(target ~ ., data = train_df) %>%
    # Drop constant columns
    step_zv(all_predictors()) %>%
    # Numeric preprocessing
    step_medianimpute(all_numeric()) %>%
    step_normalize(all_numeric(), -all_outcomes()) %>%
    # Categorical preprocessing
    step_other(all_nominal(), -all_outcomes()) %>%
    step_dummy(all_nominal(), -all_outcomes())

  # Run the model
  set.seed(42)
  model <- train(model_recipe, train_df, method = "gbm")

  # Save model
  if(
    substr(output_dir,
          nchar(output_dir),
          nchar(output_dir)) == '/'
    ) {
    seperator = ''
  } else {
    seperator = '/'
  }

  model_path <- file.path(
    paste(
      output_dir, outfile, sep=seperator
    )
  )
  saveRDS(model, file = model_path)
}


#"
#    Below are all the hooks you can use to provide your own implementation.
#    All hooks are currently commented out so uncomment a hook function in
#    order to use it.
#"
#
##' This hook can be implemented to adjust logic in the training and scoring mode.
##'
##' This hook is executed once the code is started.
##' Load any libraries your serialized model requires so that scoring will work.
##' For example, if your model was trained using method='brnn', you will need
##' to add library(brnn) in this method
##'
##' @param ... keyword args, additional params to the method.
##' code_dir - code folder passed in --code_dir argument
##'
##' @return NA, this method is intended for side effects only
##' @export
##'
##' @examples
#init <- function(...) {
#}
#
##' This hook can be implemented to adjust logic in the scoring mode.
##'
##' Modify this method to deserialize your model, if this environment's standard
##' model loader is insufficient. For example, if your code directory
##' contains multiple rds files, you must explicitly load which ever one
##' corresponds to your serialized model here
##'
##' @return the deserialized model
##' @export
##'
##' @examples
#load_model <- function(input_dir) {
#    # Returning a string with value "dummy" as the model.
#    "dummy"
#}
#
##' This hook can be implemented to adjust logic in the scoring mode.
##'
##' Intended to apply transformations to the prediction data before making
##' predictions. This is most useful if drum supports the model's library, but
##' your model requires additional data processing before it can make predictions
##'
##' @param data data.frame, data.frame given to drum to make predictions on
##' @param model Any, is the deserialized model loaded by drum or by the
##' load_model hook, if supplied
##'
##' @return data.frame, a data.frame after transformation needed
##' @export
##'
##' @examples
#transform <- function(data, model) {
#    data
#}
#
##' This hook can be implemented to adjust logic in the scoring mode.
##'
##' This method should return predictions as a data.frame with the following
##' format:
##'
##' Binary Classification:
##' * Must have columns for each class label with floating-point class
##'   probabilities as values.
##' * Each row should sum to 1.0
##'
##' Regression:
##' * Must have a single column called "Predictions" with numerical values
##'
##' This hook is only needed if you would like to use drum with a framework not
##' natively supported by the tool.
##'
##' @param data data.frame, the data.frame to make predictions against. If
##' transform is supplied, data will be the transformed data.
##' @param model Any, is the deserialized model loaded by drum or by load_model
##' hook, if supplied
##' @param ... keyword args, additional params to the method. If model is
##' classification, positive_class_label and negative_class_label will be
##' provided as parameters.
##' @return: data.frame with the structure defined above
##' @export
##'
##' @examples
#score <- function(data, model, ...) {
#    # return a data.frame with a 'Predictions' column with a single value of 42
#    data.frame('Predictions'=42)
#}
#
##' This hook can be implemented to adjust logic in the scoring mode.
##'
##' This method should return predictions as a data.frame with the following
##' format:
##'
##' Binary Classification:
##' * Must have columns for each class label with floating-point class
##'   probabilities as values.
##' * Each row should sum to 1.0
##'
##' Regression:
##' * Must have a single column called `Predictions` with numerical values
##'
##' This method is required if your model's output does not match the above
##' expectations, or if additional transformations are desired on your model's
##' raw outputs.
##'
##' @param predictions data.frame, predictions produced by `cmrun` or by the
##' `score` hook, if supplied
##' @param model Any, the deserialized model loaded by `cmrun` or by
##' `load_model`, if supplied
##' @return: data.frame with the structure defined above
##' @export
##'
##' @examples
#post_process <- function(predictions, model) {
#}
