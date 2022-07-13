# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# 
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
library(recipes)

fit <- function(X, y, output_dir, ...){
  "
  This hook defines how DataRobot will train this task.
  DataRobot runs this hook when the task is being trained inside a blueprint.
  As an output, this hook is expected to create an artifact containing a trained object, that is then used to transform new data.

  Parameters
  -------
  X: data.frame
      Training data that DataRobot passes when this task is being trained.
  y: vector
      Project's target column.
  output_dir: str
      A path to the output folder; the artifact [in this example - containing the trained recipe] must be saved into this folder.

  Returns
  -------
  None
      fit() doesn't return anything, but must output an artifact (typically containing a trained object) into output_dir
      so that the trained object can be used during transform.
  "
  # set up dataframe for modeling
  train_df <- X

  # y can be null if performing anomaly detection
  if (is.null(y)) {
    rcp <- recipe(train_df)
  } else {
    train_df$target <- unlist(y)
    rcp <- recipe(target ~ ., data = train_df)
  }

  # set up the preprocessing recipe
  model_recipe <- rcp %>%
    # Drop constant columns
    step_zv(all_predictors()) %>%
    # Numeric preprocessing
    step_impute_median(all_numeric()) %>%
    step_normalize(all_numeric(), -all_outcomes()) %>%
    # Categorical preprocessing
    step_other(all_nominal(), -all_outcomes()) %>%
    step_dummy(all_nominal(), -all_outcomes())

  # prep (fit) the preprocessing recipe
  model <- prep(model_recipe, train_df)

  # Save model
  outfile <- 'r_transform.rds'
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

# transform <- function(X, transformer, ...){
#   "
#   This hook defines how DataRobot will use the trained object from fit() to transform new data.
#   As an output, this hook is expected to return the transformed data.
#
#   NOTE: If fit() outputs a single recipe artifact, then you do not need to populate the transform() function. It
#   will automatically call bake as shown below. If your artifact is anything else, you must populate transform().
#   In addition, update the model-metadata.yaml file to reflect new input and output requirements.
#
#   Parameters
#   -------
#   X: data.frame
#       Data that DataRobot passes for transforming.
#   transformer: Any
#       Trained object, extracted by DataRobot from the artifact created in fit().
#       In this example, contains the trained recipe extracted from r_transform_recipe.rds.
#
#   Returns
#   -------
#   data.frame
#       Returns the transformed dataframe
#   "
#   bake(transformer, X)
# }