library(recipes)

fit <- function(X, y, output_dir, class_order=NULL, row_weights=NULL, ...){
  "
  This hook defines how DataRobot will train this task.
  DataRobot runs this hook when the task is being trained inside a blueprint.
  As an output, this hook is expected to create an artifact containg a trained object, that is then used to transform new data.
  The input parameters are passed by DataRobot based on project and blueprint configuration.

  Parameters
  -------
  X: data.frame
      Training data that DataRobot passes when this task is being trained.
  y: vector
      Project's target column.
  output_dir: str
      A path to the output folder; the artifact [in this example - containing the trained recipe] must be saved into this folder.
  class_order: list
      [Only passed for a binary estimator] a list containing names of classes: first, the one that is considered negative inside DR, then the one that is considered positive.
  row_weights: list
      A list of weights. DataRobot passes it in case of smart downsampling or when weights column is specified in project settings.

  Returns
  -------
  None
      fit() doesn't return anything, but must output an artifact (typically containing a trained object) into output_dir
      so that the trained object can be used during transform.
  "
  # set up dataframe for modeling
  train_df <- X
  train_df$target <- unlist(y)

  outfile <- 'r_transform.rds'

  # set up the preprocessing pipeline
  rcp <- recipe(target ~ ., data = train_df)
  model_recipe <- rcp %>%
    # Drop constant columns
    step_zv(all_predictors()) %>%
    # Numeric preprocessing
    step_medianimpute(all_numeric()) %>%
    step_normalize(all_numeric(), -all_outcomes()) %>%
    # Categorical preprocessing
    step_other(all_nominal(), -all_outcomes()) %>%
    step_dummy(all_nominal(), -all_outcomes())

  # prep (fit) the preprocessing recipe
  model <- prep(model_recipe, train_df)

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

# transform <- function(X, transformer, ...){
#   "
#   This hook defines how DataRobot will use the trained object from fit() to transform new data.
#   DataRobot runs this hook when the task is used for scoring inside a blueprint.
#   As an output, this hook is expected to return the scored data.
#   The input parameters are passed by DataRobot based on dataset and blueprint configuration.
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
#       In this example, contains the trained recipe extracted from r_transform.rds.
#
#   Returns
#   -------
#   data.frame
#       Returns a dataframe with scored data
#       In case of regression, score() must return a dataframe with a single column with column name 'Predictions'.
#   "
#   bake(transformer, X)
# }