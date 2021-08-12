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
  median = apply(X, 2, median)

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
  saveRDS(median, file = model_path)
}

transform <- function(X, transformer, ...){
  "
  This hook defines how DataRobot will use the trained object from fit() to transform new data.
  As an output, this hook is expected to return the transformed data.

  Parameters
  -------
  X: data.frame
      Data that DataRobot passes for transforming.
  transformer: Any
      Trained object, extracted by DataRobot from the artifact created in fit().
      In this example, transformer contains the median values stored in a vector.

  Returns
  -------
  data.frame
      Returns the transformed dataframe
  "
  median <- transformer

  # replace NA values with the median values for each column
  for (i in 1:ncol(X)) {
    X[is.na(X[,i]), i] <- median[i]
  }
  X
}