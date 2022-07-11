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

fit <- function(X, y, output_dir, class_order=NULL, row_weights=NULL, ...){
  # set up dataframe for modeling
  train_df <- X

  if (is.null(y)) {
    rcp <- recipe(train_df)
  } else {
    train_df$target <- unlist(y)
    rcp <- recipe(target ~ ., data = train_df)
  }

  outfile <- 'r_transform.rds'

  # set up the preprocessing pipeline
  model_recipe <- rcp %>%
    # Drop constant columns
    step_zv(all_predictors()) %>%
    # Numeric preprocessing
    # step_impute_median in the current example stopped working with R4.2.1
    # step_impute_median(all_numeric()) %>%
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