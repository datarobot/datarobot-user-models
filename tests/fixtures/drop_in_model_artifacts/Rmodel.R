# Loading packages
library(tidyverse)
library(caret)
library(recipes)
library(glmnet)

# URL to dataset
PATH_TO_BIN_DATAFRAME <- "iris_binary_training.csv"
PATH_TO_REG_DATAFRAME <- "boston_housing.csv"

# Reading in dataset
bin_df <- read.csv(PATH_TO_BIN_DATAFRAME)
reg_df <- read.csv(PATH_TO_REG_DATAFRAME)

# Convert target into a character
bin_df$Species <- make.names(bin_df$Species)


# Modeling pipeline
bin_model_recipe <- recipe(Species ~ ., data = bin_df) %>%
  # Drop constant columns
  step_zv(all_predictors()) %>% 
  # Numeric preprocessing
  step_medianimpute(all_numeric()) %>%
  step_normalize(all_numeric(), -all_outcomes()) %>% 
  # Categorical preprocessing
  step_other(all_nominal(), -all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes())

reg_model_recipe <- recipe(MEDV ~ ., data = reg_df) %>%
  # Drop constant columns
  step_zv(all_predictors()) %>% 
  # Numeric preprocessing
  step_medianimpute(all_numeric()) %>%
  step_normalize(all_numeric(), -all_outcomes()) %>% 
  # Categorical preprocessing
  step_other(all_nominal(), -all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes())

# Run regularized logistic regression
bin_model <- train(bin_model_recipe, bin_df, method = "glmnet", trControl = trainControl(method = "none", classProbs = TRUE))

# Run a regression model
reg_model <- train(reg_model_recipe, reg_df, method = "brnn")

# set the path of model
bin_model_path = file.path("r_bin.rds")
reg_model_path = file.path("r_reg.rds")

# Save model
saveRDS(bin_model, file = bin_model_path)
saveRDS(reg_model, file = reg_model_path)

predict(bin_model, bin_df, type = "prob")
predict(reg_model, reg_df)
