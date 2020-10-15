# Loading packages
library(tidyverse)
library(caret)
library(recipes)
library(glmnet)

# URL to dataset
TEST_DATA_ROOT <- "~/workspace/datarobot-user-models/tests/testdata"
PATH_TO_BIN_DATAFRAME <- file.path(TEST_DATA_ROOT, "iris_binary_training.csv")
PATH_TO_REG_DATAFRAME <- file.path(TEST_DATA_ROOT, "boston_housing.csv")
PATH_TO_MULTI_DATAFRAME <- file.path(TEST_DATA_ROOT, "Skyserver_SQL2_27_2018 6_51_39 PM.csv")

# Reading in dataset
bin_df <- read.csv(PATH_TO_BIN_DATAFRAME)
reg_df <- read.csv(PATH_TO_REG_DATAFRAME)
multi_df <- read.csv(PATH_TO_MULTI_DATAFRAME)

# Convert target into a character
bin_df$Species <- make.names(bin_df$Species)
multi_df$class <- make.names(multi_df$class)


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

multi_model_recipe <- recipe(class ~ ., data = multi_df) %>%
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

# Run a multiclass model
multi_model <- train(multi_model_recipe, multi_df, method = "glmnet", trControl = trainControl(method = "none", classProbs = TRUE))

# set the path of model
FIXTURE_ROOT <- "~/workspace/datarobot-user-models/tests/fixtures/drop_in_model_artifacts"
bin_model_path <- file.path(FIXTURE_ROOT, "r_bin.rds")
reg_model_path <- file.path(FIXTURE_ROOT, "r_reg.rds")
multi_model_path <- file.path(FIXTURE_ROOT, "r_multi.rds")

# Save model
saveRDS(bin_model, file = bin_model_path)
saveRDS(reg_model, file = reg_model_path)
saveRDS(multi_model, file = multi_model_path)

predict(bin_model, bin_df, type = "prob")
predict(reg_model, reg_df)
predict(multi_model, multi_df, type = "prob")
