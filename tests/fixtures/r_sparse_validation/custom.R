# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# 
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
# This custom estimator task implements a GLM regressor with sparse input data

init <- function(code_dir) {
  # load required libraries
  library(glmnet)
}

fit <- function(X, y, output_dir, class_order=NULL, row_weights=NULL, ...){
    " This hook defines how DataRobot will train this task.
    DataRobot runs this hook when the task is being trained inside a blueprint.
    As an output, this hook is expected to create an artifact containg a trained object, that is then used to score new data.
    The input parameters are passed by DataRobot based on project and blueprint configuration.

    Parameters
    -------
    X: Matrix.dgTMatrix
        Sparse training data that DataRobot passes when this task is being trained.
    y: vector
        Project's target column.
    output_dir: str
        A path to the output folder; the artifact [in this example - containing the trained GLM] must be saved into this folder.
    class_order: list
        [Only passed for a binary estimator] a list containing names of classes: first, the one that is considered negative inside DR, then the one that is considered positive.
    row_weights: list
        A list of weights. DataRobot passes it in case of smart downsampling or when weights column is specified in project settings.

    Returns
    -------
    None
        fit() doesn't return anything, but must output an artifact (typically containing a trained object) into output_dir
        so that the trained object can be used during scoring.
    "
	expected_colnames <- c(
        "abatjours",
        "abaton",
        "abator",
        "abators",
        "ABATS",
        "abattage",
        "abattis",
        "abattised",
        "abattises",
        "abattoir",
        "abattoirs",
        "abattu",
        "abattue",
        "Abatua",
        "abature",
        "abaue",
        "abave",
        "abaxial",
        "abaxile",
        "abaze",
        "abb",
        "Abba",
        "abbacy",
        "abbacies",
        "abbacomes",
        "Abbadide",
        "Abbai",
        "abbaye",
        "abbandono",
        "abbas",
        "abbasi",
        "Abbasid",
        "abbassi",
        "Abbassid",
        "Abbasside",
        "Abbate",
        "abbatial",
        "abbatical",
        "abbatie",
        "Abbe",
        "Abbey",
        "abbeys",
        "abbey's",
        "abbeystead",
        "abbeystede",
        "abbes",
        "abbess",
        "abbesses",
        "abbest",
        "Abbevilean",
        "Abbeville",
        "Abbevillian",
        "Abbi",
        "Abby",
        "Abbie",
        "Abbye",
        "Abbyville",
        "abboccato",
        "abbogada",
        "Abbot",
        "abbotcy",
        "abbotcies",
        "abbotnullius",
        "abbotric",
        "abbots",
        "abbot's",
        "Abbotsen",
        "Abbotsford",
        "abbotship",
        "abbotships",
        "Abbotson",
        "Abbotsun",
        "Abbott",
        "Abbottson",
        "Abbottstown",
        "Abboud",
        "abbozzo",
        "ABBR",
        "abbrev",
        "abbreviatable",
        "abbreviate",
        "abbreviated",
        "abbreviately",
        "abbreviates",
        "abbreviating",
        "abbreviation",
        "abbreviations",
        "abbreviator",
        "abbreviatory",
        "abbreviators",
        "abbreviature",
        "abbroachment",
        "ABC",
        "abcess",
        "abcissa",
        "abcoulomb",
        "ABCs",
        "abd",
        "abdal",
        "abdali",
        "abdaria",
        "abdat",
        "Abdel",
        "Abd-el-Kadir",
        "Abd-el-Krim",
        "Abdella",
        "Abderhalden",
        "Abderian",
        "Abderite",
        "Abderus",
        "abdest",
        "Abdias",
        "abdicable",
        "abdicant",
        "abdicate",
        "abdicated",
        "abdicates",
        "abdicating",
        "abdication",
        "abdications",
        "abdicative",
        "abdicator",
        "Abdiel",
        "abditive",
        "abditory",
        "abdom",
        "abdomen",
        "abdomens",
        "abdomen's",
        "abdomina",
        "abdominal",
        "Abdominales",
        "abdominalia",
        "abdominalian",
        "abdominally",
        "abdominals",
        "abdominoanterior",
        "abdominocardiac",
        "abdominocentesis",
        "abdominocystic",
        "abdominogenital",
        "abdominohysterectomy",
        "abdominohysterotomy",
        "abdominoposterior",
        "abdominoscope",
        "abdominoscopy",
        "abdominothoracic",
        "abdominous",
        "abdomino-uterotomy",
        "abdominovaginal",
        "abdominovesical",
        "Abdon",
        "Abdu",
        "abduce",
        "abduced",
        "abducens",
        "abducent",
        "abducentes",
        "abduces",
        "abducing",
        "abduct",
        "abducted"
    )
    # TODO: Never have weights in sparse input X
	if("some-weights" %in% colnames(X)) {
		expected_colnames <- c(expected_colnames, "some-weights")
	}
	stopifnot(colnames(X) == expected_colnames)
	stopifnot(is(X, 'sparseMatrix'))

    # Cast dgTMatrix to dgCMatrix
    X <- as(X, "dgCMatrix")

    # train model
    e <- glmnet(X, y)

    # Save model
    model_path <- file.path(output_dir, 'artifact.rds')
    saveRDS(e, file = model_path)
}

score <- function(data, model, ...){
    " This hook defines how DataRobot will use the trained object from fit() to score new data.
    DataRobot runs this hook when the task is used for scoring inside a blueprint. 
    As an output, this hook is expected to return the scored data.
    The input parameters are passed by DataRobot based on dataset and blueprint configuration.

    Parameters
    -------
    data: Matrix.dgTMatrix
        Sparse input data that DataRobot passes for scoring.
    model: Any
        Trained object, extracted by DataRobot from the artifact created in fit().
        In this example, contains trained glmnet extracted from artifact.rds.
    
    Returns
    -------
    data.frame
        Returns a dataframe with scored data
        In case of regression, score() must return a dataframe with a single column with column name 'Predictions'.
    "
    # Cast dgTMatrix to dgCMatrix
    data <- as(data, "dgCMatrix")

    # Predict, and set the column name to 'Predictions'
    predictions <- data.frame(predict(model, newx = data, s=0.01, type="response"))
    colnames(predictions) <- "Predictions"

    return(predictions)
}
