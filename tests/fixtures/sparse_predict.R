# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# 
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
init <- function(code_dir) {
  # custom init function to load required libraries
#  library(tidyverse)
#  library(caret)
#  library(recipes)
#  library(e1071)
#  library(gbm)
    library(glmnet)
}

score <- function(data, model, ...){
    " This hook defines how DataRobot will use the trained object from fit() to score new data.
    DataRobot runs this hook when the task is used for scoring inside a blueprint.
    As an output, this hook is expected to return the scored data.
    The input parameters are passed by DataRobot based on dataset and blueprint configuration.

    Parameters
    -------
    data: data.frame
        Data that DataRobot passes for scoring.
    model: Any
        Trained object, extracted by DataRobot from the artifact created in fit().
        In this example, contains trained GLM extracted from artifact.rds.

    Returns
    -------
    data.frame
        Returns a dataframe with scored data
        In case of regression, score() must return a dataframe with a single column with column name 'Predictions'.
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
	stopifnot(colnames(data) == expected_colnames)
	stopifnot(is(data, 'sparseMatrix'))
    return(data.frame(Predictions = predict(model, newx=data, type = "response")[,"s0"]))
}
