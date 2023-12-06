# This code is for the Otto Product Classification Competition in Kaggle

# Purpose of the competition -----------------------

# The otto group has many products that are used worldwide and sometimes
# get classified differently. Evaluating a product correctly requires putting
# it into the context of similar products. For example, if the otto group sold
# road bikes, mountain bikes, and donuts, you would expect the performance of 
# road bikes to differ significantly from donuts, but not necessarily from
# mountain bikes.
# Being able to accurately classify and cluster products into similarity groups
# is essential to evaluating product performance. Hence the competition to
# find how to more accurately classify these products.

# Libraries -----------------------------------
library(tidymodels)
library(vroom)
library(tidyverse)
library(embed)
library(ranger)
library(discrim)
library(themis) # for SMOTE
library(bonsai)
library(lightgbm)

# Data read-in and description -------------------------------

# Data fields
#   id - anonymous id unique to each product
#   feat_1, feat_2, feat_3, ..., feat_93 - the various features of a product
#   target - the class of a product
# Data description
#   Each row corresponds to a single product. The 93 features are all numerical,
#   and represent counts of various events. These events are not described,
#   and are simply labeled as feat_1, feat_2, etc. for analysis purposes.
#   There are 9 possible classification categories. Each target category 
#   represents one of the most important product categories for Otto Group.
# Submission details
#   Each row should represent a singular product as labeled by its id. The
#   predictions should be submitted as a separate probability for each
#   classification. 
#   E.g. 0.34 for Class_1, 0.005 for Class_2, 0.8 for Class_3, etc.

base_folder <- "OttoProductClassification/"
otto_train <- vroom(paste0(base_folder, "train.csv"))
otto_test <- vroom(paste0(base_folder, "test.csv"))
glimpse(otto_train)
glimpse(otto_test)

# EDA ----------------------

# General strategy

# Create the final submission -------------------------
otto_submission <- data.frame("id" = var0,
                              "Class_1" = var1,
                              "Class_2" = var2,
                              "Class_3" = var3,
                              "Class_4" = var4,
                              "Class_5" = var5,
                              "Class_6" = var6,
                              "Class_7" = var7,
                              "Class_8" = var8,
                              "Class_9" = var9)