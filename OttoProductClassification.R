# Introduction ---------------------------

# The Otto Group has many products that are used worldwide. Evaluating a product 
# correctly requires comparing it to similar products. For example, if the 
# Otto Group sells road bikes, mountain bikes, and donuts, you would expect 
# the performance of road bikes to differ significantly from donuts, but not 
# necessarily from mountain bikes.
# Being able to accurately classify products into similarity groups
# is essential to evaluating product performance. The goal of this competition 
# is to find the model that predicts the classification for a product 
# as accurately as possible.

# ### Data fields
# - id - anonymous id unique to each product
# - feat_1, feat_2, feat_3, ..., feat_93 - the various features of a product
# - target - the class of a product
# 
# ### Data description
# Each row corresponds to a single product. The 93 features are all numerical,
# and represent counts of various events. These events are not described,
# and are simply labeled as feat_1, feat_2, etc. for analysis purposes.
# There are 9 possible target classification categories. Each target category 
# represents one of the most important product categories for Otto Group.
# ### Submission details
# Each row should represent a singular product as labeled by its id. The
# predictions should be submitted as a separate probability for each
# classification. 
# E.g. 0.34 for Class_1, 0.005 for Class_2, 0.8 for Class_3, etc.

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

# Read in the data -------------------------------
base_folder <- "/kaggle/input/otto-group-product-classification-challenge/"
otto_train <- vroom(paste0(base_folder, "train.csv"))
otto_test <- vroom(paste0(base_folder, "test.csv"))
# Don't use the id as a predictor.
glimpse(otto_train)
glimpse(otto_test)

# # EDA ----------------------
# 
# # Check to see how balanced the group assignments are
# class_balance_frame <- otto_train %>% group_by(target) %>% summarize("total" = n())
# class_balance_frame
# class_balance_plot <- class_balance_frame %>% 
#   ggplot(aes(x = target, y = total)) + geom_col(fill="turquoise4")
# # Class assignments are not balanced. They are dominated by classes 2 and 6.
# class_balance_plot
# ggsave(filename = paste0(base_folder, "Class Balance Plot.png"),
#        plot = class_balance_plot)
# 
# # For each classification group, make a column chart of all features
# feature_counts_chart <- function(class_num) {
#   otto_train %>% filter(target == paste0("Class_", class_num)) %>% 
#     select(-target, -id) %>% apply(MARGIN=2, FUN=sum) %>%
#     data.frame("feature" = factor(1:length(names(.))), "class_total" = unname(.)) %>% 
#     ggplot(aes(x = feature, y = class_total)) + geom_col(fill = "hotpink3")
# }
# 
# feature_counts_panel_plot <- plotly::subplot(feature_counts_chart(1),
#                                              feature_counts_chart(2),
#                                              feature_counts_chart(3),
#                                              feature_counts_chart(4),
#                                              feature_counts_chart(5),
#                                              feature_counts_chart(6),
#                                              feature_counts_chart(7),
#                                              feature_counts_chart(8),
#                                              feature_counts_chart(9),
#                                              nrows=3)
# feature_counts_panel_plot
# 
# # It seems to me that there is enough distinction between the different
# #   kinds of products by class. I'll focus first on trying to run a model
# #   that just accounts for balancing the data.

# Recipes -----------------------------------
smote_neighbors <- 10
threshold_value <- 0.7
tree_smote_recipe <- recipe(target ~ ., data = otto_train) %>% 
  update_role(id, new_role = "Id") %>% 
  # id will stay in the dataset but won't be used as a predictor
  step_mutate_at(all_outcomes(), fn = factor, skip = TRUE) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_smote(all_outcomes(), neighbors = smote_neighbors)
tree_downsample_recipe <- recipe(target ~ ., data = otto_train) %>%
  update_role(id, new_role = "Id") %>% 
  # id will stay in the dataset but won't be used as a predictor
  step_mutate_at(all_outcomes(), fn = factor, skip = TRUE) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_downsample(all_outcomes())
tree_pca_downsample_recipe <- recipe(target ~ ., data = otto_train) %>%
  update_role(id, new_role = "Id") %>% 
  # id will stay in the dataset but won't be used as a predictor
  step_mutate_at(all_outcomes(), fn = factor, skip = TRUE) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_downsample(all_outcomes()) %>%
  step_pca(all_predictors(), threshold = threshold_value)
tree_upsample_recipe <- recipe(target ~ ., data = otto_train) %>%
  update_role(id, new_role = "Id") %>% 
  # id will stay in the dataset but won't be used as a predictor
  step_mutate_at(all_outcomes(), fn = factor, skip = TRUE) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_upsample(all_outcomes())
penalized_mnm_downsample_recipe <- recipe(target ~ ., data = otto_train) %>% 
  update_role(id, new_role = "Id") %>% 
  # id will stay in the dataset but won't be used as a predictor
  step_mutate_at(all_outcomes(), fn = factor, skip = TRUE) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_downsample(all_outcomes())
penalized_mnm_upsample_recipe <- recipe(target ~ ., data = otto_train) %>% 
  update_role(id, new_role = "Id") %>% 
  # id will stay in the dataset but won't be used as a predictor
  step_mutate_at(all_outcomes(), fn = factor, skip = TRUE) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_upsample(all_outcomes())
penalized_mnm_smote_recipe <- recipe(target ~ ., data = otto_train) %>%
  update_role(id, new_role = "Id") %>% 
  # id will stay in the dataset but won't be used as a predictor
  step_mutate_at(all_outcomes(), fn = factor, skip = TRUE) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_smote(all_outcomes(), neighbors = smote_neighbors)

# Data Balancing/Resizing --------------------------------------------
# Max possible iterations for manual downsampling
class_balance_frame <- otto_train %>% group_by(target) %>% summarize("total" = n())
max_downsample_size <- min(class_balance_frame$total)

# Create small dataset to test models more quickly
sample_size <- 200
num_classes <- 3
set.seed(12)
for (i in 1:num_classes) {
  if (i == 1) {
    # Filter to just one class
    small_otto_train <- otto_train %>% filter(target == paste0("Class_", i)) %>%
      # then take a sample from that class
      .[sample(1:length(.$feat_1), size = sample_size),]
  } else {
    # Filter to just one class
    small_otto_train <- otto_train %>% filter(target == paste0("Class_", i)) %>%
      # then take a sample from that class
      .[sample(1:length(.$feat_1), size = sample_size),] %>%
      # then add this sample to the existing samples
      bind_rows(small_otto_train, .)
  }
}
# small_otto_train


# Create medium dataset to leverage available data but not overdo runtime
# I.e. manual downsampling of the majority classes, recipe upsampling of minority classes
summarized_train <- otto_train %>% group_by(target) %>% summarize("class_count" = n())
medium_sample_size <- summarized_train$class_count %>% mean()

num_classes <- 9
medium_sample_size <- floor(medium_sample_size)
for (i in 1:num_classes) {
  if (i == 1) {
    # Filter to just one class
    target_class <- otto_train %>% filter(target == paste0("Class_", i))
    class_count <- length(target_class$feat_1)
    
    # Then take a sample from that class
    # Minority classes -> keep all the data (will be upsampled to the average class_count)
    # Majority classes -> sample down to the average class_count
    medium_otto_train <- target_class[sample(1:class_count, size = min(medium_sample_size, class_count)),]
  } else {
    # Filter to just one class
    target_class <- otto_train %>% filter(target == paste0("Class_", i))
    class_count <- length(target_class$feat_1)
    
    # Then add this class's sample to the overall
    # Minority classes -> keep all the data (will be upsampled to the average class_count)
    # Majority classes -> sample down to the average class_count
    medium_otto_train <- bind_rows(medium_otto_train, target_class[sample(1:class_count, size = min(medium_sample_size, class_count)),])
  }
}
# medium_otto_train

# # Random Forest SMOTE ---------------------------------
# 
# otto_forest_smote_model <- rand_forest(mtry = tune(),
#                                        min_n = tune(),
#                                        trees = 100) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# # Create a workflow using the model and recipe
# otto_forest_smote_wf <- workflow() %>%
#   add_model(otto_forest_smote_model) %>%
#   add_recipe(tree_smote_recipe)
# 
# # Set up the grid with the tuning values
# num_non_predictors <- 2
# otto_forest_smote_grid <- grid_regular(mtry(range = c(1, (length(otto_train)-num_non_predictors))), 
#                                        min_n(), levels = 3)
# 
# # Set up the K-fold CV
# otto_forest_smote_folds <- vfold_cv(data = otto_train, v = 5, repeats = 1)
# 
# # Find best tuning parameters
# forest_cv_results <- otto_forest_smote_wf %>%
#   tune_grid(resamples = otto_forest_smote_folds,
#             grid = otto_forest_smote_grid,
#             metrics = metric_set(roc_auc))
# 
# # Finalize the workflow using the best tuning parameters and predict
# # The best parameters were mtry = 9 and min_n = 2
# 
# # Find out the best tuning parameters
# best_forest_tune <- forest_cv_results %>% select_best("roc_auc")
# 
# # Use the best tuning parameters for the model
# forest_final_wf <- otto_forest_smote_wf %>%
#   finalize_workflow(best_forest_tune) %>%
#   fit(data = otto_train)
# 
# otto_forest_smote_predictions <- predict(forest_final_wf,
#                                          new_data = otto_test,
#                                          type = "prob")
# otto_forest_smote_predictions

# #  Upsampling Forest ---------------------------------
# 
# # Update the upsample recipe to use the partially downsampled data
# tree_upsample_recipe <- recipe(target ~ ., data = medium_otto_train) %>%
#   update_role(id, new_role = "Id") %>% 
#   # id will stay in the dataset but won't be used as a predictor
#   step_mutate_at(all_outcomes(), fn = factor, skip = TRUE) %>% 
#   step_normalize(all_numeric_predictors()) %>% 
#   step_upsample(all_outcomes())
# 
# otto_forest_upsample_model <- rand_forest(mtry = tune(),
#                                           min_n = tune(),
#                                           trees = 500) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# # Create a workflow using the model and recipe
# otto_forest_upsample_wf <- workflow() %>%
#   add_model(otto_forest_upsample_model) %>%
#   add_recipe(tree_upsample_recipe)
# 
# # Set up the grid with the tuning values
# num_non_predictors <- 2
# otto_forest_upsample_grid <- grid_regular(mtry(range = c(1, (length(medium_otto_train)-num_non_predictors))),
#                                           min_n(), levels = 3)
# 
# # Set up the K-fold CV
# otto_forest_upsample_folds <- vfold_cv(data = medium_otto_train, v = 3, repeats = 1)
# 
# # Find best tuning parameters
# forest_cv_results <- otto_forest_upsample_wf %>%
#   tune_grid(resamples = otto_forest_upsample_folds,
#             grid = otto_forest_upsample_grid,
#             metrics = metric_set(accuracy))
# best_forest_tune <- forest_cv_results %>% select_best("accuracy")
# 
# # Use the best tuning parameters for the model
# forest_final_wf <- otto_forest_upsample_wf %>%
#   finalize_workflow(best_forest_tune) %>%
#   fit(data = medium_otto_train)
# 
# otto_forest_upsample_predictions <- predict(forest_final_wf,
#                                             new_data = otto_test,
#                                             type = "prob")
# otto_forest_upsample_predictions
# 
# # Create the submission
# otto_forest_upsample_submission <- data.frame("id" = otto_test$id,
#                                               "Class_1" = otto_forest_upsample_predictions$.pred_Class_1,
#                                               "Class_2" = otto_forest_upsample_predictions$.pred_Class_2,
#                                               "Class_3" = otto_forest_upsample_predictions$.pred_Class_3,
#                                               "Class_4" = otto_forest_upsample_predictions$.pred_Class_4,
#                                               "Class_5" = otto_forest_upsample_predictions$.pred_Class_5,
#                                               "Class_6" = otto_forest_upsample_predictions$.pred_Class_6,
#                                               "Class_7" = otto_forest_upsample_predictions$.pred_Class_7,
#                                               "Class_8" = otto_forest_upsample_predictions$.pred_Class_8,
#                                               "Class_9" = otto_forest_upsample_predictions$.pred_Class_9)
# otto_forest_upsample_submission
# vroom_write(otto_forest_upsample_submission, "submission.csv", delim = ",")

# #  Downsampling Forest ---------------------------------
# 
# otto_forest_downsample_model <- rand_forest(mtry = tune(),
#                                             min_n = tune(),
#                                             trees = 500) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# # Create a workflow using the model and recipe
# otto_forest_downsample_wf <- workflow() %>%
#   add_model(otto_forest_downsample_model) %>%
#   add_recipe(tree_downsample_recipe)
# 
# # Set up the grid with the tuning values
# num_non_predictors <- 2
# otto_forest_downsample_grid <- grid_regular(mtry(range = c(1, (length(otto_train)-num_non_predictors))),
#                                             min_n(), levels = 3)
# 
# # Set up the K-fold CV
# otto_forest_downsample_folds <- vfold_cv(data = otto_train, v = 4, repeats = 1)
# 
# # Find best tuning parameters
# forest_cv_results <- otto_forest_downsample_wf %>%
#   tune_grid(resamples = otto_forest_downsample_folds,
#             grid = otto_forest_downsample_grid,
#             metrics = metric_set(roc_auc))
# best_forest_tune <- forest_cv_results %>% select_best("roc_auc")
# 
# # Use the best tuning parameters for the model
# forest_final_wf <- otto_forest_downsample_wf %>%
#   finalize_workflow(best_forest_tune) %>%
#   fit(data = otto_train)
# 
# otto_forest_downsample_predictions <- predict(forest_final_wf,
#                                               new_data = otto_test,
#                                               type = "prob")
# otto_forest_downsample_predictions
# 
# # Create the submission
# otto_forest_downsample_submission <- data.frame("id" = otto_test$id,
#                                                 "Class_1" = otto_forest_downsample_predictions$.pred_Class_1,
#                                                 "Class_2" = otto_forest_downsample_predictions$.pred_Class_2,
#                                                 "Class_3" = otto_forest_downsample_predictions$.pred_Class_3,
#                                                 "Class_4" = otto_forest_downsample_predictions$.pred_Class_4,
#                                                 "Class_5" = otto_forest_downsample_predictions$.pred_Class_5,
#                                                 "Class_6" = otto_forest_downsample_predictions$.pred_Class_6,
#                                                 "Class_7" = otto_forest_downsample_predictions$.pred_Class_7,
#                                                 "Class_8" = otto_forest_downsample_predictions$.pred_Class_8,
#                                                 "Class_9" = otto_forest_downsample_predictions$.pred_Class_9)
# otto_forest_downsample_submission
# vroom_write(otto_forest_downsample_submission, "submission.csv", delim = ",")

# # PCA Downsampling Forest ---------------------------------
# 
# otto_forest_pca_downsample_model <- rand_forest(mtry = tune(),
#                                                 min_n = tune(),
#                                                 trees = 500) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# # Create a workflow using the model and recipe
# otto_forest_pca_downsample_wf <- workflow() %>%
#   add_model(otto_forest_pca_downsample_model) %>%
#   add_recipe(tree_pca_downsample_recipe)
# 
# # Set up the grid with the tuning values
# num_non_predictors <- 2
# otto_forest_pca_downsample_grid <- grid_regular(mtry(range = c(1, (length(otto_train)-num_non_predictors))),
#                                                 min_n(), levels = 3)
# 
# # Set up the K-fold CV
# otto_forest_pca_downsample_folds <- vfold_cv(data = otto_train, v = 4, repeats = 1)
# 
# # Find best tuning parameters
# forest_cv_results <- otto_forest_pca_downsample_wf %>%
#   tune_grid(resamples = otto_forest_pca_downsample_folds,
#             grid = otto_forest_pca_downsample_grid,
#             metrics = metric_set(roc_auc))
# best_forest_tune <- forest_cv_results %>% select_best("roc_auc")
# 
# # Use the best tuning parameters for the model
# forest_final_wf <- otto_forest_pca_downsample_wf %>%
#   finalize_workflow(best_forest_tune) %>%
#   fit(data = otto_train)
# 
# otto_forest_pca_downsample_predictions <- predict(forest_final_wf,
#                                                   new_data = otto_test,
#                                                   type = "prob")
# otto_forest_pca_downsample_predictions
# 
# # Create the submission
# otto_forest_pca_downsample_submission <- data.frame("id" = otto_test$id,
#                                                     "Class_1" = otto_forest_pca_downsample_predictions$.pred_Class_1,
#                                                     "Class_2" = otto_forest_pca_downsample_predictions$.pred_Class_2,
#                                                     "Class_3" = otto_forest_pca_downsample_predictions$.pred_Class_3,
#                                                     "Class_4" = otto_forest_pca_downsample_predictions$.pred_Class_4,
#                                                     "Class_5" = otto_forest_pca_downsample_predictions$.pred_Class_5,
#                                                     "Class_6" = otto_forest_pca_downsample_predictions$.pred_Class_6,
#                                                     "Class_7" = otto_forest_pca_downsample_predictions$.pred_Class_7,
#                                                     "Class_8" = otto_forest_pca_downsample_predictions$.pred_Class_8,
#                                                     "Class_9" = otto_forest_pca_downsample_predictions$.pred_Class_9)
# otto_forest_pca_downsample_submission
# vroom_write(otto_forest_pca_downsample_submission, "submission.csv", delim = ",")

# # Penalized Multinomial Regression Model (Downsampling)-----------------------
# 
# # Set the model
# penalized_mnm_mod <- multinom_reg(mixture = tune(), penalty = tune()) %>%
#   set_engine("glmnet")
# 
# # Set the workflow
# penalized_mnm_wf <- workflow() %>%
#   add_recipe(penalized_mnm_downsample_recipe) %>%
#   add_model(penalized_mnm_mod)
# 
# # Set the tuning grid
# penalized_mnm_tuning_grid <- grid_regular(penalty(),
#                                           mixture(),
#                                           levels = 3)
# 
# # Set up the CV
# penalized_mnm_folds <- vfold_cv(otto_train, v = 5, repeats = 1)
# 
# # Run the CV
# penalized_mnm_CV_results <- penalized_mnm_wf %>%
#   tune_grid(resamples = penalized_mnm_folds,
#             grid = penalized_mnm_tuning_grid,
#             metrics = metric_set(roc_auc)) #, f_meas, sens, recall, spec,
# # precision, accuracy))
# 
# # Find out the best tuning parameters
# penalized_mnm_best_tune <- penalized_mnm_CV_results %>% select_best("roc_auc")
# penalized_mnm_best_tune
# 
# # Use the best tuning parameters for the model
# final_penalized_mnm_wf <- penalized_mnm_wf %>%
#   finalize_workflow(penalized_mnm_best_tune) %>%
#   fit(data = otto_train)
# 
# # Predictions
# penalized_mnm_preds <- final_penalized_mnm_wf %>%
#   predict(new_data = otto_test, type = "prob")
# 
# # Create the submission
# penalized_mnm_submission <- data.frame("id" = otto_test$id,
#                                        "Class_1" = penalized_mnm_preds$.pred_Class_1,
#                                        "Class_2" = penalized_mnm_preds$.pred_Class_2,
#                                        "Class_3" = penalized_mnm_preds$.pred_Class_3,
#                                        "Class_4" = penalized_mnm_preds$.pred_Class_4,
#                                        "Class_5" = penalized_mnm_preds$.pred_Class_5,
#                                        "Class_6" = penalized_mnm_preds$.pred_Class_6,
#                                        "Class_7" = penalized_mnm_preds$.pred_Class_7,
#                                        "Class_8" = penalized_mnm_preds$.pred_Class_8,
#                                        "Class_9" = penalized_mnm_preds$.pred_Class_9)
# penalized_mnm_submission
# vroom_write(penalized_mnm_submission, "submission.csv", delim = ",")

# # Penalized Multinomial Regression Model (Upsampling)-----------------------
# 
# # Update the recipe to use the partially downsampled data
# penalized_mnm_upsample_recipe <- recipe(target ~ ., data = medium_otto_train) %>% 
#   update_role(id, new_role = "Id") %>% 
#   # id will stay in the dataset but won't be used as a predictor
#   step_mutate_at(all_outcomes(), fn = factor, skip = TRUE) %>% 
#   step_normalize(all_numeric_predictors()) %>% 
#   step_upsample(all_outcomes())
# 
# # Set the model
# penalized_mnm_mod <- multinom_reg(mixture = tune(), penalty = tune()) %>%
#   set_engine("glmnet")
# 
# # Set the workflow
# penalized_mnm_wf <- workflow() %>%
#   add_recipe(penalized_mnm_upsample_recipe) %>%
#   add_model(penalized_mnm_mod)
# 
# # Set the tuning grid
# penalized_mnm_tuning_grid <- grid_regular(penalty(),
#                                           mixture(),
#                                           levels = 5)
# 
# # Set up the CV
# penalized_mnm_folds <- vfold_cv(medium_otto_train, v = 10, repeats = 1)
# 
# # Run the CV
# penalized_mnm_CV_results <- penalized_mnm_wf %>%
#   tune_grid(resamples = penalized_mnm_folds,
#             grid = penalized_mnm_tuning_grid,
#             metrics = metric_set(roc_auc)) #, f_meas, sens, recall, spec,
# # precision, accuracy))
# 
# # Find out the best tuning parameters
# penalized_mnm_best_tune <- penalized_mnm_CV_results %>% select_best("roc_auc")
# penalized_mnm_best_tune
# 
# # Use the best tuning parameters for the model
# final_penalized_mnm_wf <- penalized_mnm_wf %>%
#   finalize_workflow(penalized_mnm_best_tune) %>%
#   fit(data = otto_train)
# 
# # Predictions
# penalized_mnm_preds <- final_penalized_mnm_wf %>%
#   predict(new_data = otto_test, type = "prob")
# 
# # Create the submission
# penalized_mnm_submission <- data.frame("id" = otto_test$id,
#                                        "Class_1" = penalized_mnm_preds$.pred_Class_1,
#                                        "Class_2" = penalized_mnm_preds$.pred_Class_2,
#                                        "Class_3" = penalized_mnm_preds$.pred_Class_3,
#                                        "Class_4" = penalized_mnm_preds$.pred_Class_4,
#                                        "Class_5" = penalized_mnm_preds$.pred_Class_5,
#                                        "Class_6" = penalized_mnm_preds$.pred_Class_6,
#                                        "Class_7" = penalized_mnm_preds$.pred_Class_7,
#                                        "Class_8" = penalized_mnm_preds$.pred_Class_8,
#                                        "Class_9" = penalized_mnm_preds$.pred_Class_9)
# penalized_mnm_submission
# vroom_write(penalized_mnm_submission, "submission.csv", delim = ",")

# # Penalized Multinomial Regression Model (Manual Downsampling)-----------------------
# # Create small dataset to test models more quickly
# sample_size <- 350
# num_classes <- 9
# set.seed(12)
# for (i in 1:num_classes) {
#   if (i == 1) {
#     # Filter to just one class
#     small_otto_train <- otto_train %>% filter(target == paste0("Class_", i)) %>%
#       # then take a sample from that class
#       .[sample(1:length(.$feat_1), size = sample_size),]
#   } else {
#     # Filter to just one class
#     small_otto_train <- otto_train %>% filter(target == paste0("Class_", i)) %>%
#       # then take a sample from that class
#       .[sample(1:length(.$feat_1), size = sample_size),] %>%
#       # then add this sample to the existing samples
#       bind_rows(small_otto_train, .)
#   }
# }
# # small_otto_train
# 
# penalized_mnm_recipe <- recipe(target ~ ., data = small_otto_train) %>%
#   update_role(id, new_role = "Id") %>%
#   # id will stay in the dataset but won't be used as a predictor
#   step_mutate_at(all_outcomes(), fn = factor, skip = TRUE) %>%
#   step_normalize(all_numeric_predictors()) # %>%
# # step_upsample(all_outcomes())
# 
# penalized_mnm_mod <- multinom_reg(mixture = tune(), penalty = tune()) %>%
#   set_engine("glmnet")
# 
# penalized_mnm_wf <- workflow() %>%
#   add_recipe(penalized_mnm_recipe) %>%
#   add_model(penalized_mnm_mod)
# 
# # Set the tuning grid
# penalized_mnm_tuning_grid <- grid_regular(penalty(),
#                                           mixture(),
#                                           levels = 8)
# 
# # Set up the CV
# penalized_mnm_folds <- vfold_cv(small_otto_train, v = 8, repeats = 1)
# 
# # Run the CV
# penalized_mnm_CV_results <- penalized_mnm_wf %>%
#   tune_grid(resamples = penalized_mnm_folds,
#             grid = penalized_mnm_tuning_grid,
#             metrics = metric_set(roc_auc)) #, f_meas, sens, recall, spec,
# # precision, accuracy))
# 
# # Find out the best tuning parameters
# penalized_mnm_best_tune <- penalized_mnm_CV_results %>% select_best("roc_auc")
# penalized_mnm_best_tune
# 
# # Use the best tuning parameters for the model
# final_penalized_mnm_wf <- penalized_mnm_wf %>%
#   finalize_workflow(penalized_mnm_best_tune) %>%
#   fit(data = otto_train)
# 
# # Predictions
# penalized_mnm_preds <- final_penalized_mnm_wf %>%
#   predict(new_data = otto_test, type = "prob")
# 
# # Create the submission
# penalized_mnm_submission <- data.frame("id" = otto_test$id,
#                                        "Class_1" = penalized_mnm_preds$.pred_Class_1,
#                                        "Class_2" = penalized_mnm_preds$.pred_Class_2,
#                                        "Class_3" = penalized_mnm_preds$.pred_Class_3,
#                                        "Class_4" = penalized_mnm_preds$.pred_Class_4,
#                                        "Class_5" = penalized_mnm_preds$.pred_Class_5,
#                                        "Class_6" = penalized_mnm_preds$.pred_Class_6,
#                                        "Class_7" = penalized_mnm_preds$.pred_Class_7,
#                                        "Class_8" = penalized_mnm_preds$.pred_Class_8,
#                                        "Class_9" = penalized_mnm_preds$.pred_Class_9)
# penalized_mnm_submission
# vroom_write(penalized_mnm_submission, "submission.csv", delim = ",")

# # Penalized Multinomial Regression Model (SMOTE)-----------------------
# 
# # Set the model
# penalized_mnm_mod <- multinom_reg(mixture = tune(), penalty = tune()) %>%
#   set_engine("glmnet")
# 
# # Set the workflow
# penalized_mnm_wf <- workflow() %>%
#   add_recipe(penalized_mnm_smote_recipe) %>%
#   add_model(penalized_mnm_mod)
# 
# # Set the tuning grid
# penalized_mnm_tuning_grid <- grid_regular(penalty(),
#                                           mixture(),
#                                           levels = 8)
# 
# # Set up the CV
# penalized_mnm_folds <- vfold_cv(otto_train, v = 10, repeats = 1)
# 
# # Run the CV
# penalized_mnm_CV_results <- penalized_mnm_wf %>%
#   tune_grid(resamples = penalized_mnm_folds,
#             grid = penalized_mnm_tuning_grid,
#             metrics = metric_set(roc_auc)) #, f_meas, sens, recall, spec,
# # precision, accuracy))
# 
# # Find out the best tuning parameters
# penalized_mnm_best_tune <- penalized_mnm_CV_results %>% select_best("roc_auc")
# penalized_mnm_best_tune
# 
# # Use the best tuning parameters for the model
# final_penalized_mnm_wf <- penalized_mnm_wf %>%
#   finalize_workflow(penalized_mnm_best_tune) %>%
#   fit(data = otto_train)
# 
# # Predictions
# penalized_mnm_preds <- final_penalized_mnm_wf %>%
#   predict(new_data = otto_test, type = "prob")
# 
# # Create the submission
# penalized_mnm_submission <- data.frame("id" = otto_test$id,
#                                        "Class_1" = penalized_mnm_preds$.pred_Class_1,
#                                        "Class_2" = penalized_mnm_preds$.pred_Class_2,
#                                        "Class_3" = penalized_mnm_preds$.pred_Class_3,
#                                        "Class_4" = penalized_mnm_preds$.pred_Class_4,
#                                        "Class_5" = penalized_mnm_preds$.pred_Class_5,
#                                        "Class_6" = penalized_mnm_preds$.pred_Class_6,
#                                        "Class_7" = penalized_mnm_preds$.pred_Class_7,
#                                        "Class_8" = penalized_mnm_preds$.pred_Class_8,
#                                        "Class_9" = penalized_mnm_preds$.pred_Class_9)
# penalized_mnm_submission
# vroom_write(penalized_mnm_submission, "submission.csv", delim = ",")

# #  Manual Downsampling Forest ---------------------------------
# 
# # Create small dataset
# sample_size <- 500
# num_classes <- 9
# set.seed(12)
# for (i in 1:num_classes) {
#   if (i == 1) {
#     # Filter to just one class
#     small_otto_train <- otto_train %>% filter(target == paste0("Class_", i)) %>%
#       # then take a sample from that class
#       .[sample(1:length(.$feat_1), size = sample_size),]
#   } else {
#     # Filter to just one class
#     small_otto_train <- otto_train %>% filter(target == paste0("Class_", i)) %>%
#       # then take a sample from that class
#       .[sample(1:length(.$feat_1), size = sample_size),] %>%
#       # then add this sample to the existing samples
#       bind_rows(small_otto_train, .)
#   }
# }
# # small_otto_train
# 
# # Update recipe to use smaller dataset
# tree_downsample_recipe <- recipe(target ~ ., data = small_otto_train) %>%
#   update_role(id, new_role = "Id") %>% 
#   # id will stay in the dataset but won't be used as a predictor
#   step_mutate_at(all_outcomes(), fn = factor, skip = TRUE) %>% 
#   step_normalize(all_numeric_predictors())
# 
# otto_forest_downsample_model <- rand_forest(mtry = tune(),
#                                             min_n = tune(),
#                                             trees = 1000) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# # Create a workflow using the model and recipe
# otto_forest_downsample_wf <- workflow() %>%
#   add_model(otto_forest_downsample_model) %>%
#   add_recipe(tree_downsample_recipe)
# 
# # Set up the grid with the tuning values
# num_non_predictors <- 2
# otto_forest_downsample_grid <- grid_regular(mtry(range = c(1, (length(otto_train)-num_non_predictors))),
#                                             min_n(), levels = 5)
# 
# # Set up the K-fold CV
# otto_forest_downsample_folds <- vfold_cv(data = small_otto_train, v = 5, repeats = 1)
# 
# # Find best tuning parameters
# forest_cv_results <- otto_forest_downsample_wf %>%
#   tune_grid(resamples = otto_forest_downsample_folds,
#             grid = otto_forest_downsample_grid,
#             metrics = metric_set(roc_auc))
# best_forest_tune <- forest_cv_results %>% select_best("roc_auc")
# 
# # Use the best tuning parameters for the model
# forest_final_wf <- otto_forest_downsample_wf %>%
#   finalize_workflow(best_forest_tune) %>%
#   fit(data = otto_train)
# 
# otto_forest_downsample_predictions <- predict(forest_final_wf,
#                                               new_data = otto_test,
#                                               type = "prob")
# otto_forest_downsample_predictions
# 
# # Create the submission
# otto_forest_downsample_submission <- data.frame("id" = otto_test$id,
#                                                 "Class_1" = otto_forest_downsample_predictions$.pred_Class_1,
#                                                 "Class_2" = otto_forest_downsample_predictions$.pred_Class_2,
#                                                 "Class_3" = otto_forest_downsample_predictions$.pred_Class_3,
#                                                 "Class_4" = otto_forest_downsample_predictions$.pred_Class_4,
#                                                 "Class_5" = otto_forest_downsample_predictions$.pred_Class_5,
#                                                 "Class_6" = otto_forest_downsample_predictions$.pred_Class_6,
#                                                 "Class_7" = otto_forest_downsample_predictions$.pred_Class_7,
#                                                 "Class_8" = otto_forest_downsample_predictions$.pred_Class_8,
#                                                 "Class_9" = otto_forest_downsample_predictions$.pred_Class_9)
# otto_forest_downsample_submission
# vroom_write(otto_forest_downsample_submission, "submission.csv", delim = ",")

# Boosted Trees (Manual Downsampling) -----------------------------
# Manual Downsampling
sample_size <- 500
num_classes <- 9
set.seed(12)
for (i in 1:num_classes) {
  if (i == 1) {
    # Filter to just one class
    small_otto_train <- otto_train %>% filter(target == paste0("Class_", i)) %>%
      # then take a sample from that class
      .[sample(1:length(.$feat_1), size = sample_size),]
  } else {
    # Filter to just one class
    small_otto_train <- otto_train %>% filter(target == paste0("Class_", i)) %>%
      # then take a sample from that class
      .[sample(1:length(.$feat_1), size = sample_size),] %>%
      # then add this sample to the existing samples
      bind_rows(small_otto_train, .)
  }
}
# small_otto_train

# Update recipe to use smaller dataset
boosted_recipe <- recipe(target ~ ., data = small_otto_train) %>%
  update_role(id, new_role = "Id") %>%
  # id will stay in the dataset but won't be used as a predictor
  step_mutate_at(all_outcomes(), fn = factor, skip = TRUE) %>%
  step_normalize(all_numeric_predictors())


# Set the model
boosted_model <- boost_tree(tree_depth = tune(),
                            trees = 750,
                            learn_rate = tune()) %>%
  set_mode("classification") %>%
  set_engine("lightgbm")

# Set workflow
boosted_wf <- workflow() %>%
  add_recipe(boosted_recipe) %>%
  add_model(boosted_model)

# Tuning
# Set up the grid with the tuning values
boosted_grid <- grid_regular(tree_depth(), learn_rate(), levels = 7)

# Set up the K-fold CV
boosted_folds <- vfold_cv(data = small_otto_train, v = 5, repeats = 1)

# Find best tuning parameters
boosted_cv_results <- boosted_wf %>%
  tune_grid(resamples = boosted_folds,
            grid = boosted_grid,
            metrics = metric_set(roc_auc))

# Select best tuning parameters
boosted_best_tune <- boosted_cv_results %>% select_best("roc_auc")
boosted_final_wf <- boosted_wf %>%
  finalize_workflow(boosted_best_tune) %>%
  fit(data = otto_train)
boosted_best_tune

# Make predictions
boosted_predictions <- predict(boosted_final_wf,
                               new_data = otto_test,
                               type = "prob")
boosted_predictions

# Create the submission
boosted_submission <- data.frame("id" = otto_test$id,
                                 "Class_1" = boosted_predictions$.pred_Class_1,
                                 "Class_2" = boosted_predictions$.pred_Class_2,
                                 "Class_3" = boosted_predictions$.pred_Class_3,
                                 "Class_4" = boosted_predictions$.pred_Class_4,
                                 "Class_5" = boosted_predictions$.pred_Class_5,
                                 "Class_6" = boosted_predictions$.pred_Class_6,
                                 "Class_7" = boosted_predictions$.pred_Class_7,
                                 "Class_8" = boosted_predictions$.pred_Class_8,
                                 "Class_9" = boosted_predictions$.pred_Class_9)
boosted_submission
vroom_write(boosted_submission, "submission.csv", delim = ",")

# # BART (Manual Downsampling) ----------------------------

# The BART model didn't end up working because it isn't configured to handle 
# multinomial response data. I would have liked to find how to implement 
# a multinomial BART model, but I didn't have enough time to do so.

# # Manual Downsampling
# sample_size <- 500
# num_classes <- 9
# set.seed(12)
# for (i in 1:num_classes) {
#   if (i == 1) {
#     # Filter to just one class
#     small_otto_train <- otto_train %>% filter(target == paste0("Class_", i)) %>%
#       # then take a sample from that class
#       .[sample(1:length(.$feat_1), size = sample_size),]
#   } else {
#     # Filter to just one class
#     small_otto_train <- otto_train %>% filter(target == paste0("Class_", i)) %>%
#       # then take a sample from that class
#       .[sample(1:length(.$feat_1), size = sample_size),] %>%
#       # then add this sample to the existing samples
#       bind_rows(small_otto_train, .)
#   }
# }
# # small_otto_train
# 
# # Update recipe to use smaller dataset
# bart_recipe <- recipe(target ~ ., data = small_otto_train) %>%
#   update_role(id, new_role = "Id") %>% 
#   # id will stay in the dataset but won't be used as a predictor
#   step_mutate_at(all_outcomes(), fn = factor, skip = TRUE) %>% 
#   step_normalize(all_numeric_predictors())
# 
# # Set the model
# bart_model <- parsnip::bart(trees = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("dbarts")
# 
# # Set workflow
# bart_wf <- workflow() %>%
#   add_recipe(bart_recipe) %>%
#   add_model(bart_model)
# 
# # Tuning
# # Set up the grid with the tuning values
# bart_grid <- grid_regular(trees(), levels = 8)
# 
# # Set up the K-fold CV
# bart_folds <- vfold_cv(data = small_otto_train, v = 5, repeats = 1)
# 
# # Find best tuning parameters
# bart_cv_results <- bart_wf %>%
#   tune_grid(resamples = bart_folds,
#             grid = bart_grid,
#             metrics = metric_set(roc))
# 
# # Select best tuning parameters
# bart_best_tune <- bart_cv_results %>% select_best("roc_auc")
# bart_final_wf <- bart_wf %>%
#   finalize_workflow(bart_best_tune) %>%
#   fit(data = otto_train)
# bart_best_tune
# 
# # Make predictions
# bart_predictions <- predict(bart_final_wf, 
#                             new_data = otto_test, 
#                             type = "prob")
# bart_predictions
# 
# # Create the submission
# bart_submission <- data.frame("id" = otto_test$id,
#                               "Class_1" = bart_predictions$.pred_Class_1,
#                               "Class_2" = bart_predictions$.pred_Class_2,
#                               "Class_3" = bart_predictions$.pred_Class_3,
#                               "Class_4" = bart_predictions$.pred_Class_4,
#                               "Class_5" = bart_predictions$.pred_Class_5,
#                               "Class_6" = bart_predictions$.pred_Class_6,
#                               "Class_7" = bart_predictions$.pred_Class_7,
#                               "Class_8" = bart_predictions$.pred_Class_8,
#                               "Class_9" = bart_predictions$.pred_Class_9)
# bart_submission
# vroom_write(bart_submission, "submission.csv", delim = ",")