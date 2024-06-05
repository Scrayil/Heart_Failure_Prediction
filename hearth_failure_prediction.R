# PROJECT DESCRIPTION:
# This project aims to predict heart failure outcomes by applying statistical
# learning algorithms. The goal is to improve prediction accuracy through the
# SuperLearner algorithm.
# More specifically, the project involves data preprocessing, visualization of 
# outliers, model training, and evaluation using cross-validation.
# The performance of individual models, such as Random Forest, GLMNet, and XGBoost,
# is assessed, and their variable importance is analyzed.
# The combined model's accuracy is evaluated and compared to the base models.


# CLEARING THE CURRENT ENVIRONMENT
# Removing environment variables
rm(list = ls())
# Deleting all plots
while (!is.null(dev.list())) dev.off()
# Cleaning the console
cat("\014")

# LOADING PACKAGES
library(naniar)
library(tidyr)
library(SuperLearner)
library(caret)
library(doParallel)
library(doRNG)
library(randomForest)
library(glmnet)
library(xgboost)
library(ggplot2)
library(pROC)
library(arm)        # For bayesglm
library(polspline)  # For polymars
library(e1071)      # For SVM
library(nnet)       # For neural networks
library(ranger)     # For ranger


#########################
# FUNCTION DECLARATIONS #
#########################
# Function used to create colored boxplots and visualize eventual outliers
visualize_outliers <- function(df, exclude_vars = NULL) {
  df_long <- pivot_longer(df, cols = -all_of(exclude_vars), names_to = "variables", values_to = "value")
  ggplot(df_long, aes(x = variables, y = value, fill = variables)) +
    geom_boxplot(width = 0.7, alpha = 0.5) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    labs(title = paste0("Boxplots", if(!is.null(exclude_vars)) paste0(" without ", exclude_vars) else ""), x = "Variables", y = "Values")
}

# Function used to extract the variable importance for random forest models
extract_importance_randomForest <- function(model) {
  importance_df <- as.data.frame(randomForest::importance(model))
  importance_df$Variable <- rownames(importance_df)
  colnames(importance_df) <- c("Importance", "Variable")
  return(importance_df)
}

# Function used to extract the variable importance for GLMNet models
extract_importance_glmnet <- function(model) {
  coef_matrix <- as.matrix(coef(model, s = "lambda.min"))
  coef_df <- data.frame(Variable = rownames(coef_matrix), Importance = abs(coef_matrix[,1]))
  # Removing the intercept
  coef_df <- coef_df[coef_df$Variable != "(Intercept)", ]
  return(coef_df)
}

# Function used to extract the variable importance for XGBoost models
extract_importance_xgboost <- function(model) {
  importance_matrix <- xgb.importance(model = model)
  importance_matrix <- importance_matrix[, c("Feature", "Gain")]
  colnames(importance_matrix) <- c("Variable", "Importance")
  return(importance_matrix)
}

# Function used to normalize the variable importance
normalize_importance <- function(importance) {
  importance / sum(importance)
}

# Function to extract the variable importance for each model
extract_importance <- function(model_name, model) {
  importance_df <- switch(model_name,
                          "SL.randomForest" = extract_importance_randomForest(model),
                          "SL.glmnet" = extract_importance_glmnet(model),
                          "SL.xgboost" = extract_importance_xgboost(model),
                          NULL)
  
  if (!is.null(importance_df)) {
    importance_df$Model <- model_name
  }
  
  return(importance_df)
}

# Function used to plot any model's variable importance
plot_variable_importance <- function(variable_importance, model_name) {
  variable_importance$Variable <- factor(variable_importance$Variable, levels = variable_importance$Variable[order(variable_importance$Importance)])
  
  ggplot(variable_importance, aes(x = Variable, y = Importance)) +
    geom_bar(stat = "identity", fill = if(model_name != "Combined") "#002d72" else "#077300", width = 0.5) +
    coord_flip() +
    labs(title = sprintf("%s Variable Importance", model_name), x = "Variables", y = "Importance") +
    theme_bw() + 
    # Removes the bars' padding to the left
    scale_y_continuous(expand = expansion(mult = c(0, 0.05)))
}

# Function to calculate accuracy for individual models
calculate_accuracy <- function(predictions, Y_test) {
  if (length(unique(Y_test)) != 2) {
    stop("Target variable Y_test is not binary.")
  }
  
  pred_factor <- factor(ifelse(predictions > 0.5, 1, 0), levels = c(0, 1))
  Y_test_factor <- factor(Y_test, levels = c(0, 1))
  accuracy <- sum(pred_factor == Y_test_factor) / length(predictions)
  return(accuracy)
}


#######################
# LOADING THE DATASET #
#######################
# Set for reproducibility
set.seed(123)

hfcr <- read.csv("data/dataset/heart_failure_clinical_records.csv")
if (!exists("hfcr")) {
  stop("Dataset not found!")
}
# Printing the features' details
str(hfcr)

# Inspecting eventual missing values
print(gg_miss_var(hfcr) + theme_bw())

# Looking for eventual outliers
print(visualize_outliers(hfcr))
print(visualize_outliers(hfcr, exclude_vars = c("platelets", "creatinine_phosphokinase")))

# The SuperLearner requires DEATH_EVENT to be a numeric variable
hfcr$DEATH_EVENT <- as.numeric(hfcr$DEATH_EVENT)

# Splitting the dataset by using 70% of the records as train set
trainIndex <- createDataPartition(hfcr$DEATH_EVENT, p = 0.7, list = FALSE)
train_data <- hfcr[trainIndex, ]
test_data <- hfcr[-trainIndex, ]

# Separating the predictive variables from the target
X_train <- subset(train_data, select = -DEATH_EVENT)
Y_train <- train_data$DEATH_EVENT
X_test <- subset(test_data, select = -DEATH_EVENT)
Y_test <- test_data$DEATH_EVENT

# Prints the available models for the SuperLearner
listWrappers()
# Defining basic models to train
learners <- c(
  "SL.randomForest", "SL.glmnet", "SL.xgboost", "SL.glm",
  "SL.svm", "SL.nnet", "SL.bayesglm", "SL.polymars", "SL.ranger"
)


##################
# MODEL TRAINING #
##################
MODELS_FOLDER = "models"
LEARNER_PATH = paste0(MODELS_FOLDER, "/super_learner.rds")

# The training phase is skipped if a trained model is available
if (!file.exists(LEARNER_PATH)) {
  # Creates the models' directory if missing
  if(!dir.exists(MODELS_FOLDER)) {
    dir.create(MODELS_FOLDER)
  }
  
  # cross-validation
  cv_control <- SuperLearner.CV.control(V = 10)  # 10-fold cross-validation
  
  # Setting up parallel processing
  cluster <- makeCluster(detectCores() - 1)  # use all cores except one
  registerDoParallel(cluster)
  registerDoRNG(seed = 123)  # Ensure reproducibility
  clusterEvalQ(cluster, {
    library(SuperLearner)
    library(randomForest)
    library(glmnet)
    library(xgboost)
    library(arm)
    library(polspline)
    library(e1071)
    library(nnet)
    library(ranger)
  })
  
  # Executing the cross-validation phases in parallel thanks to parLapply
  clusterExport(cluster, c("Y_train", "X_train", "learners", "cv_control"))
  super_learner <- SuperLearner(Y = Y_train, X = X_train, family = binomial(), SL.library = learners, method = "method.NNLS", cvControl = cv_control, verbose = TRUE)
  
  # Saving the SuperLearner object at the given path
  saveRDS(super_learner, file=LEARNER_PATH)
  
  # Stopping the cluster
  stopCluster(cluster)
  registerDoSEQ()  # Return to sequential execution
}


####################
# MODEL EVALUATION #
####################
# Loading the pre-existing model (or newly saved)
super_learner <- readRDS(LEARNER_PATH)
# Ensuring the model has been loaded correctly
if (is.null(super_learner)) {
  stop("Failed loading the SuperLearner. No existing model found!")
} else {
  print("SuperLearner loaded successfully!")
}
# Showing the model's summary
print(summary(super_learner))

# Printing each model summary
lapply(super_learner$fitLibrary, function(model) {
  print(summary(model$object))
})

# Making predictions on the test set
predictions <- predict(super_learner, newdata = X_test)$pred

# Checking predictions for the Super Learner
if (any(predictions < 0 | predictions > 1)) {
  stop("Super Learner predictions contain values outside the range [0, 1]")
}

# Computing the ROC and AUC indexes
roc_curve <- roc(Y_test, predictions)
auc_value <- auc(roc_curve)

# ROC curve visualization
plot.roc(roc_curve, main = paste("ROC Curve (AUC =", round(auc_value, 2), ")")) + theme_bw()

# Evaluating the overall accuracy of the Super Learner
super_learner_accuracy <- calculate_accuracy(predictions, Y_test)

# Extracting predictions for each base model directly from the SuperLearner object
base_model_predictions <- super_learner$library.predict

# Checking base model predictions range
if (any(base_model_predictions < 0 | base_model_predictions > 1)) {
  stop("Base model predictions contain values outside the range [0, 1]")
}

# Calculating accuracy for each base model
base_model_accuracies <- apply(base_model_predictions, 2, function(preds) {
  accuracy <- calculate_accuracy(preds, Y_test)
  print(paste("Model accuracy:", accuracy))
  return(accuracy)
})

# Creating a dataframe to store accuracies
accuracy_df <- data.frame(
  Model = c(learners, "Super Learner"),
  Accuracy = c(base_model_accuracies, super_learner_accuracy)
)

# Removing NA values
accuracy_df <- accuracy_df[complete.cases(accuracy_df), ]

# Plotting the accuracies
print(ggplot(accuracy_df, aes(x = reorder(Model, Accuracy), y = Accuracy)) +
        geom_bar(stat = "identity", fill = "#730044", width = 0.4) +
        coord_flip() +
        labs(title = "Model Accuracies", x = "Models", y = "Accuracy") +
        theme_bw() +
        # Removes the bars' padding to the left
        scale_y_continuous(expand = expansion(mult = c(0, 0.05)))
)

# Extracting the variable importance for each model
importances <- lapply(learners, function(learner) {
  model <- super_learner$fitLibrary[[paste0(learner, "_All")]]$object
  if (!is.null(model)) {
    return(extract_importance(learner, model))
  } else {
    return(NULL)
  }
})

# Plotting individual variable importances
invisible(lapply(seq_along(importances), function(i) {
  imp <- importances[[i]]
  if (!is.null(imp)) {
    print(plot_variable_importance(imp, learners[i]))
  }
}))

# Combining the importances
valid_importances <- importances[!sapply(importances, is.null)]
normalized_importances <- lapply(valid_importances, function(imp_df) {
  imp_df$Importance <- normalize_importance(imp_df$Importance)
  return(imp_df)
})
combined_importance <- do.call(rbind, normalized_importances)

# Aggregating by Variable and calculating MeanImportance
combined_importance_agg <- aggregate(Importance ~ Variable, data = combined_importance, FUN = mean)

# Plotting combined importance
print(plot_variable_importance(combined_importance_agg, "Combined"))