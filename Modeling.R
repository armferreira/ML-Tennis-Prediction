# Load required libraries
library(dplyr) # Data manipulation and transformation.
library(car) # Functions for applied regression, including various diagnostic plots for regression models, outlier tests.
library(ggplot2) # Data visualization library.
library(ggeffects) # Regression visualization library.
library(caret) # Classification and Regression Training
## devtools::install_github("dongyuanwu/RSBID")
library(RSBID) # Provides SMOTE-NC method for over-sampling.
library(pROC) # Tools for ROC analysis. Provides functions for plotting ROC curves and computing AUC (Area Under the Curve).
library(partykit) # Allows the creation and visualization of tree-based models.
library(rpart) # Recursive Partitioning and Regression Trees.
library(rpart.plot) # Plots decision trees created with 'rpart'.
library(fastDummies) # Quickly create dummy variables from categorical data.
library(neuralnet) # Training of neural networks using backpropagation.
library(keras) # High-level neural networks API, written in R and based on TensorFlow.
library(tensorflow) # Interface to the TensorFlow deep learning library.
library(h2o) # Provides a platform for building with AutoML.

# !Usar dataset atp após limpeza com Cleaning.R

############## 0. Data Pre-processing ##############
atp <- read.csv("G:/O meu disco/MS Data Science/Trabalho MTCD&MP/Scripts e datasets/atp.csv", row.names=1, stringsAsFactors = T)

# Setting seed to make results reproducible
set.seed(2023)

# Dividing between best of 5 and best of 3 matches (then removing the variable)
atp5 <- atp[atp$best_of == 5, ] %>% subset(select = -best_of)
atp3 <- atp[atp$best_of == 3, ] %>% subset(select = -best_of)

# Removing matches with less than 3 sets for 'bo5', and less of 2 for 'bo3'
atp5 <- atp5 %>%  filter(no_sets %in% c('X3', 'X4', 'X5')) %>%  droplevels()
atp3 <- atp3 %>%  filter(no_sets %in% c('X2', 'X3')) %>%  droplevels()

# Dividing between train and test sets
index <- sample(1:nrow(atp5), 2/3* nrow(atp5))
atp5.train <- atp5[index,]
atp5.test <- atp5[-index,]

index <- sample(1:nrow(atp3), 2/3* nrow(atp3))
atp3.train <- atp3[index,]
atp3.test <- atp3[-index,]

# Over-sampling no_sets with SMOTE-NC
atp3.train.over <- SMOTE_NC(atp3.train, 'no_sets')

# Under-sampling no_sets with random down-sampling
atp3.train.under <- downSample(atp3.train[,-19], atp3.train[,19])
names(atp3.train.under)[19] <- "no_sets"

atp5.train.under <- downSample(atp5.train[,-19], atp5.train[,19])
names(atp5.train.under)[19] <- "no_sets"

############## 1. Logistic Regression ##############
atp3.logreg <- glm(no_sets~ ., data = atp3.train, family = "binomial")
summary(atp3.logreg)

# Confusion Matrix
pred <- predict(atp3.logreg, atp3.test, type = 'response')
pred <- ifelse(pred > 0.50, 3,2)
pred <- factor(pred, labels = c("X2", "X3"))
confusionMatrix(pred, atp3.test$no_sets)

# Testing for multicollinearity
vif(atp3.logreg)

# Picking best model with AIC
stepAIC(atp3.logreg, direction="both")

# Re-modeling according to AIC
atp3.logreg.aic <- glm(no_sets ~ p1_rank + p2_rank + p1_age + p1_height + 
                         p2_height,
                       data = atp3.train, family = "binomial")
summary(atp3.logreg.aic)
pred <- predict(atp3.logreg.aic, atp3.test, type = 'response')
pred <- ifelse(pred > 0.50, 3,2)
pred <- factor(pred, labels = c("X2", "X2"))
confusionMatrix(pred, atp3.test$no_sets)

# Plotting curves
predictions <- ggpredict(atp3.logreg.aic, c("p1_rank"))
ggplot(predictions, aes(x = x, y = predicted)) +
  geom_line(color = "darkred") +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.2) +  # Confidence interval
  labs(x = "p2_height", y = "Predicted Probability")

# Re-modelling according to AIC and with re-sampling
atp3.logreg.over <- glm(no_sets ~ p1_rank + p2_rank + p1_age + p1_height + p2_height, data = atp3.train.over, family = "binomial")
summary(atp3.logreg.over)

atp3.logreg.under <- glm(no_sets ~ p1_rank + p2_rank + p1_age + p1_height + p2_height, data = atp3.train.under, family = "binomial")
summary(atp3.logreg.under)

# Testing improved models
pred.over <- predict(atp3.logreg.over, atp3.test, type = 'response')
pred.over <- ifelse(pred.over > 0.50,3,2)
pred.over <- factor(pred.over, labels = c("X2", "X3"))
confusionMatrix(pred.over, atp3.test$no_sets)

pred.under <- predict(atp3.logreg.under, atp3.test, type = 'response')
pred.under <- ifelse(pred.under > 0.50,3,2)
pred.under <- factor(pred.under, labels = c("X2", "X3"))
confusionMatrix(pred.under, atp3.test$no_sets)

# Plotting ROC curves
roc <- roc(atp3.test$no_sets, as.numeric(pred) - 1)
roc.over <- roc(atp3.test$no_sets, as.numeric(pred.over) - 1)
roc.under <- roc(atp3.test$no_sets, as.numeric(pred.under) - 1)
roc
g3 <- ggroc(list(step_aic = roc, over_sampling = roc.over, under_sampling = roc.under))
g3
# AUC calculation
auc(roc)
auc(roc.over)
auc(roc.under)

############## 2. Decision Trees ##############

#### 2.1 Decision trees with RPART 10-fold Cross-Validation
atp3.tree <- rpart(no_sets ~ p1_rank + p2_rank + p1_age + p1_height + p2_height, data = atp3.train.over, 
                   method="class", 
                   control = rpart.control(xval = 10, minsplit = 500, cp= 0.01))
# Summary of Decision Tree
atp3.tree.party <- as.party(atp3.tree)
atp3.tree.party

# Decision tree visualization
rpart.rules(atp3.tree)
rpart.plot(atp3.tree)
printcp(atp3.tree)
plotcp(atp3.tree)
# atp3.train.over%>%
#   ggplot(aes(x=p2_rank, y= p1_age)) +
#   geom_jitter(aes(col=no_sets), alpha=0.7) +
#   geom_parttree(data = atp3.tree, aes(fill=no_sets), alpha = 0.1)

# Measuring decision tree accuracy
atp3.tree.predict <- predict(atp3.tree, newdat = atp3.train.under, type="class")
confusionMatrix(atp3.tree.predict, atp3.train.under$no_sets)

# OPTIONAL: Decision tree with Caret's train() - 10-fold Cross-Validation and tunelength)
cv.control <- trainControl(method="cv",number=10, classProbs = TRUE, savePredictions="final")
atp3.tree2 <- caret::train(no_sets ~ p1_rank + p2_rank + p1_age + p1_height + p2_height, data=atp3.train.under, method="rpart", tuneLength=10, trControl=cv.control)
rpart.plot(atp3.tree2$finalModel, box.palette=c("lightcoral", "#98bfad"))
plot(atp3.tree2)
atp3.tree$cptable
pruned_tree <- prune(atp3.tree2$finalModel, cp = 0.009631491)
rpart.plot(pruned_tree, box.palette=c("lightcoral", "#98bfad"))
atp3.tree2.predict <- predict(atp3.tree2, newdat = atp3.test, type="raw")
confusionMatrix(atp3.tree2.predict, atp3.test$no_sets)

var_imp <- varImp(atp3.tree2)
top_variables <- head(var_imp$importance, 10)
ggplot(top_variables, aes(x = Overall, y = reorder(rownames(top_variables), Overall))) +
  geom_bar(stat = "identity", fill = "lightcoral") +
  labs(x = "Importance", y = "Features")

#### 2.2 Decision Tree with Bagging (nbagg = 100)
cv.control <- trainControl(method="cv", number=10, classProbs = TRUE, savePredictions="final")
atp3.bag <- caret::train(no_sets ~ p1_rank, data=atp5.train, method="treebag", metric='Accuracy', 
                         nbagg=100, tuneLength=5,
                          trControl=cv.control)
var_imp <- varImp(atp3.bag)
top_variables <- head(var_imp$importance, 10)
ggplot(top_variables, aes(x = Overall, y = reorder(rownames(top_variables), Overall))) +
  geom_bar(stat = "identity", fill = "lightcoral") +
  labs(x = "Importance", y = "Features")

atp3.bag.predict <- predict(atp3.bag, newdat = atp5.train, type="raw")
confusionMatrix(atp3.bag.predict, atp5.train$no_sets)

#### 2.3 Random Forest
cv.control <- trainControl(method="cv",number=10, classProbs = TRUE, savePredictions="final")
tune_forest <- expand.grid(mtry =c(1), # number of features in each tree (should be p/3 in classification)
                           splitrule="gini", 
                           min.node.size=c(4,5,7,9,10))
atp3.forest <- caret::train(no_sets ~ p1_rank, data=atp5.train.under, 
                      method="ranger", 
                      num.trees=200, # rule states 10x the number of features
                      tuneGrid=tune_forest,
                      importance="impurity", # compare variable importance
                      tuneLength=5,
                      metric='Accuracy',
                      trControl=cv.control)
ggplot(atp3.forest, aes(x = interaction(mtry, eta))) +  labs(x = "No of Features", y = "Accuracy")
var_imp <- varImp(atp3.forest)
top_variables <- head(var_imp$importance, 10)
ggplot(top_variables, aes(x = Overall, y = reorder(rownames(top_variables), Overall))) +
  geom_bar(stat = "identity", fill = "lightcoral") +
  labs(x = "Importance", y = "Features")
atp3.forest.predict <- predict(atp3.forest, newdat = atp5.test, type="raw")
confusionMatrix(atp3.forest.predict, atp5.test$no_sets)

#### 2.4 XGBoost
cv.control <- trainControl(method = "cv", number = 10, classProbs = TRUE, savePredictions = "final")
tune_xgboost <- expand.grid(
  nrounds = 100,          # Number of boosting rounds
  eta = 0.3,              # Learning rate
  max_depth = c(3, 6, 9),  # Maximum tree depth
  gamma = c(0, 0.5, 0.9),              # Minimum loss reduction to make a further partition
  colsample_bytree = c(0.1, 0.5, 0.9), # Sub-sample ratio of columns when constructing each tree
  min_child_weight = 1,  # Minimum sum of instance weight (hessian) needed in a child
  subsample = 1
)
atp3.xgboost <- caret::train(
  no_sets ~ p1_rank + p2_rank + p1_age,
  data = atp3.train.under,
  method = "xgbTree",
  trControl = cv.control,
  tuneGrid = tune_xgboost,
  metric = 'Accuracy'
)
ggplot(atp3.xgboost, aes(x = interaction(max_depth))) +  labs(x = "Max depth", y = "Accuracy")
var_imp <- varImp(atp3.xgboost)
top_variables <- head(var_imp$importance, 10)
ggplot(top_variables, aes(x = Overall, y = reorder(rownames(top_variables), Overall))) +
  geom_bar(stat = "identity", fill = "lightcoral") +
  labs(x = "Importance", y = "Features")
atp3.xgboost.predict <- predict(atp3.xgboost, newdata = atp3.test, type = "raw")
confusionMatrix(atp3.xgboost.predict, atp3.test$no_sets)
plot(atp3.test$no_sets, atp3.xgboost.predict, main = "Predicted classification with XGBoost", xlab = "Reference", ylab = "Predicted")

############## 3. Neural Networks ##############
atp3.train.dummy <- dummy_cols(atp3.train.over, 
                               select_columns = c("tournament", "surface", "round", 
                                                  "p1_hand", "p2_hand", "p1_nat", "p2_nat", "no_sets"), 
                               remove_selected_columns = TRUE)
atp3.test.dummy <- dummy_cols(atp3.test, 
                              select_columns = c("tournament", "surface", "round", 
                                                 "p1_hand", "p2_hand", "p1_nat", "p2_nat", "no_sets"), 
                              remove_selected_columns = TRUE)

#### 3.1 With neuralnet
atp_nn <- neuralnet(no_sets_X2 + no_sets_X3 ~.,  data = atp3.train.dummy,
                      hidden = c(45), # Between input and output size, 2/3*input, < 2*input 
                      act.fct = 'logistic',
                      stepmax = 1e+05,
                      learningrate = 0.01,
                      algorithm = 'rprop+',
                      err.fct ='ce', # cross-entropy for classification problems
                      linear.output = FALSE,
                      likelihood= TRUE)
plot(atp_nn, rep="best", col.entry = "lightcoral", 
     col.hidden= "lightcoral",
     col.hidden.synapse = "darkred",
     col.out.synapse = "darkred",
     col.intercept = "lightgreen",
     fontsize = 8,
     show.weights = F)
nn_pred <- predict(atp_nn, atp3.test.dummy, type = "class")
class_labels <- levels(atp3.test$no_sets)
predicted_sets <- class_labels[max.col(nn_pred, "first")] %>% as.factor()
confusionMatrix(predicted_sets, atp3.test$no_sets)

#### 3.2 With Keras
atp3.train.features <- atp3.train.dummy %>% dplyr::select(-c(no_sets_X2,no_sets_X3))
atp3.test.features <- atp3.test.dummy %>% dplyr::select(-c(no_sets_X2,no_sets_X3))
atp3.train.target <- atp3.train.dummy %>% dplyr::select(c(no_sets_X2,no_sets_X3))
atp3.test.target <- atp3.test.dummy %>% dplyr::select(c(no_sets_X2,no_sets_X3))

nn <- keras_model_sequential() %>%
  layer_dense(input_shape = dim(atp3.train.features)[-1], units = 45, activation = 'sigmoid') %>%
  layer_dense(units = 2, activation = 'sigmoid') %>% 
  compile(optimizer = "rmsprop", loss = 'categorical_crossentropy', metrics =c("accuracy"))

atp3.keras <- nn %>% fit(as.matrix(atp3.train.features), as.matrix(atp3.train.target), epochs = 50, validation_split = 0.3)
atp3.keras
plot(atp3.keras)

atp3.keras.pred <- nn %>% predict(as.matrix(atp3.test.features))
class_labels <- levels(atp3.test$no_sets)
predicted_sets <- class_labels[max.col(atp3.keras.pred, "first")] %>% as.factor()
confusionMatrix(predicted_sets, atp3.test$no_sets)

############## 4. AutoML ##############
h2o.init()

atp3.h <- as.h2o(atp3)

atp3_splits <- h2o.splitFrame(
  data = atp3.h,
  ratios = c(0.7, 0.15),
  seed= 123
)

features <- colnames(atp3)[-ncol(atp3)]
target <- "no_sets"

atp3.h.train <- atp3_splits[[1]]
atp3.h.validate <- atp3_splits[[2]]
atp3.h.test <- atp3_splits[[3]]

# GLM: Generalized Linear Model
# GBM: Gradient Boosted Regression Tree
# DRF: Distributed Random Forest
# XGBoost
# DeepLearning: Deep Neural Network
# StackedEnsemble

automl_algorithms <- c("GLM", "GBM","DRF", "XGBoost", "DeepLearning", "StackedEnsemble")

atp3.automl <- h2o.automl(
  x = features,
  y = target,
  training_frame = atp3.h.train,
  validation_frame = atp3.h.validate,
  nfolds = 5,
  balance_classes = TRUE,
  include_algos = automl_algorithms,
  max_runtime_secs = 123,
  sort_metric = "mean_per_class_error"
)

atp3.automl.best <- atp3.automl@leader
atp3.automl.best@model$model_summary

atp3.automl.leaderboard <- h2o.get_leaderboard(object = atp3.automl)
head(atp3.automl.leaderboard, 10)

model_ids <- as.vector(atp3.automl@leaderboard$model_id)
index <- 2
second_best_model <- h2o.getModel(model_ids[index])

atp_automl_validation_metrics <- h2o.performance(
  model = atp3.automl.best,
  valid = TRUE
)
atp_automl_validation_metrics

atp_automl_test_metrics <- h2o.performance(
  model = atp3.automl.best,
  newdata = atp3.h.test
)
atp_automl_test_metrics

atp_automl_test_predictions <- h2o.predict(
  object = atp3.automl.best,
  newdata = atp3.h.test
)
X2_values <- as.vector(atp_automl_test_predictions$X2)
X3_values <- as.vector(atp_automl_test_predictions$X3)
predicted_sets_h2o <- data.frame(X2 = X2_values, X3 = X3_values)

predicted_sets_h2o$predict <- apply(predicted_sets_h2o, 1, function(row) {
  colnames(predicted_sets_h2o)[which.max(row)]
})

predicted_sets_h2o$predict <- as.factor(predicted_sets_h2o$predict)
#predicted_sets_h2o$predict <- ifelse(predicted_sets_h2o$X2 > predicted_sets_h2o$X3, 'X2', 'X3') %>% as.factor()
actual_sets_h2o <- as.vector(atp3.h.test$no_sets) %>% as.factor
confusionMatrix(predicted_sets_h2o$predict, actual_sets_h2o)

# h2o.varimp_plot(
#   model = atp3.automl.best,
#   num_of_features = 10
# )

h2o.shutdown(prompt = FALSE)