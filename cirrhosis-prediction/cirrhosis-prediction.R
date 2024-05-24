# Load necessary libraries
library(caret)
library(ROSE)
library(randomForest)
library(e1071)
library(keras)

# Suppress warnings
options(warn=-1)

# Command line arguments
args <- commandArgs(trailingOnly=TRUE)

# Read CSV file
df <- read.csv('input/cirrhosis.csv')

# Drop 'ID' column
df <- df[, -which(names(df) == 'ID')]

# Handling Missing Values

# Drop rows with missing 'Stage'
df <- df[!is.na(df$Stage), ]

# Numerical columns: Fill with median
numerical_columns <- names(df)[sapply(df, is.numeric)]
df[, (numerical_columns) := lapply(.SD, function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x)), .SDcols = numerical_columns]

# Categorical columns: Fill with most frequent
categorical_columns <- names(df)[sapply(df, is.character)]
df[, (categorical_columns) := lapply(.SD, function(x) ifelse(is.na(x), levels(factor(x))[which.max(table(factor(x)))], x)), .SDcols = categorical_columns]

# Convert 'Stage' to integer
df$Stage <- as.integer(df$Stage)

# Recoding and Mutating Data

# Translate categorical values to numerical values
df$Sex <- ifelse(df$Sex == 'M', 0, 1)
df$Ascites <- ifelse(df$Ascites == 'N', 0, 1)
df$Drug <- ifelse(df$Drug == 'D-penicillamine', 0, 1)
df$Hepatomegaly <- ifelse(df$Hepatomegaly == 'N', 0, 1)
df$Spiders <- ifelse(df$Spiders == 'N', 0, 1)
df$Edema <- ifelse(df$Edema == 'N', 0, ifelse(df$Edema == 'Y', 1, -1))
df$Status <- ifelse(df$Status == 'C', 0, ifelse(df$Status == 'CL', 1, -1))

# Select input and output variables
X <- df[, !names(df) %in% c('Status', 'N_Days', 'Stage')]
y <- df$Stage

# Upsampling using SMOTE
df_resampled <- ovun.sample(y, method = "over", N = length(y), seed = 123)

# Scaling Data
X_scaled <- scale(df_resampled$data)

# Splitting the data 80/20
set.seed(23)
split <- createDataPartition(df_resampled$y, p = 0.8, list = FALSE)
X_train <- X_scaled[split, ]
X_test <- X_scaled[-split, ]
y_train <- df_resampled$y[split]
y_test <- df_resampled$y[-split]

# Modeling the Cirrhosis prediction

# Random Forest
modelTrainerRF_SVM <- function(X_train, y_train, modelName, predictDataPoint) {
  models <- list(SVM = svm(), RF = randomForest())
  model <- models[[modelName]]
  model <- train(X_train, y_train, method = model, trControl = trainControl(method = 'cv'))
  
  dataPoint <- read.csv(predictDataPoint)
  model_predict <- predict(model, newdata = dataPoint)
  
  cat(sprintf("\n[%s] Cirrhosis Predicted Stage: %s\n", modelName, model_predict))
}

# MLP
modelTrainerMLP <- function(X_train, y_train, X_test, y_test, predictDataPoint) {
  model_mlp <- keras_model_sequential()
  
  input_shape <- dim(X_train)[2]
  
  model_mlp %>% 
    layer_dense(units = 32, activation = 'relu', input_shape = input_shape) %>% 
    layer_dense(units = 64, activation = 'relu') %>% 
    layer_dense(units = 32, activation = 'relu') %>% 
    layer_dense(units = 5, activation = 'softmax')
  
  model_mlp %>% compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = c('accuracy'))
  
  model_mlp %>% fit(X_train, y_train, epochs = 20, validation_data = list(X_test, y_test))
  
  dataPoint <- read.csv(predictDataPoint)
  MLPModel_predict <- predict(model_mlp, dataPoint)
  MLP_predict <- max.col(MLPModel_predict)
  
  cat(sprintf("\n[MLP] Cirrhosis Predicted Stage: %s\n", MLP_predict))
}

# CNN
modelTrainerCNN <- function(X_train, y_train, X_test, y_test, predictDataPoint) {
  model_cnn <- keras_model_sequential()
  
  model_cnn %>% 
    layer_conv_1d(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = c(16, 1)) %>% 
    layer_max_pooling_1d(pool_size = 2) %>% 
    layer_conv_1d(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu') %>% 
    layer_max_pooling_1d(pool_size = 2) %>% 
    layer_conv_1d(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu') %>% 
    layer_max_pooling_1d(pool_size = 2) %>% 
    layer_dropout(0.25) %>% 
    layer_flatten() %>% 
    layer_dense(units = 16, activation = 'relu') %>% 
    layer_dense(units = 5, activation = 'softmax')
  
  model_cnn %>% compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = 'accuracy')
  
  model_cnn %>% fit(x = array_reshape(X_train, dim = c(dim(X_train), 1)), y = y_train, validation_data = list(array_reshape(X_test, dim = c(dim(X_test), 1)), y_test), epochs = 20)
  
  dataPoint <- read.csv(predictDataPoint)
  CNNModel_predict <- predict(model_cnn, array_reshape(dataPoint, dim = c(dim(dataPoint), 1)))
  CNN_predict <- max.col(CNNModel_predict)
  
  cat(sprintf("\n[CNN] Cirrhosis Predicted Stage: %s\n", CNN_predict))
}

if (args[1] == 'RF') {
  modelTrainerRF_SVM(X_train, y_train, 'RF', args[2])
} else if (args[1] == 'SVM') {
  modelTrainerRF_SVM(X_train, y_train, 'SVM', args[2])
} else if (args[1] == 'MLP') {
  modelTrainerMLP(X_train, y_train, X_test, y_test, args[2])
} else if (args[1] == 'CNN') {
  modelTrainerCNN(X_train, y_train, X_test, y_test, args[2])
}
