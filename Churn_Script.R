library(reticulate)
library(keras)
library(corrr)
library(easypackages)
libraries<-c("lime","dplyr", "tidyquant", "rsample", "recipes","yardstick","corrr","keras","kernlab","e1071","caret","ellipse","randomForest","reticulate","devtools","igraph"  )
libraries(libraries)





churndata <- read.csv("telcocustomer.csv")
# check if columns are NA we can choose to drop them or make them 0's
library(mice)
md.pattern(churndata)
apply(churndata, 2, function(x) any(is.na(x)))
sum(is.na(churndata$TotalCharges))
churn_data_tbl <- churndata %>%
  select(-customerID) %>%
  drop_na() %>%
  select(Churn, everything())
glimpse(churn_data_tbl)
# Split test/training sets
set.seed(100)
train_test_split <- initial_split(churn_data_tbl, prop = 0.8)
train_test_split
# Retrieve train and test sets
train_tbl <- training(train_test_split)
test_tbl  <- testing(train_test_split)
# explore the data (distributions)
#72/12= 6 bins 12 months = 1 yr does yr affect churning
summary(train_tbl$tenure)

hist(train_tbl$tenure)
#t <- log(train_tbl$tenure)
#hist(t)
# 9/5 8:02 pm learn how to churn
breaks <- c(0,12,24,36,48,60,72)
# specify interval/bin labels
length(train_tbl$tenure)
labels <- c("0-12","13-24","25-36","37-48","49-60","61-72")
bins <- cut(train_tbl$tenure, breaks, include.lowest = T, right=FALSE, labels=labels)
summary(bins)
plot(bins, main="Tenure Binned into years", xlab="Tenure Months",ylab="Frequency",col="bisque")
#Customer Churn Analytics 2 way
summary(train_tbl$TotalCharges)
hist(train_tbl$TotalCharges)
hist(log(train_tbl$TotalCharges))
# Determine if log transformation improves correlation
# between TotalCharges and Churn
#select train tbl churn and total charges we mutate
#try other transformatios?
train_tbl %>%
  select(Churn, TotalCharges) %>%
  mutate(
    Churn = Churn %>% as.factor() %>% as.numeric(),
    LogTotalCharges = log(TotalCharges)
  ) %>%
  correlate() %>%
  focus(Churn) %>%
  fashion()

rec_obj <- recipe(Churn ~ ., data = train_tbl) %>%
  step_discretize(tenure, options = list(cuts = 6)) %>%
  step_log(TotalCharges) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep(data = train_tbl)
## step 1 discretize training 
## step 2 log training 
## step 3 dummy training 
## step 4 center training 
## step 5 scale training
rec_obj

x_train_tbl <- bake(rec_obj, new_data = train_tbl) %>% select(-Churn)
x_test_tbl <- bake(rec_obj, new_data = test_tbl) %>% select(-Churn)

glimpse(x_train_tbl)

print(y_train_vec)
y_train_vec <- ifelse(pull(train_tbl, Churn) == "Yes", 1, 0)
y_test_vec  <- ifelse(pull(test_tbl, Churn) == "Yes", 1, 0)



model_keras <- keras_model_sequential()

model_keras %>% 
  # First hidden layer
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", 
    activation         = "relu", 
    input_shape        = ncol(x_train_tbl)) %>% 
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  # Second hidden layer
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", 
    activation         = "relu") %>% 
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  # Output layer
  layer_dense(
    units              = 1, 
    kernel_initializer = "uniform", 
    activation         = "sigmoid") %>% 
  # Compile ANN
  compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = c('accuracy')
  )
model_keras

fit_keras <- fit(
  object           = model_keras, 
  x                = as.matrix(x_train_tbl), 
  y                = y_train_vec,
  batch_size       = 50, 
  epochs           = 35,
  validation_split = 0.30
)
fit_keras

library(ggplot2)
plot(fit_keras)
# Plot the training/validation history of our Keras model
plot(fit_keras) +
  theme_tq() +
  scale_color_tq() +
  scale_fill_tq() +
  labs(title = "Deep Learning Training Results")

yhat_keras_class_vec <- predict_classes(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()

# Predicted Class Probability
yhat_keras_prob_vec  <- predict_proba(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()

estimates_keras_tbl <- tibble(
  truth      = as.factor(y_test_vec) %>% fct_recode(yes = "1", no = "0"),
  estimate   = as.factor(yhat_keras_class_vec) %>% fct_recode(yes = "1", no = "0"),
  class_prob = yhat_keras_prob_vec
)

estimates_keras_tbl
options(yardstick.event_first = FALSE)
str(estimates_keras_tbl)

estimates_keras_tbl %>% conf_mat(truth, estimate)
conf_mat(estimates_keras_tbl,truth,estimate)
estimates_keras_tbl %>% metrics(truth, estimate)
estimates_keras_tbl %>% roc_auc(truth, class_prob)
estimates_keras_tbl$truth
# Precision
tibble(
  precision =precision(estimates_keras_tbl$truth, estimates_keras_tbl$estimate),
  recall    =  recall(estimates_keras_tbl$truth, estimates_keras_tbl$estimate)
)
estimates_keras_tbl %>% f_meas(truth, estimate, beta = 1)
str(estimates_keras_tbl$truth)


# visualization
class(model_keras)

model_type.keras.models.Sequential <- function(x, ...) {
  return("classification")
}
library(lime)
# Setup lime::predict_model() function for keras
predict_model.keras.models.Sequential <- function(x, newdata, type, ...) {
  pred <- predict_proba(object = x, x = as.matrix(newdata))
  return(data.frame(Yes = pred, No = 1 - pred))
}


# Test our predict_model() function
predict_model.keras.models.Sequential(x = model_keras, newdata = x_test_tbl, type = 'raw')%>% tibble::as_tibble()
print(model_keras)

test <- x_test_tbl[1:10,]
#test <- as.character(test)
explainer <- lime::lime(
  x              = x_train_tbl, 
  model          = model_keras, 
  bin_continuous = FALSE)

str(x_train_tbl)
explanation <- lime::explain(
  x_test_tbl[1:10,], 
  explainer    = explainer, 
  n_labels     = 1, 
  n_features   = 4,
  kernel_width = 0.5)


plot_features(explanation) +
  labs(title = "LIME Variable Visualization",
       subtitle = "10 cases")

plot_explanations(explanation) +
  labs(title = "LIME Feature Importance Heatmap",
       subtitle = "Hold Out (Test) Set, First 10 Cases Shown")

correlation_analysis <- x_train_tbl %>%
  mutate(Churn = y_train_vec) %>%
  correlate() %>%
  focus(Churn) %>%
  rename(feature = rowname) %>%
  arrange(abs(Churn)) %>%
  mutate(feature = as_factor(feature)) 
correlation_analysis

# Correlation visualization
correlation_analysis %>%
  ggplot(aes(x = Churn, y = fct_reorder(feature, desc(Churn)))) +
  geom_point() +
  # Positive Correlations - Contribute to churn
  geom_segment(aes(xend = 0, yend = feature), 
               color = palette_light()[[2]], 
               data = correlation_analysis %>% filter(Churn > 0)) +
  geom_point(color = palette_light()[[2]], 
             data = correlation_analysis %>% filter(Churn > 0)) +
  # Negative Correlations - Prevent churn
  geom_segment(aes(xend = 0, yend = feature), 
               color = palette_light()[[1]], 
               data = correlation_analysis %>% filter(Churn < 0)) +
  geom_point(color = palette_light()[[1]], 
             data = correlation_analysis %>% filter(Churn < 0)) +
  # Vertical lines
  geom_vline(xintercept = 0, color = palette_light()[[5]], size = 1, linetype = 2) +
  geom_vline(xintercept = -0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
  geom_vline(xintercept = 0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
  # Aesthetics
  labs(title = "Churn Correlation Analysis",
       subtitle = "Positive Correlations (contribute to churn), Negative Correlations (prevent churn)",
       y = "Feature Importance")


# Variables that affect churn
# Senior Citizen
churndata %>%
  mutate(Churn = ifelse(Churn == "Yes", 1, 0)) %>%
  ggplot(aes(x = as.factor(SeniorCitizen), y = Churn)) +
  geom_jitter(alpha = 0.25, color = palette_light()[[6]]) +
  geom_violin(alpha = 0.6, fill = palette_light()[[1]]) +
  labs(
    title = "Senior Citizen",
    subtitle = "Non-senior citizens less likely to leave",
    x = "Senior Citizen (Yes = 1)"
  )

# Online Security
churndata %>%
  mutate(Churn = ifelse(Churn == "Yes", 1, 0)) %>%
  ggplot(aes(x = OnlineSecurity, y = Churn)) +
  geom_jitter(alpha = 0.25, color = palette_light()[[6]]) +
  geom_violin(alpha = 0.6, fill = palette_light()[[1]]) +
  labs(
    title = "Online Security",
    subtitle = "Customers without online security are more likely to leave"
  )

# Internet Service
churndata %>%
  mutate(Churn = ifelse(Churn == "Yes", 1, 0)) %>%
  ggplot(aes(x = as.factor(InternetService), y = Churn)) +
  geom_jitter(alpha = 0.25, color = palette_light()[[6]]) +
  geom_violin(alpha = 0.6, fill = palette_light()[[1]]) +
  labs(
    title = "Internet Service",
    subtitle = "Fiber optic more likely to leave",
    x = "Internet Service"
  )

# Payment Method
churndata %>%
  mutate(Churn = ifelse(Churn == "Yes", 1, 0)) %>%
  ggplot(aes(x = as.factor(PaymentMethod), y = Churn)) +
  geom_jitter(alpha = 0.25, color = palette_light()[[6]]) +
  geom_violin(alpha = 0.6, fill = palette_light()[[1]]) +
  labs(
    title = "Payment Method",
    subtitle = "Electronic check more likely to leave",
    x = "Payment Method"
  )

# Tenure
churndata %>%
  ggplot(aes(x = Churn, y = tenure)) +
  geom_jitter(alpha = 0.25, color = palette_light()[[6]]) +
  geom_violin(alpha = 0.6, fill = palette_light()[[1]]) +
  labs(
    title = "Tenure",
    subtitle = "Customers with lower tenure are more likely to leave"
  )

# Contract
churndata %>%
  mutate(Churn = ifelse(Churn == "Yes", 1, 0)) %>%
  ggplot(aes(x = as.factor(Contract), y = Churn)) +
  geom_jitter(alpha = 0.25, color = palette_light()[[6]]) +
  geom_violin(alpha = 0.6, fill = palette_light()[[1]]) +
  labs(
    title = "Contract Type",
    subtitle = "Two and one year contracts much less likely to leave",
    x = "Contract"
  )