library(dplyr)

# Data Preparation --------------------------------------------------------

tbl_iris <- iris |> 
  mutate(Species = case_when(Species == "setosa" ~ 1, 
                             .default = 0))

set.seed(1001)
tbl_iris <- tbl_iris |> 
  slice_sample(prop = 1, replace = FALSE)

library(ggplot2)

tbl_iris |> 
  ggplot(aes(Sepal.Length, Petal.Length, color = factor(Species))) + 
  geom_point()


library(tidymodels)

set.seed(1001)
iris_split <- tbl_iris |> 
  initial_split(prop = 0.80, strata = Species)

iris_train <- iris_split |> 
  training()

iris_test <- iris_split |> 
  testing()

iris_train |> 
  count(Species) |> 
  mutate(pct = n/sum(n))

iris_test |> 
  count(Species) |> 
  mutate(pct = n/sum(n))

iris_recipe <- recipe(Species ~ ., data = iris_train) |> 
  step_normalize(all_numeric_predictors()) |> 
  prep()

iris_recipe |> 
  bake(new_data = iris_train)

x_train_iris <- iris_recipe |> 
  bake(new_data = NULL) |> 
  select(-Species)
x_test_iris <- iris_recipe |> 
  bake(new_data = iris_test) |> 
  select(-Species)

y_train_iris <- iris_recipe |> 
  bake(new_data = NULL) |> 
  pull(Species)

y_test_iris <- iris_recipe |> 
  bake(new_data = iris_test) |> 
  pull(Species)


library(keras)

set.seed(1001)
iris_keras <- keras_model_sequential() |> 
  layer_dense(
    units = 16, 
    kernel_initializer = "uniform", 
    activation = "relu", 
    input_shape = ncol(x_train_iris)
  ) |> 
  layer_dropout(rate = 0.1, seed = 1001) |> 
  layer_dense(
    units = 16, 
    kernel_initializer = "uniform", 
    activation = "relu"
  ) |> 
  layer_dropout(rate = 0.1, seed = 1001) |> 
  layer_dense(
    units = 1, 
    kernel_initializer = "uniform", 
    activation = "sigmoid"
  ) |> 
  compile(
    optimizer = optimizer_adam(learning_rate = 0.001), # "adam", 
    loss = "binary_crossentropy", 
    metrics = c("accuracy")
  )

history <- fit(
  object = iris_keras, 
  x = as.matrix(x_train_iris), 
  y = y_train_iris, 
  batch_size = 20, 
  epochs = 150, 
  validation_split = 0.2, 
  verbose = 2
)

plot(history)
history

iris_test_pred <- iris_keras |>  
  predict(x = as.matrix(x_test_iris)) |> 
  as.data.frame() |> 
  transmute(
    .pred_prob = V1, 
    .pred_class = factor(if_else(V1 > 0.5, 1, 0), 
                         levels = c(1, 0)), 
    .truth = factor(iris_test$Species, 
                    levels = c(1, 0))
  ) |> 
  as_tibble()

iris_test_pred |> 
  conf_mat(truth = .truth, estimate = .pred_class)

iris_test_pred |> 
  accuracy(truth = .truth, estimate = .pred_class) 

iris_test_pred |> 
  roc_curve(truth = .truth, estimate = .pred_prob) |> 
  autoplot()

iris_test_pred |> 
  roc_auc(truth = .truth, estimate = .pred_prob) 

iris_test_pred |> 
  f_meas(truth = .truth, estimate = .pred_class) 


# All Data ----------------------------------------------------------------

tbl_iris_bake <- iris_recipe |> 
  bake(new_data = tbl_iris)  

tbl_iris_pred <- iris_keras |> 
  predict(x = as.matrix(tbl_iris_bake[,-5])) |> 
  as.data.frame() |> 
  transmute(
    .pred_prob = V1, 
    .pred_class = factor(if_else(V1 > 0.5, 1, 0), 
                         levels = c(1, 0)), 
    .truth = factor(tbl_iris_bake$Species, 
                       levels = c(1, 0))
    ) |> 
  as_tibble()

tbl_iris_pred |> 
  conf_mat(truth = .truth, estimate = .pred_class)

tbl_iris_pred |> 
  accuracy(truth = .truth, estimate = .pred_class) 

tbl_iris_pred |> 
  roc_curve(truth = .truth, estimate = .pred_prob) |> 
  autoplot()

tbl_iris_pred |> 
  roc_auc(truth = .truth, estimate = .pred_prob) 

tbl_iris_pred |> 
  f_meas(truth = .truth, estimate = .pred_class) 


iris_keras |> 
  save_model_hdf5("output/iris_keras.h5")

my_iris_keras <- load_model_hdf5("output/iris_keras.h5")

my_iris_keras |> 
  predict(x = as.matrix(tbl_iris_bake[,-5])) |> 
  as.data.frame() |> 
  transmute(
    .pred_prob = V1, 
    .pred_class = factor(if_else(V1 > 0.5, "setosa", "not-setosa"), 
                         levels = c("setosa", "not-setosa"))
  )

library(DALEXtra)

iris_explainer <- explain(
    model = iris_keras, 
    data = as.matrix(iris_train[,-5]), 
    y = iris_train$Species,
    verbose = TRUE, 
    label = "Iris Species Prediction"
  )

set.seed(12345)
x <- iris_test |> 
  nrow() |> 
  sample(3)
new_obs <- iris_test |>
  dplyr::slice(x) 

new_obs <- iris_recipe |> 
  bake(new_data = new_obs)

iris_keras |> 
  predict(x = as.matrix(new_obs[,-5]))

library(modelStudio)
modelStudio(
  explainer = iris_explainer, 
  new_observation = as.matrix(new_obs[,-5]), 
  new_observation_y = new_obs$Species,
  max_features = 4, 
  facet_dim = c(2, 3)
)
