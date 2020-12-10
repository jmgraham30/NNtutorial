# load libraries
library(tidyverse)
library(tidymodels)
# load complete data
all_data <- readRDS("./mnist_r_data.Rds")

digits_data <- all_data$digits_data # mnist data predictors
digits_targets <- all_data$digits_targets # mnist data responses
digits_images <- all_data$digits_images # mnist image pixel value matrices
digits_df <- all_data$df  # mnist data scaled and formatted as data frame
x_train <- all_data$x_train # training set predictors
y_train <- all_data$y_train # training set targets
x_test <- all_data$x_test # test set predictors
y_test <- all_data$y_test # test set targets

# plot a single image
reshape2::melt(t(digits_images[1, , ])) %>%
  ggplot(aes(x=Var1,y=Var2,fill=value)) + 
  geom_raster() + scale_fill_continuous(type = "viridis")

# function to convert digit to zero/one vector 
set_to_one <- function(n){
  x <- rep(0,10)
  x[n+1] <- 1
  return(x)
}

# convert targets to zero/one vectors 
y_v_train <- map(y_train,set_to_one)
y_v_test <- map(y_test,set_to_one)

(y_train[1])
(y_v_train[[1]])

