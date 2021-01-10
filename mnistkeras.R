library(tidyverse)
library(keras)

mnist <- dataset_mnist()

c(train_images, train_labels) %<-% mnist$train
c(test_images, test_labels) %<-% mnist$test

class_names = c('zero',
                'one',
                'two',
                'three',
                'four', 
                'five',
                'six',
                'seven',
                'eight',
                'nine')

dim(train_images)

dim(train_labels)

dim(test_images)

dim(test_labels)

image_1 <- as.data.frame(train_images[1, , ])
colnames(image_1) <- seq_len(ncol(image_1))
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1, "x", "value", -y)
image_1$x <- as.integer(image_1$x)

ggplot(image_1, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank())   +
  theme(aspect.ratio = 1) +
  xlab("") +
  ylab("")

train_images <- train_images / 255
test_images <- test_images / 255

par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- train_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_names[train_labels[i] + 1]))
}

model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 30, activation = 'sigmoid') %>%
  layer_dense(units = 10, activation = 'sigmoid')

model %>% compile(
  optimizer = 'sgd', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

model %>% fit(train_images, train_labels, epochs = 20, verbose = 2)

score <- model %>% evaluate(test_images, test_labels, verbose = 0)

cat('Test accuracy:', score[2], "\n")

predictions <- model %>% predict(test_images)

predictions[1, ]

which.max(predictions[1, ]) - 1
test_labels[1]

class_pred <- model %>% predict_classes(test_images)
class_pred[1:20]
test_labels[1:20]

(conf_mat <- table(test_labels,class_pred))

par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- test_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  # subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ]) - 1
  true_label <- test_labels[i]
  if (predicted_label == true_label) {
    color <- '#008800' 
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label + 1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
}

