
############################
# LOADING PACKAGES #########
############################

if(!require(devtools, quietly = TRUE)) install.packages('devtools')
if(!require(pander, quietly = TRUE)) install.packages('pander')
if(!require(knitr, quietly = TRUE)) install.packages('knitr')
if(!require(dplyr, quietly = TRUE)) install.packages('dplyr')
if(!require(tidyr, quietly = TRUE)) install.packages('tidyr')
if(!require(stringr, quietly = TRUE)) install.packages('stringr')
if(!require(lubridate, quietly = TRUE)) install.packages('lubridate')
if(!require(purrr, quietly = TRUE)) install.packages('purrr')
if(!require(DT, quietly = TRUE)) install.packages('DT')
if(!require(tidytext, quietly = TRUE)) install.packages('tidytext')
if(!require(ggplot2, quietly = TRUE)) install.packages('ggplot2')
if(!require(textstem, quietly = TRUE)) install.packages('textstem')
if(!require(tm, quietly = TRUE)) install.packages('tm')
if(!require(splitstackshape, quietly = TRUE)) install.packages('splitstackshape')
if(!require(text2vec, quietly = TRUE)) install.packages('text2vec')
if(!require(reshape, quietly = TRUE)) install.packages('reshape')
if(!require(readr, quietly = TRUE)) install.packages('readr')
if(!require(zoo, quietly = TRUE)) install.packages('zoo')
if(!require(keras, quietly = TRUE)) install.packages('keras')
if(!require(ROCR, quietly = TRUE)) install.packages('ROCR')
if(!require(caret, quietly = TRUE)) install.packages('caret')
if(!require(svMisc, quietly = TRUE)) install.packages('svMisc')

pkg <- c("devtools",
         "pander",
         "knitr",
         "dplyr",
         "tidyr",
         "stringr",
         "lubridate",
         "purrr",
         "DT",
         "tidytext",
         "ggplot2",
         "textstem",
         "tm",
         "splitstackshape",
         "text2vec",
         "reshape",
         "readr",
         "zoo",
         "keras",
         "ROCR",
         "caret",
         "svMisc")
invisible(lapply(pkg, library, character.only = TRUE))

############################
# MODEL NAMES ##############
############################

models <- c("covnet_fasttext_embeddings",
            "covnet",
            "covnet_bidirectional_lstm")

############################
# HELPER FUNCTIONS #########
############################

exclude_stop_words_function <- function(data){
  data(stop_words)
  data_with_id <- data %>% 
    mutate(sent_id = row_number())
  data <- data %>% 
    mutate(sent_id = row_number()) %>% 
    unnest_tokens(output = word, input = Text, token = 'words') %>% 
    anti_join(stop_words, by = c('word' = 'word')) %>% 
    group_by(sent_id) %>% 
    summarise(Text = paste(word, collapse = ' ')) %>% 
    ungroup() %>% 
    left_join(data_with_id, by = "sent_id") %>% 
    select(-Text.x, -sent_id) %>% 
    dplyr::rename(Text = Text.y)
  data
}

data_processing_pipeline <- function(notes_training,
                                     notes_test,
                                     max_words,
                                     max_length,
                                     train_validation_split = 0.8,
                                     exclude_stop_words = FALSE,
                                     seed = 2019){
  
  # Train-Test Split
  notes_training <- notes_training %>% 
    mutate(value = 1)  %>% 
    spread(key = Category, value = value,  fill = 0) 
  notes_test <- notes_test %>% 
    mutate(value = 1)  %>% 
    spread(key = Category, value = value,  fill = 0) 
  
  if (exclude_stop_words) {
    notes_training <- exclude_stop_words_function(notes_training)
    notes_test <- exclude_stop_words_function(notes_test)
    notes_full <- exclude_stop_words_function(notes_full)
  }
  
  # Turn Data into Matrix
  texts <- notes_full %>% 
    pull(Text)
  
  texts_training <- notes_training %>%
    pull(Text)
  labels_training <- notes_training %>% 
    select(-Text) %>% 
    as.matrix()
  
  texts_test <- notes_test %>% 
    pull(Text)
  labels_test <- notes_test %>%     
    select(-Text) %>% 
    as.matrix()
  
  # TOKENIZING
  tokenizer <- text_tokenizer(num_words = max_words) %>%
    fit_text_tokenizer(texts)
  word_index = tokenizer$word_index
  
  # TRAINING
  sequences_training <- texts_to_sequences(tokenizer, texts_training)
  data_training <- pad_sequences(sequences_training, maxlen = max_length)
  labels_training <- as.array(labels_training)
  
  # TEST
  sequences_test <- texts_to_sequences(tokenizer, texts_test)
  data_test <- pad_sequences(sequences_test, maxlen = max_length)
  labels_test <- as.array(labels_test)
  
  set.seed(seed)
  training_samples <- floor(nrow(data_training) * train_validation_split)
  validation_samples <- nrow(data_training) - training_samples
  
  indices <- sample(1:nrow(data_training))
  training_indices <- indices[1:training_samples]
  validation_indices <- indices[(training_samples + 1):(training_samples + validation_samples)]
  
  x_train <- data_training[training_indices,]
  y_train <- labels_training[training_indices,]
  x_val <- data_training[validation_indices,]
  y_val <- labels_training[validation_indices,]
  x_test <- data_test
  y_test <- labels_test
  
  list('x_train' = x_train,
       'y_train' = y_train,
       'x_val' = x_val,
       'y_val' = y_val,
       'x_test' = x_test,
       'y_test' = y_test,
       'word_index' = word_index)
}

train_validate_test <- function(data,
                                model, 
                                epoch,
                                batch_size,
                                model_folder){
  # Create history object
  history <- model %>% keras::fit(
    data$x_train, 
    data$y_train,
    epochs = epoch,
    batch_size = batch_size,
    validation_data = list(data$x_val, 
                           data$y_val))
  
  # Saving model weights
  model_name <- deparse(substitute(model))
  file_path <- file.path(model_folder, 
                         paste(model_name, "h5", sep = "."))
  model %>% 
    save_model_hdf5(file_path)
  
  result <- model %>% 
    evaluate(data$x_test, data$y_test)
  
  list('history' = history,
       'result' = result)
}

model_deep_covnet <- function(max_len,
                              max_features,
                              optimizer,
                              loss = "binary_crossentropy",
                              output_unit = 1){
  
  model_deep_covnet <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_features, 
                    output_dim = 128,
                    input_length = max_len) %>%
    layer_conv_1d(filters = 32, 
                  kernel_size = 3, 
                  activation = "relu") %>%
    layer_max_pooling_1d(pool_size = 5) %>%
    layer_dropout(0.2) %>% 
    layer_conv_1d(filters = 64, 
                  kernel_size = 3, 
                  activation = "relu") %>%
    layer_max_pooling_1d(pool_size = 5) %>%
    layer_dropout(0.2) %>% 
    layer_conv_1d(filters = 64, 
                  kernel_size = 3, 
                  activation = "relu") %>%
    layer_max_pooling_1d(pool_size = 5) %>%
    layer_dropout(0.2) %>% 
    layer_conv_1d(filters = 128, 
                  kernel_size = 3, 
                  activation = "relu") %>%
    layer_global_max_pooling_1d() %>%
    layer_dense(units = output_unit, activation = "sigmoid")
  
  model_deep_covnet %>% compile(
    optimizer = optimizer,
    loss = loss,
    metrics = c("acc"))
  
  model_deep_covnet
  
}

model_deep_covnet_embeddings <- function(max_len,
                                         max_features,
                                         optimizer,
                                         embedding_dim,
                                         word_index,
                                         embeddings_index,
                                         loss = "binary_crossentropy",
                                         output_unit = 1){
  
  embedding_matrix <- array(0, c(max_features, embedding_dim))
  
  for (word in names(word_index)) {
    index <- word_index[[word]]
    if (index < max_features) {
      embedding_vector <- embeddings_index[[word]]
      if (!is.null(embedding_vector))
        embedding_matrix[index+1,] <- embedding_vector
    }
  }
  
  model_deep_covnet_embeddings <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_features, 
                    output_dim = embedding_dim, 
                    input_length = max_len,
                    name = "embedding_1") %>%
    layer_conv_1d(filters = 32, 
                  kernel_size = 3, 
                  activation = "relu") %>%
    layer_max_pooling_1d(pool_size = 5) %>%
    layer_dropout(0.2) %>% 
    layer_conv_1d(filters = 64, 
                  kernel_size = 3, 
                  activation = "relu") %>%
    layer_max_pooling_1d(pool_size = 5) %>%
    layer_dropout(0.2) %>% 
    layer_conv_1d(filters = 64, 
                  kernel_size = 3, 
                  activation = "relu") %>%
    layer_max_pooling_1d(pool_size = 5) %>%
    layer_dropout(0.2) %>% 
    layer_conv_1d(filters = 128, 
                  kernel_size = 3, 
                  activation = "relu") %>%
    layer_global_max_pooling_1d() %>%
    layer_dense(units = output_unit, activation = "sigmoid")
  
  get_layer(model_deep_covnet_embeddings, name = "embedding_1") %>%
    set_weights(list(embedding_matrix)) %>%
    freeze_weights()
  
  model_deep_covnet_embeddings %>% compile(
    optimizer = optimizer,
    loss = loss,
    metrics = c("acc"))
  
  model_deep_covnet_embeddings
  
}

model_deep_covnet_blstm <- function(max_len,
                                    max_features,
                                    optimizer,
                                    loss = "binary_crossentropy",
                                    output_unit = 1){
  
  model_deep_covnet_blstm <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_features, 
                    output_dim = 128,
                    input_length = max_len) %>%
    layer_conv_1d(filters = 32, 
                  kernel_size = 3, 
                  activation = "relu") %>%
    layer_max_pooling_1d(pool_size = 5) %>%
    layer_dropout(0.2) %>% 
    layer_conv_1d(filters = 64, 
                  kernel_size = 3, 
                  activation = "relu") %>%
    layer_max_pooling_1d(pool_size = 5) %>%
    layer_dropout(0.2) %>% 
    layer_conv_1d(filters = 64, 
                  kernel_size = 3, 
                  activation = "relu") %>%
    layer_max_pooling_1d(pool_size = 5) %>%
    layer_dropout(0.2) %>% 
    layer_conv_1d(filters = 128, 
                  kernel_size = 3, 
                  activation = "relu") %>%
    bidirectional(layer_lstm(units = 128)) %>%
    layer_dense(units = output_unit, activation = "sigmoid")
  
  model_deep_covnet_blstm %>% compile(
    optimizer = optimizer,
    loss = loss,
    metrics = c("acc"))
  
  model_deep_covnet_blstm
  
}

generate_metrics <- function(model,
                             data,
                             threshold,
                             optimal_threshold = FALSE){
  y_true <- data$y_test
  x_test <- data$x_test
  y_predicted <- model %>%  
    predict(x_test)
  
  y_predicted[y_predicted < threshold] <- 0
  y_predicted[y_predicted >= threshold] <- 1
  y_predicted <- as.vector(y_predicted)
  
  accuracy <- confusionMatrix(as.factor(y_predicted), 
                              as.factor(y_true), 
                              positive = "1")$overall[1]
  
  precision <- posPredValue(as.factor(y_predicted), 
                            as.factor(y_true), 
                            positive="1")
  
  recall <- sensitivity(as.factor(y_predicted), 
                        as.factor(y_true), 
                        positive="1")
  
  F1 <- (2 * precision * recall) / (precision + recall)
  
  if(optimal_threshold){
    cat(paste("Out-of-Sample Accuracy at Optimal Threshold:", round(accuracy, 2)))
    cat(paste("Out-of-Sample Precision at Optimal Threshold:", round(precision, 2)))
    cat(paste("Out-of-Sample Recall at Optimal Threshold:", round(recall, 2)))
    cat(paste("Out-of-Sample F1 Score at Optimal Threshold:", round(F1, 2)))
  } else {
    cat(paste("Out-of-Sample Accuracy:", round(accuracy, 2)))
    cat(paste("Out-of-Sample Precision:", round(precision, 2)))
    cat(paste("Out-of-Sample Recall:", round(recall, 2)))
    cat(paste("Out-of-Sample F1 Score:", round(F1, 2)))
  }
  
  list("accuracy" = accuracy,
       "precision" = precision,
       "recall" = recall,
       "F1" = F1)
}

get_optimal_threshold <- function(model,
                                  data,
                                  interval = 0.05) {
  
  y_true <- data$y_test
  x_test <- data$x_test
  y_predicted_original <- model %>%  
    predict(x_test)
  
  F1_list <- c()
  for (i in seq(0, 1, interval)) {
    
    y_predicted <- y_predicted_original
    y_predicted[y_predicted < i] <- 0
    y_predicted[y_predicted >= i] <- 1
    precision <- posPredValue(as.factor(y_predicted), 
                              as.factor(y_true), 
                              positive="1")
    recall <- sensitivity(as.factor(y_predicted), 
                          as.factor(y_true), 
                          positive="1")
    F1 <- (2 * precision * recall) / (precision + recall)
    F1_list <- c(F1_list, F1)
  }
  names(F1_list) <- seq(0, 1, interval)
  optimal_F1 <- F1_list[which.max(F1_list)]
  
  list("optimal_F1" = optimal_F1,
       "optimal_threshold" = as.numeric(names(optimal_F1)))
}


