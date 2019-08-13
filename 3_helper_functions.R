#####################
# HELPER FUNCTIONS ##
#####################

#####################
# DATA PROCESSING ###
#####################

data_processing <- function(notes_training,
                            notes_test,
                            max_length,
                            max_words,
                            exclude_stop_words = FALSE,
                            seed = 1174,
                            train_validation_split,
                            label){
  
  notes_full <- notes_training %>% 
    bind_rows(notes_test)
  
  exclude_stop_words_function <- function(data){
    data(stop_words)
    
    data_with_id <- data %>% 
      mutate(sent_id = row_number())
    
    data <- data %>% 
      mutate(sent_id = row_number()) %>% 
      unnest_tokens(output = word, input = note_processed, token = 'words') %>% 
      anti_join(stop_words, by = c('word' = 'word')) %>% 
      group_by(sent_id) %>% 
      summarise(note_processed = paste(word, collapse = ' ')) %>% 
      ungroup() %>% 
      left_join(data_with_id, by = "sent_id") %>% 
      select(-note_processed.x, -sent_id) %>% 
      dplyr::rename(note_processed = note_processed.y)
    data
  }
  
  if (exclude_stop_words) {
    notes_training <- exclude_stop_words_function(notes_training)
    notes_test <- exclude_stop_words_function(notes_test)
    notes_full <- exclude_stop_words_function(notes_full)
  }
  
  texts <- notes_full %>% 
    pull(note_processed)
  
  texts_training <- notes_training %>%
    pull(note_processed)
  labels_training <- notes_training %>% 
    pull(label)
  
  texts_test <- notes_test %>% 
    pull(note_processed)
  labels_test <- notes_test %>% 
    pull(label)
  
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
  y_train <- labels_training[training_indices]
  x_val <- data_training[validation_indices,]
  y_val <- labels_training[validation_indices]
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



pull_function <- function(data, label){
  data %>% 
    pull(label)
}

#################################
# DATA PROCESSING MULTI-LABEL ###
#################################

data_processing_multilabel <- function(notes_training,
                                       notes_test,
                                       max_length,
                                       max_words,
                                       exclude_stop_words = FALSE,
                                       seed = 1174,
                                       train_validation_split) {
  
  notes_full <- notes_training %>% 
    bind_rows(notes_test)
  
  exclude_stop_words_function <- function(data){
    data(stop_words)
    
    data_with_id <- data %>% 
      mutate(sent_id = row_number())
    
    data <- data %>% 
      mutate(sent_id = row_number()) %>% 
      unnest_tokens(output = word, input = note_processed, token = 'words') %>% 
      anti_join(stop_words, by = c('word' = 'word')) %>% 
      group_by(sent_id) %>% 
      summarise(note_processed = paste(word, collapse = ' ')) %>% 
      ungroup() %>% 
      left_join(data_with_id, by = "sent_id") %>% 
      select(-note_processed.x, -sent_id) %>% 
      dplyr::rename(note_processed = note_processed.y)
    data
  }
  
  if (exclude_stop_words) {
    notes_training <- exclude_stop_words_function(notes_training)
    notes_test <- exclude_stop_words_function(notes_test)
    notes_full <- exclude_stop_words_function(notes_full)
  }
  
  texts <- notes_full %>% 
    pull(note_processed)
  
  texts_training <- notes_training %>%
    pull(note_processed)
  labels_training <- notes_training %>% 
    select(-c(note_processed, report_head, report_no, with_labels)) %>% 
    as.matrix()
  
  texts_test <- notes_test %>% 
    pull(note_processed)
  labels_test <- notes_test %>%     
    select(-c(note_processed, report_head, report_no, with_labels)) %>% 
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

#########################
# TRAIN-VALIDATE-TEST ###
#########################

train_validate_test <- function(data,
                                model, 
                                epoch,
                                batch_size,
                                model_folder,
                                label){
  history <- model %>% fit(
    data$x_train, 
    data$y_train,
    epochs = epoch,
    batch_size = batch_size,
    validation_data = list(data$x_val, 
                           data$y_val))
  
  file_path <- file.path(model_folder, paste(paste(label, model_name, sep = "_"),
                                             "h5", sep = "."))
  save_model_weights_hdf5(model, 
                          file_path)

  result <- model %>% 
    evaluate(data$x_test, data$y_test)
  
  list('history' = history,
       'result' = result)
}

#######################
# RNN #################
#######################

model_rnn <- function(max_features,
                      optimizer,
                      loss = "binary_crossentropy",
                      output_unit = 1){
  
  model_rnn <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_features, 
                    output_dim = 32) %>%
    layer_simple_rnn(units = 32) %>%
    layer_dropout(0.5) %>% 
    layer_dense(units = 16, 
                activation = "relu",
                kernel_regularizer = regularizer_l2(0.001)) %>% 
    layer_dropout(0.5) %>% 
    layer_dense(units = 8, 
                activation = "relu") %>% 
    layer_dropout(0.5) %>% 
    layer_dense(units = output_unit, activation = "sigmoid")
  
  model_rnn %>% compile(
    optimizer = optimizer,
    loss = "binary_crossentropy",
    metrics = c("acc"))
  
  model_rnn
}

#######################
# BIDIRECTIONAL RNN ###
#######################

model_bidirectional_rnn <- function(max_features,
                                    optimizer,
                                    loss = "binary_crossentropy",
                                    output_unit = 1){
  
  model_bidirectional <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_features, 
                    output_dim = 32) %>% 
    bidirectional(
      layer_lstm(units = 32)) %>% 
    layer_dense(units = 16, 
                activation = "relu",
                kernel_regularizer = regularizer_l2(0.001)) %>%
    layer_dropout(rate = 0.5) %>% 
    layer_dense(units = output_unit, activation = "sigmoid")
  
  model_bidirectional %>% compile(
    optimizer = optimizer,
    loss = loss,
    metrics = c("acc"))
  
  model_bidirectional
}

#######################
# LSTM ################
#######################

model_lstm <- function(max_features,
                       optimizer,
                       loss = "binary_crossentropy",
                       output_unit = 1){
  
  model_lstm <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_features, 
                    output_dim = 32) %>%
    layer_lstm(units = 32) %>%
    layer_dense(units = 16, 
                activation = "relu",
                kernel_regularizer = regularizer_l2(0.01)) %>% 
    layer_dropout(0.5) %>% 
    layer_dense(units = 8, 
                activation = "relu",
                kernel_regularizer = regularizer_l2(0.01)) %>% 
    layer_dropout(0.5) %>% 
    layer_dense(units = output_unit, activation = "sigmoid")
  
  model_lstm %>% compile(
    optimizer = optimizer,
    loss = loss,
    metrics = c("acc"))
  
  model_lstm 
}


###############################
# LSTM + GLOVE ################
###############################

model_lstm_glove <- function(maxlen,
                             max_words,
                             word_index,
                             embeddings_index,
                             optimizer,
                             loss = "binary_crossentropy",
                             output_unit = 1) {
  
  embedding_dim <- 100
  embedding_matrix <- array(0, c(max_words, embedding_dim))
  
  for (word in names(word_index)) {
    index <- word_index[[word]]
    if (index < max_words) {
      embedding_vector <- embeddings_index[[word]]
      if (!is.null(embedding_vector))
        embedding_matrix[index+1,] <- embedding_vector
    }
  }
  
  input <- layer_input(
    shape = list(NULL),
    dtype = "int32",
    name = "input"
  )
  
  encoded <- input %>% 
    layer_embedding(input_dim = max_words, 
                    output_dim = embedding_dim, 
                    input_length = maxlen,
                    name = "embedding_1") %>%
    layer_lstm(units = maxlen,
               dropout = 0.2,
               recurrent_dropout = 0.5,
               return_sequences = FALSE) 
  
  dense <- encoded %>% 
    layer_dropout(rate = 0.5) %>% 
    layer_dense(units = 32, activation = "relu") %>% 
    layer_dropout(rate = 0.5) %>% 
    layer_dense(units = output_unit, activation = "sigmoid")
  
  model_lstm_glove <- keras_model(input, dense)
  
  get_layer(model_lstm_glove, name = "embedding_1") %>%
    set_weights(list(embedding_matrix)) %>%
    freeze_weights()
  
  model_lstm_glove %>% compile(
    optimizer = optimizer,
    loss = loss,
    metrics = c("acc"))
  
  model_lstm_glove 
}


###############################
# GLOVE #######################
###############################

model_glove <- function(maxlen, 
                        max_words,
                        word_index, 
                        embeddings_index,
                        optimizer,
                        loss = "binary_crossentropy",
                        output_unit = 1){
  
  embedding_dim <- 100
  embedding_matrix <- array(0, c(max_words, embedding_dim))
  
  for (word in names(word_index)) {
    index <- word_index[[word]]
    if (index < max_words) {
      embedding_vector <- embeddings_index[[word]]
      if (!is.null(embedding_vector))
        embedding_matrix[index+1,] <- embedding_vector
    }
  }
  
  model_glove <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_words,
                    output_dim = embedding_dim,
                    input_length = maxlen,
                    name = "embedding") %>%
    layer_flatten() %>%
    layer_dropout(0.5) %>% 
    layer_dense(units = 32, activation = "relu") %>%
    layer_dropout(0.5) %>% 
    layer_dense(units = 16, activation = "relu") %>%
    layer_dropout(0.5) %>%
    layer_dense(units = output_unit, activation = "sigmoid")
  
  get_layer(model_glove, name = "embedding") %>%
    set_weights(list(embedding_matrix)) %>%
    freeze_weights()
  
  model_glove %>% compile(
    optimizer = "rmsprop",
    loss = loss,
    metrics = c("acc"))
  
  model_glove
}

