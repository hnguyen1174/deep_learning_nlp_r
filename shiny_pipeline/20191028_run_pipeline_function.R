
data_path <-             ### PATH TO DATA (CSV FILE) #########################
model_folder <-          ### A FOLDER THAT THE MODELS ARE SAVED TO ###########
result_folder <-         ### A FOLDER THAT THE RESULT OBJECTS ARE SAVED TO ###
embedding_path <-        ### PATH TO EMBEDDING (.VEC FILE) ###################
helper_functions_path <- ### PATH TO HELPER FUNCTION FILE ####################

# EXAMPLE:
data_path <- "/Users/nguyenh/Desktop/stuff/cumc/deep_learning_for_nlp/data/test_data.csv"
model_folder <- "/Users/nguyenh/Desktop/stuff/cumc/deep_learning_for_nlp/final_model_folder"
result_folder <- "/Users/nguyenh/Desktop/stuff/cumc/deep_learning_for_nlp/final_result_folder"
embedding_path <- "/Users/nguyenh/Desktop/stuff/cumc/deep_learning_for_nlp/data/wiki-news-300d-1M.vec"
helper_functions_path <- "/Users/nguyenh/Desktop/stuff/cumc/deep_learning_for_nlp/20190907_helper_functions.R"

run_deep_learning_pipeline <- function(data_path,
                                       model_folder,
                                       result_folder,
                                       model_choice,
                                       embedding_path,
                                       helper_functions_path,
                                       data_train_test_split = 0.8,
                                       data_train_validation_split = 0.8,
                                       data_exclude_stop_words = FALSE,
                                       epoch = 20,
                                       batch_size = 64) {
  
  writeLines("\n ############# SETTING UP ############# \n")
  # Installing packages
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
  
  source(helper_functions_path)
  
  writeLines("\n ############# DATA PROCESSING ############# \n")
  data <- data_processing_pipeline(data_path,
                                   train_test_split = data_train_test_split,
                                   train_validation_split = data_train_validation_split,
                                   exclude_stop_words = data_exclude_stop_words,
                                   seed = 2019)
  max_length <- data$max_length
  max_words <- data$max_words
  output_units <- data$y_train %>% dim() %>% .[2]
  

  if(model_choice == "covnet"){
    writeLines("\n ############# MODEL TRAINING ############# \n")
    model <- model_deep_covnet(max_len = max_length,
                               max_features = max_words,
                               optimizer = "rmsprop",
                               loss = "binary_crossentropy",
                               output_unit = output_units)
    
    result <- train_validate_test(data,
                                  model = model,
                                  epoch = epoch,
                                  batch_size = batch_size,
                                  model_folder = model_folder)
    saveRDS(result, 
            file.path(result_folder, "result_multilabel_deep_covnet.rds"))
    
    writeLines("\n ############# METRICS GENERATION ############# \n")
    metrics <- generate_metrics(model, data, threshold = 0.5)
    optimal_threshold <- get_optimal_threshold(model, data, interval = 0.05)$optimal_threshold
    metrics_at_optimal_threshold <- generate_metrics(model, 
                                                     data, 
                                                     threshold = optimal_threshold, 
                                                     optimal_threshold = TRUE)
  
    } else if (model_choice == "covnet_fasttext_embeddings"){
    
    writeLines("\n ############# EMBEDDINGS LOADING ############# \n")
    lines <- readLines(embedding_path)
    embeddings_index_fasttext_wikinews <- new.env(hash = TRUE, parent = emptyenv())
    
    for (i in 1:length(lines)) {
      cat(paste('\n ============', round((i/length(lines))*100, 2), '%', '============ \n'))
      line <- lines[[i]]
      values <- strsplit(line, " ")[[1]]
      word <- values[[1]]
      embeddings_index_fasttext_wikinews[[word]] <- as.double(values[-1])
    }
    
    writeLines("\n ############# MODEL TRAINING ############# \n")
    model <- model_deep_covnet_embeddings(max_len = max_length,
                                          max_features = max_words,
                                          embedding_dim = 300,
                                          embeddings_index = embeddings_index_fasttext_wikinews,
                                          word_index = data$word_index,
                                          optimizer = "rmsprop",
                                          loss = "binary_crossentropy",
                                          output_unit = output_units)
    result <- train_validate_test(data,
                                  model = model,
                                  epoch = epoch,
                                  batch_size = batch_size,
                                  model_folder = model_folder)
    saveRDS(result, file.path(result_folder, "result_multilabel_covnet_fasttext.rds"))
    
    writeLines("\n ############# METRICS GENERATION ############# \n")
    metrics <- generate_metrics(model, data, threshold = 0.5)
    optimal_threshold <- get_optimal_threshold(model, data, interval = 0.05)$optimal_threshold
    metrics_at_optimal_threshold <- generate_metrics(model, 
                                                     data, 
                                                     threshold = optimal_threshold, 
                                                     optimal_threshold = TRUE)
    
    } else if(model_choice == "covnet_bidirectional_lstm") {
      
      writeLines("\n ############# MODEL TRAINING ############# \n")
      model <- model_deep_covnet_blstm(max_len = max_length,
                                       max_features = max_words,
                                       optimizer = "rmsprop",
                                       loss = "binary_crossentropy",
                                       output_unit = output_units)
      result <- train_validate_test(data,
                                    model = model,
                                    epoch = epoch,
                                    batch_size = batch_size,
                                    model_folder = model_folder)
      saveRDS(result, file.path(result_folder, "result_multilabel_covnet_blstm.rds"))
      
      writeLines("\n ############# METRICS GENERATION ############# \n")
      metrics <- generate_metrics(model, data, threshold = 0.5)
      optimal_threshold <- get_optimal_threshold(model, data, interval = 0.05)$optimal_threshold
      metrics_at_optimal_threshold <- generate_metrics(model, 
                                                       data, 
                                                       threshold = optimal_threshold, 
                                                       optimal_threshold = TRUE)
    }
  
  return(list('model' = model,
              'result' = result,
              'metrics' = metrics,
              'metrics_at_optimal_threshold' = metrics_at_optimal_threshold))
}

# RUN THESE CODE LINES BELOW TO GET THE METRICS FOR EACH MODEL

##########################
# GENERATE METRICS #######
##########################

# CONVOLUTED NEURAL NETWORK

metrics <- run_deep_learning_pipeline(data_path = data_path,
                                      model_folder = model_folder,
                                      result_folder = result_folder,
                                      helper_functions_path = helper_functions_path,
                                      model_choice = "covnet",
                                      embedding_path = embedding_path)$metrics

# CONVOLUTED NEURAL NETWORK + BIDIRECTIONAL LSTM

metrics <- run_deep_learning_pipeline(data_path = data_path,
                                      model_folder = model_folder,
                                      result_folder = result_folder,
                                      helper_functions_path = helper_functions_path,
                                      model_choice = "covnet_bidirectional_lstm",
                                      embedding_path = embedding_path)$metrics

# CONVOLUTED NEURAL NETWORK + FASTTEXT EMBEDDINGS

metrics <- run_deep_learning_pipeline(data_path = data_path,
                                      model_folder = model_folder,
                                      result_folder = result_folder,
                                      helper_functions_path = helper_functions_path,
                                      model_choice = "covnet_fasttext_embeddings",
                                      embedding_path = embedding_path)$metrics

