#####################################
# BACK-END COMPONENT ################
#####################################

server <- function(input, output, session){

  options(shiny.maxRequestSize= 50*1024^2)
  
  # PRINTING TEXT ########################
  
  output$header_text <- renderUI({
    tagList(
      h4(paste0(
        str_replace(toupper(input$model), "_", " ")))
    )
  })
  
  # PARAMETERS ########################
  
  observe({
    inFile_training <- input$file_training
    inFile_test <- input$file_test
    if(!is.null(inFile_training) & !is.null(inFile_test)){
      notes_training <- readr::read_csv(inFile_training$datapath)
      notes_test <- readr::read_csv(inFile_test$datapath)
      notes_full <- notes_training %>% 
        bind_rows(notes_test)
      
      # Derive Max words and Max Length
      max_words <- notes_full %>% 
        unnest_tokens(output = "word",
                      input = "Text",
                      token = "words") %>% 
        pull(word) %>% 
        unique() %>% 
        length()
      
      output$slider_max_words <- renderUI({
        sliderInput("max_words",
                    "Slider for Max Words", 
                    min   = 0, 
                    max   = max_words,
                    value = 0.8*max_words
        )})
      
      max_length <- notes_full %>% 
        mutate(sentence_length = sapply(strsplit(Text, " "), length)) %>% 
        pull(sentence_length) %>% 
        max()
      
      output$slider_max_length <- renderUI({
        sliderInput("max_length",
                    "Slider for Max Length", 
                    min   = 0, 
                    max   = max_length,
                    value = 0.8*max_length
        )})
      
    }
  })
  
  # TRAINING ##########################
  
  observeEvent(input$training, {
    
    inFile_training <- input$file_training
    if (is.null(inFile_training)){
      return(NULL)
    }
    notes_training <- readr::read_csv(inFile_training$datapath)
    
    inFile_test <- input$file_test
    if (is.null(inFile_test)){
      return(NULL)
    }
    notes_test <- readr::read_csv(inFile_test$datapath)
    
    writeLines("\n ############# DATA PROCESSING ############# \n")
    data <- data_processing_pipeline(notes_training,
                                     notes_test,
                                     max_words = input$max_words,
                                     max_length = input$max_length,
                                     train_validation_split = 0.8,
                                     exclude_stop_words = FALSE,
                                     seed = 2019)
    output_units <- data$y_train %>% dim() %>% .[2]
    
    if(input$model == "covnet") {
      writeLines("\n ############# MODEL TRAINING ############# \n")
      model <- model_deep_covnet(max_len = input$max_length,
                                 max_features = input$max_words,
                                 optimizer = "rmsprop",
                                 loss = "binary_crossentropy",
                                 output_unit = output_units)
      result <- train_validate_test(data,
                                    model = model,
                                    epoch = input$epoch,
                                    batch_size = input$batch_size,
                                    model_folder = input$model_folder)
      saveRDS(result, 
              file.path(input$result_folder, "result_multilabel_deep_covnet.rds"))
      
      
      output$model_plot <- renderPlot(
        plot(result$history)
      )
      
      writeLines("\n ############# METRICS GENERATION ############# \n")
      metrics <- generate_metrics(model, 
                                  data, 
                                  threshold = 0.5)
      metrics <- c(metrics$accuracy,
                   metrics$precision,
                   metrics$recall,
                   metrics$F1)
      
      optimal_threshold <- get_optimal_threshold(model, 
                                                 data, 
                                                 interval = 0.05)$optimal_threshold
      metrics_at_optimal_threshold <- generate_metrics(model, 
                                                       data, 
                                                       threshold = optimal_threshold, 
                                                       optimal_threshold = TRUE)
      metrics_at_optimal_threshold <- c(metrics_at_optimal_threshold$accuracy,
                                        metrics_at_optimal_threshold$precision,
                                        metrics_at_optimal_threshold$recall,
                                        metrics_at_optimal_threshold$F1)
      
      metrics_tibble <- tibble("Type" = c("Accuracy", "Precision", "Recall", "F1"),
                               "Metrics" = metrics,
                               "Metrics at Optimal Threshold" = metrics_at_optimal_threshold)
      output$optimal_threshold <- renderText(paste("The optimal threshold is:",
                                                   optimal_threshold))
      output$metrics_table <- renderTable(metrics_tibble)
    }
    
    
    if (input$model == "covnet_fasttext_embeddings"){
      
      writeLines("\n ############# EMBEDDINGS LOADING ############# \n")
      lines <- readLines(input$embedding_path)
      embeddings_index_fasttext_wikinews <- new.env(hash = TRUE, parent = emptyenv())
      
      for (i in 1:length(lines)) {
        cat(paste('\n ============', round((i/length(lines))*100, 2), '%', '============ \n'))
        line <- lines[[i]]
        values <- strsplit(line, " ")[[1]]
        word <- values[[1]]
        embeddings_index_fasttext_wikinews[[word]] <- as.double(values[-1])
      }
      
      writeLines("\n ############# MODEL TRAINING ############# \n")
      model <- model_deep_covnet_embeddings(max_len = input$max_length,
                                            max_features = input$max_words,
                                            embedding_dim = 300,
                                            embeddings_index = embeddings_index_fasttext_wikinews,
                                            word_index = data$word_index,
                                            optimizer = "rmsprop",
                                            loss = "binary_crossentropy",
                                            output_unit = output_units)
      
      result <- train_validate_test(data,
                                    model = model,
                                    epoch = input$epoch,
                                    batch_size = input$batch_size,
                                    model_folder = input$model_folder)
      
      saveRDS(result, file.path(input$result_folder, "result_multilabel_covnet_fasttext.rds"))
      
      output$model_plot <- renderPlot(
        plot(result$history)
      )
      
      writeLines("\n ############# METRICS GENERATION ############# \n")
      metrics <- generate_metrics(model, 
                                  data, 
                                  threshold = 0.5)
      metrics <- c(metrics$accuracy,
                   metrics$precision,
                   metrics$recall,
                   metrics$F1)
      
      optimal_threshold <- get_optimal_threshold(model, 
                                                 data, 
                                                 interval = 0.05)$optimal_threshold
      metrics_at_optimal_threshold <- generate_metrics(model, 
                                                       data, 
                                                       threshold = optimal_threshold, 
                                                       optimal_threshold = TRUE)
      metrics_at_optimal_threshold <- c(metrics_at_optimal_threshold$accuracy,
                                        metrics_at_optimal_threshold$precision,
                                        metrics_at_optimal_threshold$recall,
                                        metrics_at_optimal_threshold$F1)
      
      metrics_tibble <- tibble("Type" = c("Accuracy", "Precision", "Recall", "F1"),
                               "Metrics" = metrics,
                               "Metrics at Optimal Threshold" = metrics_at_optimal_threshold)
      output$optimal_threshold <- renderText(paste("The optimal threshold is:",
                                                   optimal_threshold))
      output$metrics_table <- renderTable(metrics_tibble)
    
    }
    
    if(input$model == "covnet_bidirectional_lstm") {
      
      writeLines("\n ############# MODEL TRAINING ############# \n")
      model <- model_deep_covnet_blstm(max_len = input$max_length,
                                       max_features = input$max_words,
                                       optimizer = "rmsprop",
                                       loss = "binary_crossentropy",
                                       output_unit = output_units)
      
      result <- train_validate_test(data,
                                    model = model,
                                    epoch = input$epoch,
                                    batch_size = input$batch_size,
                                    model_folder = input$model_folder)
      
      saveRDS(result, file.path(input$result_folder, "result_multilabel_covnet_blstm.rds"))
      
      output$model_plot <- renderPlot(
        plot(result$history)
      )
      
      writeLines("\n ############# METRICS GENERATION ############# \n")
      metrics <- generate_metrics(model, 
                                  data, 
                                  threshold = 0.5)
      metrics <- c(metrics$accuracy,
                   metrics$precision,
                   metrics$recall,
                   metrics$F1)
      
      optimal_threshold <- get_optimal_threshold(model, 
                                                 data, 
                                                 interval = 0.05)$optimal_threshold
      metrics_at_optimal_threshold <- generate_metrics(model, 
                                                       data, 
                                                       threshold = optimal_threshold, 
                                                       optimal_threshold = TRUE)
      metrics_at_optimal_threshold <- c(metrics_at_optimal_threshold$accuracy,
                                        metrics_at_optimal_threshold$precision,
                                        metrics_at_optimal_threshold$recall,
                                        metrics_at_optimal_threshold$F1)
      
      metrics_tibble <- tibble("Type" = c("Accuracy", "Precision", "Recall", "F1"),
                               "Metrics" = metrics,
                               "Metrics at Optimal Threshold" = metrics_at_optimal_threshold)
      output$optimal_threshold <- renderText(paste("The optimal threshold is:",
                                                   optimal_threshold))
      output$metrics_table <- renderTable(metrics_tibble)
      
    }
    
  })
  
  
  # STOP SHINY APP ########################
  session$onSessionEnded(function() {
    stopApp()
  })
  
}