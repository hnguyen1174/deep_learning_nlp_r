#####################################
# BACK-END COMPONENT ################
#####################################

server <- function(input, output, session){
  
  # PLOTTING ########################
  
  output$plot <- renderPlot(
    data %>% 
      filter(model_type == input$model,
             symptom_type == input$symptom) %>% 
      pull(history) %>% 
      .[[1]] %>%
      plot()
  )
  
  # PRINTING TEXT ########################
  
  output$header_text <- renderUI({
    tagList(
      h4(paste0(
        str_replace(toupper(input$model), "_", " "), 
        " MODEL FOR ", 
        toupper(input$symptom)))
    )
  })
  
  # PRINTING ACCURACY/LOSS ########################
  
  output$accuracy <- renderUI({
    data_reactive <- data %>% 
      filter(model_type == input$model,
             symptom_type == input$symptom)
    tagList(
      HTML(paste0("<b>Accuracy: </b>", round(data_reactive$accuracy, 2))),
      br(),
      HTML(paste0("<b>Loss: </b>", round(data_reactive$loss, 2)))
    )
  })
  
  
  
  
}