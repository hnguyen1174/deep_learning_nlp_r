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
      h4(paste0(toupper(input$model), " FOR ", toupper(input$symptom)))
    )
  })
}