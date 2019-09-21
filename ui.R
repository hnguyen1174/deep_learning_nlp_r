
#####################################
# FRONT-END COMPONENT ###############
#####################################

ui <- fluidPage(
  shinyjs::useShinyjs(),
  titlePanel("Deep Learning Result Visualization"),
  sidebarLayout(
    sidebarPanel(
      selectInput(inputId = "symptom",
                  label = "Symptoms you want to view",
                  choices = symptoms,
                  selected = "dyspnea"),
      selectInput(inputId = "model",
                  label = "Models you want to view",
                  choices = models,
                  selected = "deep_covnet")),
      mainPanel(
        uiOutput("header_text"),
        hr(),
        plotOutput("plot")
      )
  )
)