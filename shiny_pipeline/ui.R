
#####################################
# FRONT-END COMPONENT ###############
#####################################

ui <- fluidPage(
  shinyjs::useShinyjs(),
  titlePanel("Deep Learning Pipeline"),
  sidebarLayout(
    sidebarPanel(
      
      # Input Folders
      textInput(inputId = "model_folder",
                label = "Please input model folder:"),
      textInput(inputId = "result_folder",
                label = "Please input result folder:"),
      textInput(inputId = "embedding_path",
                label = "Please input path to embeddings:"),
      
      # Input: Select a file ----
      fileInput("file_training", 
                "Choose CSV File for Training:",
                multiple = FALSE,
                accept = c("text/csv",
                           "text/comma-separated-values,text/plain",
                           ".csv")),
      # Input: Select a file ----
      fileInput("file_test", 
                "Choose CSV File for Test:",
                multiple = FALSE,
                accept = c("text/csv",
                           "text/comma-separated-values,text/plain",
                           ".csv")),
      
      # Input: Select a model ----
      selectInput(inputId = "model",
                  label = "Models you want to view:",
                  choices = models,
                  selected = "covnet"),
      
      # Parameter: ----
      numericInput(inputId = "epoch",
                   label = "Number of Epochs:",
                   value = 20,
                   min = 0,
                   max = 500,
                   step = 1),
      numericInput(inputId = "batch_size",
                   label = "Batch Size:",
                   value = 64,
                   min = 0,
                   max = 64*10,
                   step = 1),
      uiOutput("slider_max_words"),
      uiOutput("slider_max_length"),
      
      # Training Button ----
      actionButton(input = "training",
                   label = "Training",
                   icon = icon("paper-plane"))
      
      ),
    mainPanel(
      uiOutput("header_text"),
      hr(),
      plotOutput("model_plot"),
      hr(),
      textOutput("optimal_threshold"),
      tableOutput("metrics_table")
    )
  )
)