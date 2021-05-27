
##############################################
# LOADING PACKAGES ###########################
##############################################

library(keras)
library(dplyr)
library(shiny)
library(shinyjs)
library(stringr)

##############################################
# LOADING RESULTS ############################
##############################################

result_dir <- "/Users/nguyenh/Desktop/cumc/deep_learning_for_nlp/final_result_folder"
file_names <- list.files(result_dir)
results <- lapply(file_names, function(x) readRDS(file.path(result_dir, x)))
history <- lapply(results, function(x) x$history)
accuracy <- lapply(results, function(x) x$result$acc) %>% unlist()
loss <- lapply(results, function(x) x$result$loss) %>% unlist()

names_history <- file_names %>% 
  str_replace_all(".rds", "") %>% 
  str_replace_all("result_", "")

##############################################
# BUILDING DATAFRAME #########################
##############################################

data <- tibble("model_name" = names_history,
               "history" = history,
               "accuracy" = accuracy,
               "loss" = loss) %>% 
  mutate("symptom_type" = case_when(
    str_detect(model_name, "anorexia") ~ "anorexia",
    str_detect(model_name, "chest.pain") ~ "chest.pain",
    str_detect(model_name, "confusion") ~ "confusion",
    str_detect(model_name, "cough") ~ "cough",
    str_detect(model_name, "dizziness") ~ "dizziness",
    str_detect(model_name, "dyspnea") ~ "dyspnea",
    str_detect(model_name, "fatique") ~ "fatique",
    str_detect(model_name, "multilabel_all") ~ "multilabel_all",
    str_detect(model_name, "multilabel") ~ "multilabel",
    str_detect(model_name, "nausea") ~ "nausea",
    str_detect(model_name, "palpitation") ~ "palpitation",
    str_detect(model_name, "peripheral.edema") ~ "peripheral.edema",
    str_detect(model_name, "weight.change") ~ "weight.change",
    )) %>% 
  mutate("model_type" = case_when(
    str_detect(model_name, "deep_covnet_blstm") ~ "deep_covnet_blstm",
    str_detect(model_name, "deep_covnet_fasttext") ~ "deep_covnet_fasttext",
    str_detect(model_name, "deep_covnet") ~ "deep_covnet"
  ))

##############################################
# VARIABLES ##################################
##############################################

symptoms <- c("anorexia",
              "chest.pain",
              "confusion",
              "cough",
              "dizziness",
              "dyspnea",
              "fatique",
              "multilabel_all",
              "multilabel",
              "nausea",
              "palpitation",
              "peripheral.edema",
              "weight.change")

models <- c("deep_covnet_blstm",
            "deep_covnet_fasttext",
            "deep_covnet")

#####################################################
# HELPER FUNCTIONS ##################################
#####################################################





