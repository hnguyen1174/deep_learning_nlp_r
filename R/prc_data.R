#' Process Gold Standard Dataset
#'
#' @param raw_data raw gold-standard data
#'
#' @return processed gold-standard data
#' @export 
prc_gold_standard <- function(raw_data) {
  
  data <- raw_data %>% 
    select(-X1) %>% 
    mutate(report_head = str_detect(Note, '^admission date'))
  
  report_head_gold_standard <- get_report_heads(data)
  
  gold_standard_sentence_level <- data %>% 
    
    # Joint with report_head dataframe, report_no show which report each sentence belongs to
    left_join(report_head_gold_standard, by = 'Note') %>% 
    mutate(report_no = na.locf(report_no),
           # remove all numbers
           Note = removeNumbers(Note)) %>% 
    # Remove lines with no sentences
    filter(Note != '') %>% 
    
    # Remove unnecessary whitespaces
    mutate(note_processed = str_squish(Note)) %>% 
    transmute(note_processed,
              cat1 = `Category 1`,
              cat2 = `Category 2`,
              cat3 = `Category 3`,
              cat4 = `Category 4`,
              cat5 = `Category 5`,
              cat6 = `Category 6`,
              cat7 = `Category 7`,
              report_head, report_no) %>% 
    
    # Create 14 label columns (one-hot encoding)
    transmute(note_processed,
              report_head,
              report_no,
              dyspnea = if_else((cat1 == "Dyspnea")|(cat2 == "Dyspnea"), 1, 0),
              confusions = if_else((cat1 == "Confusion")|(cat2 == "Confusion"), 1, 0),
              fatique = if_else((cat1 == "Fatigue")|(cat2 == "Fatigue"), 1, 0),
              cough = if_else((cat1 == "Cough")|(cat2 == "Cough"), 1, 0),
              peripheral.edema = if_else((cat1 == "peripheral.edema")|(cat2 == "peripheral.edema"), 1, 0),
              anorexia = if_else((cat1 == "Anorexia")|(cat2 == "Anorexia"),1,0),
              weight.change = if_else((cat1 == "Weight.loss.or.weight.gain")|(cat2 == "Weight.loss.or.weight.gain"), 1, 0),
              nausea = if_else((cat1 == "Nausea")|(cat2 == "Nausea"), 1, 0),
              chest.pain = if_else((cat1 == "Chest.pain")|(cat2 == "Chest.pain"), 1, 0),
              palpitation = if_else((cat1 == "Palpitation")|(cat2 == "Palpitation"), 1, 0),
              dizziness = if_else((cat1 == "Dizziness")|(cat2 == "Dizziness"), 1, 0)) %>% 
    
    # Replace NA with 0
    replace(is.na(.), 0) %>% 
    mutate(with_labels = if_else(rowSums(.[4:14]) > 0, TRUE, FALSE))

  note_level_labels <- gold_standard_sentence_level %>% 
    group_by(report_no) %>% 
    summarize_if(is.numeric, list(sum)) %>% 
    mutate_at(vars(-report_no), labeling)
  
  gold_standard_note_level <- get_note_level_data(gold_standard_sentence_level, 
                                                  note_level_labels)
  
  list(
    'gold_standard_sentence_level' = gold_standard_sentence_level,
    'gold_standard_note_level' = gold_standard_note_level
  )
}


#' Process Training Dataset
#'
#' @param raw_data raw training data
#'
#' @return processed training data
#' @export 
prc_training_data <- function(raw_data) {
  
  data <- raw_data %>% 
    # X1 is the index column, deselect this column
    select(-X1) %>% 
    # report_head indicates the start of a note
    mutate(report_head = str_detect(Note, '^admission date'))
  
  # "report_head" variable contains the column report_no, a unique identifier for each report
  # the report_head dataframe contain report_no, a unique identifier for each report
  report_head_training <- get_report_heads(data)
  
  training_sentence_level <- data %>% 
    # joint with report_head dataframe, report_no show which report each sentence belongs to
    left_join(report_head_training, by = 'Note') %>% 
    mutate(report_no = na.locf(report_no),
           # remove all numbers
           Note = removeNumbers(Note)) %>% 
    # remove lines with no sentences
    mutate(note_processed = str_squish(Note)) %>% 
    filter(note_processed != '') %>% 
    # remove unnecessary whitespaces
    transmute(note_processed,
              report_head,
              report_no,
              dyspnea = `Dyspnea (# of simclins)`,
              confusions = `Confusion (# of simclins)`,
              fatique = `Fatigue (# of simclins)`,
              cough = `Cough (# of simclins)`,
              peripheral.edema = `peripheral.edema (# of simclins)`,
              anorexia = `Anorexia.decreased.appetite (# of simclins)`,
              weight.change = `Weight.loss.or.weight.gain (# of simclins)`,
              nausea = `Nausea (# of simclins)`,
              chest.pain = `Chest.pain (# of simclins)`,
              palpitation = `Palpitation (# of simclins)`,
              dizziness = `Dizziness (# of simclins)`) %>% 
    # replace NA with 0
    replace(is.na(.), 0) %>% 
    mutate_at(vars(-c(note_processed,
                      report_head,
                      report_no)), labeling) %>% 
    mutate(with_labels = if_else(rowSums(.[4:14]) > 0, TRUE, FALSE))
  
  training_note_level_labels <- training_sentence_level %>% 
    group_by(report_no) %>% 
    summarize_if(is.numeric, list(sum)) %>% 
    mutate_at(vars(-report_no), labeling)
  
  # Note level data
  training_note_level <- get_note_level_data(training_sentence_level, 
                                             training_note_level_labels)
  
  list(
    'training_sentence_level' = training_sentence_level,
    'training_note_level' = training_note_level
  )
  
}

#' Get Report Head (the first sentence in a report)
#'
#' @param raw_data raw data
#'
#' @return report head
get_report_heads <- function(raw_data) {
  
  raw_data %>% 
    filter(report_head) %>% 
    select(Note, report_head) %>% 
    mutate(report_no = row_number()) %>% 
    select(-report_head)
}

#' Get Note-level Data
#'
#' @param sentence_level_data sentence-level data
#' @param note_level_label note-level label
#'
#' @return note-level data
get_note_level_data <- function(sentence_level_data, note_level_label) {
  
  note_level_data <- sentence_level_data %>% 
    group_by(report_no) %>% 
    summarize(note_processed = paste(note_processed, collapse = ' ')) %>% 
    left_join(note_level_label, by = 'report_no')
  
  note_level_data
}