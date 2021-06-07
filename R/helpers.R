#' Labeling functions
#'
#' @param x a column to label
#'
#' @return labelled column
labeling <- function(x) {
  if_else(x > 0, 1, 0)
}