########## COFFEE DATASET PREPROCESSING ######

## SOURCES: 
# https://github.com/jldbc/coffee-quality-database/tree/master/data
# https://www.kaggle.com/code/reminho/coffee-quality-score-prediction-using-tree-models


# required packages
library(data.table)
library(dplyr)
library(forcats)
library(stringr)

# readin
coffee = data.table(read.csv("real_world_example/raw_data/arabica_data_cleaned.csv", sep = ",", header = TRUE, row.names = "X"))
dim(coffee)

## Preproocessing
# blanks to NAs
indx2 <- which(sapply(coffee, is.character)) 
for (j in indx2) set(coffee, i = grep("^$|^ $", coffee[[j]]), j = j, value = NA_character_)

# column names to lower
setnames(coffee,tolower(names(coffee)))

# keep everything but quality measures and (presumably) non-informative columns
coffee <- coffee %>%
  # select everything except quality measures
  select(country.of.origin, harvest.year, variety:moisture, color, altitude_mean_meters)

# summary
dim(coffee)

# omit all NAs in dataset
coffee = na.omit(coffee)
dim(coffee)

# replace weird year format
coffee <- coffee %>%
  mutate(harvest.year = str_replace_all(harvest.year, regex('^\\d+\\s?/\\s?(\\d+)'), regex('\\1'))) %>% 
  mutate(harvest.year = str_replace_all(harvest.year, regex('^\\d+\\s?-\\s?(\\d+)'), regex('\\1'))) 

# table(coffee$harvest.year)
# table(coffee$processing.method)
# table(coffee$color)
# table(coffee$variety)

# bin categories of variety
coffee = coffee %>%
  mutate(variety = fct_lump_n(variety, 5))

# some "outliers" in altitute_mean_meters?
plot(density(coffee$altitude_mean_meters))
sort(coffee$altitude_mean_meters)
coffee = coffee[altitude_mean_meters < 8000,]
dim(coffee)

# binarize target (total.cup.points --> quality) 
hist(coffee$total.cup.points)
median(coffee$total.cup.points) # 82.42 

# new target: quality - label "good" if total.cup.points > median(coffee$total.cup.points) 
coffee$quality = ifelse(coffee$total.cup.points > median(coffee$total.cup.points), "good", "bad")
coffee$total.cup.points = NULL

write.csv(coffee, "real_world_example/coffee_data_cleaned.csv", row.names = F)
