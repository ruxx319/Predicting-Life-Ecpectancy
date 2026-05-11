# Libraries
library(tidyverse)
library(rvest)
library(tidymodels)
library(janitor)
library(countrycode)
library(glmnet)

# Scraping Function
get_standardized_wiki <- function(url, table_idx) {
  raw_table <- read_html(url) %>% 
    html_nodes(".wikitable") %>% 
    .[[table_idx]] %>% 
    html_table(fill = TRUE) %>%
    clean_names()
  
  df <- raw_table %>%
    mutate(across(everything(), ~str_remove_all(as.character(.), "\\[.*?\\]"))) %>%
    mutate(iso_code = countrycode(.[[1]], origin = 'country.name', destination = 'iso3c')) %>%
    filter(!is.na(iso_code))
  
  return(df)
}

# Scrapping
le_raw     <- get_standardized_wiki("https://en.wikipedia.org/wiki/List_of_countries_by_life_expectancy", 1)
gdp_raw    <- get_standardized_wiki("https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)_per_capita", 1)
pop_raw    <- get_standardized_wiki("https://en.wikipedia.org/wiki/List_of_countries_by_population_growth_rate", 1)
health_raw <- get_standardized_wiki("https://en.wikipedia.org/wiki/List_of_countries_by_total_health_expenditure_per_capita", 2)
youth_lit_raw <- get_standardized_wiki("https://en.wikipedia.org/wiki/List_of_countries_by_youth_literacy_rate", 1)

# Data Preprocessing & Joining
le_clean <- le_raw %>%
  select(iso_code, country_name = 1, le_val = 2) %>% 
  mutate(le_val = parse_number(le_val))

gdp_clean <- gdp_raw %>% 
  select(iso_code, gdp_capita = 4) %>% 
  mutate(gdp_capita = parse_number(gdp_capita))

pop_clean <- pop_raw %>% 
  select(iso_code, pop_growth = 2) %>%
  mutate(pop_growth = parse_number(pop_growth))

health_clean <- health_raw %>% 
  select(iso_code, health_exp = 2) %>% 
  mutate(health_exp = parse_number(health_exp))

youth_lit_clean <- youth_lit_raw %>%
  select(iso_code, youth_lit = 4) %>%
  mutate(youth_lit = parse_number(youth_lit))


final_df <- le_clean %>%
  left_join(gdp_clean, by = "iso_code") %>%
  left_join(pop_clean, by = "iso_code") %>%
  left_join(health_clean, by = "iso_code") %>%
  left_join(youth_lit_clean, by = "iso_code") %>%
  mutate(is_high_le = factor(ifelse(le_val >= median(le_val, na.rm=T), "High", "Low"), 
                             levels = c("High", "Low")))

# Plots
p1 <- ggplot(final_df, aes(x = le_val)) +
  geom_histogram(bins = 20, fill = "#2c3e50", color = "white") +
  theme_minimal() +
  labs(title = "Distribution of Life Expectancy", x = "Years", y = "Number of Countries")

p2 <- ggplot(final_df, aes(x = log(gdp_capita), y = le_val, color = is_high_le)) +
  geom_point(size = 3, alpha = 0.6) +
  scale_color_manual(values = c("High" = "#2980b9", "Low" = "#c0392b")) +
  theme_minimal() +
  labs(title = "Log GDP vs. Life Expectancy", x = "Log(GDP per Capita)", y = "Life Expectancy (Years)")

p3 <- final_df %>% 
  arrange(desc(health_exp)) %>% 
  head(10) %>%
  ggplot(aes(x = reorder(country_name, health_exp), y = health_exp, fill = health_exp)) +
  geom_col() + 
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 10 Health Spenders", x = NULL, y = "Health Exp (% of GDP)")

p4 <-ggplot(final_df, aes(is_high_le, youth_lit, fill = is_high_le)) +
  geom_violin(trim = FALSE, alpha = 0.7) +
  geom_jitter(width = 0.1, alpha = 0.4, size = 1) +
  theme_minimal() +
  labs(
    title = "Youth Literacy Rate Distribution by Life Expectancy Group",
    x = "Life Expectancy Group",
    y = "Youth Literacy Rate (%)"
  )

print(p1);
print(p2);
print(p3);
print(p4);


#Feature selection
anova_results <- tibble(
  feature = c("gdp_capita", "pop_growth", "health_exp", "youth_lit"),
  p_value = c(
    summary(aov(gdp_capita ~ is_high_le, final_df))[[1]][["Pr(>F)"]][1],
    summary(aov(pop_growth ~ is_high_le, final_df))[[1]][["Pr(>F)"]][1],
    summary(aov(health_exp ~ is_high_le, final_df))[[1]][["Pr(>F)"]][1],
    summary(aov(youth_lit ~ is_high_le, final_df))[[1]][["Pr(>F)"]][1]
  )
)

anova_results %>%
  arrange(p_value)


# Modeling
set.seed(42)
data_split <- initial_split(final_df, prop = 0.8, strata = is_high_le)
train_data <- training(data_split)
test_data  <- testing(data_split)

model_recipe <- recipe(is_high_le ~ gdp_capita + pop_growth + health_exp+ youth_lit, data = train_data) %>%
  step_impute_median(all_numeric_predictors()) %>% 
  step_log(gdp_capita) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors())

lr_spec <- logistic_reg(penalty = 0.1, mixture = 1) %>% set_engine("glmnet")
final_wf <- workflow() %>% add_recipe(model_recipe) %>% add_model(lr_spec)
model_fit <- final_wf %>% fit(data = train_data)

# Evaluation
results <- test_data %>%
  bind_cols(predict(model_fit, test_data)) %>%
  bind_cols(predict(model_fit, test_data, type = "prob"))

#Confusion matrix
cm_df <- conf_mat(
  results,
  truth = is_high_le,
  estimate = .pred_class
)$table %>%
  as_tibble()

cm_prop <- cm_df %>%
  group_by(Truth) %>%
  mutate(prop = n / sum(n))

p9 <-ggplot(cm_prop, aes(x = Prediction, y = Truth, fill = prop)) +
  geom_tile(color = "white") +
  geom_text(aes(label = scales::percent(prop, accuracy = 1)), size = 5) +
  scale_fill_gradient(low = "#ecf0f1", high = "#27ae60") +
  theme_minimal() +
  labs(
    title = "Normalized Confusion Matrix (Row-wise)",
    x = "Predicted Class",
    y = "True Class",
    fill = "Proportion"
  )
print(p9);

# Results
cat("\n--- Final Model Metrics ---\n")
print(results %>% my_metrics(truth = is_high_le, estimate = .pred_class))

