# import libraries and data
library(dplyr)
library(tidyverse)
library(tidyr)
library(forcats)
library(tibble)
library(ggplot2)
library(gridExtra)
library(reshape2)
library(ggh4x)
library(patchwork)
library(ChemmineR)

data <- read.csv(file.path("raw_data", "data_20260611.csv"))

# reorganize biomimox data
data$pH <- as.factor(data$pH)

# rescale percent so adds up to 100% (i.e. excluding the drug amount)
data <- data %>%
  group_by(name, pH) %>%
  mutate(
    .grp_total = sum(percent[biomim == 1 & conv == 0], na.rm = TRUE),
    percent_metab = if_else(
      biomim == 1 & conv == 0 & .grp_total > 0,
      100 * percent / .grp_total,
      NA_real_
    )
  ) %>%
  ungroup() %>%
  select(-.grp_total)

data_conv <- data %>%
  filter(conv == 1) %>%
  select(c(name, smiles, pH, biomim, percent))

data_metab_full <- data %>%
  filter(conv == 0, biomim == 1) %>%
  select(-c(metabolite, metabolite_gen, invivo, biomim, conv, percent)) %>%
  rename(percent = percent_metab)

data_metab_filt <- data %>%
  filter(conv == 0, biomim == 1,
         percent > 5, percent_metab > 10) %>% #remove minor metabolites
  select(-c(metabolite, metabolite_gen, invivo, biomim, conv, percent)) %>%
  rename(percent = percent_metab)

# distribution of biomimetic and spontaneous oxidation
# 1) define bins
breaks  <- c(-Inf, 33.3333, 66.6667, Inf)
labels  <- c("Low (<33%)", "Medium (33–67%)", "High (>67%)")

# 2) prep + summarize
summary_df <- data_conv %>%
  # keep valid percents
  filter(!is.na(percent), percent >= 0, percent <= 100) %>%
  mutate(
    band   = cut(percent, breaks = breaks, labels = labels, right = FALSE),
    biomim = as.integer(biomim)   # ensure 0/1 as integers for facet labels
  ) %>%
  count(biomim, pH, band, name = "n") %>%
  group_by(biomim, pH) %>%
  mutate(total = sum(n), frac = if_else(total > 0, n / total, 0), lbl = scales::percent(frac, accuracy = 1)) %>%
  ungroup()
summary_df_lab <- summary_df %>%
  group_by(biomim, pH) %>%
  arrange(desc(band), .by_group = TRUE) %>%
  mutate(
    yshift = frac / 2,
    ystart = cumsum(frac),
    ylab   = ystart - yshift,
    close  = any(diff(100 * ylab) < 10),
    ylab   = ylab + ifelse(close, c(0, 0.05, 0), 0)
  ) %>%
  ungroup()

# 3) pie charts (stacked bar + coord_polar), faceted by biomim and pH
custom_colors <- c(
  "Low (<33%)"     = "#6baed6",  # blue
  "Medium (33–67%)"= "#fd8d3c",  # orange
  "High (>67%)"    = "#31a354"   # green
)

plot_conv <- ggplot(summary_df_lab, aes(x = "", y = frac, fill = band)) +
  geom_col(width = 1, color = "white") +
  coord_polar(theta = "y") +
  geom_text(
    aes(y = ylab, label = ifelse(n == 0, "", lbl)),
    size = 4
  ) +
  facet_grid(
    biomim ~ pH,
    labeller = labeller(
      biomim = c(`0` = "non-biomimetic", `1` = "biomimetic")
    )
  ) +
  scale_fill_manual(values = custom_colors, name = "Percent band") +
  guides(fill = guide_legend(title = "Percent conversion")) +
  theme_void() +
  theme(
    plot.title      = element_text(hjust = 0.5, size = 15, face = "bold"),
    plot.subtitle   = element_text(hjust = 0.5, size = 13),
    strip.text      = element_text(size = 12),
    legend.position = "bottom"
  ) +
  labs(
    title    = "Distribution of oxidation conversion rates",
    subtitle = "Faceted by pH"
  )

# effect of pH on metabolite pathway distribution
# 1) identify category columns
class_cols <- setdiff(
  colnames(data_metab_filt),
  c("name", "smiles", "pH",
    "H_prox", "X_exc",
    "sp3_sp2", "sp2_sp", "R_Ar", "Q",
    "uncat", "unk", "percent")
)

# 2) Long format -> keep rows where the flag is 1 -> count by pH & category
plot_df <- data_metab_filt %>%
  pivot_longer(all_of(class_cols), names_to = "category", values_to = "flag") %>%
  filter(flag == 1) %>%
  count(pH, category, name = "n") %>%
  group_by(pH) %>%
  mutate(category = fct_reorder(category, n, .desc = TRUE)) %>%
  ungroup()

# 3) Bar charts faceted by pH (one panel per pH)
ph_levels <- unique(plot_df$pH)
facet_colors <- setNames(RColorBrewer::brewer.pal(length(ph_levels), "Set3"), ph_levels)

# define one common category order
cat_levels <- plot_df %>%
  filter(pH == "2.7") %>%
  group_by(category) %>%
  summarise(total_n = sum(n), .groups = "drop") %>%
  arrange(desc(total_n)) %>%
  pull(category)

plot_df2 <- plot_df %>%
  mutate(category = factor(category, levels = cat_levels))

plot_cat_main <- ggplot(plot_df2, aes(x = n, y = category)) +
  geom_col(fill = "#4682B4") +
  facet_wrap2(
    ~ pH,
    scales = "free_x",
    nrow   = 1,
    strip  = strip_themed(
      background_x = lapply(
        facet_colors,
        function(col) element_rect(fill = col, color = NA)
      )
    )
  ) +
  labs(
    title = "Reaction category frequency by pH",
    x     = "Count",
    y     = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title       = element_text(hjust = 0.5, face = "bold"),
    axis.text.y      = element_blank(),
    axis.ticks.y     = element_blank(),
    axis.title.y     = element_blank(),
    strip.text       = element_text(size = 12, color = "black"),
    strip.background = element_rect(fill = "grey85", color = NA)
  )

label_df <- tibble(
  category = factor(cat_levels, levels = cat_levels),
  x = 1
)

plot_cat_labels <- ggplot(label_df, aes(x = x, y = category, label = category)) +
  geom_text(hjust = 1, size = 3.5) +
  scale_x_continuous(limits = c(0, 1)) +
  labs(x = NULL, y = NULL) +
  theme_void(base_size = 12) +
  theme(
    plot.margin = margin(t = 5.5, r = 2, b = 5.5, l = 0.5)
  )

plot_cat <- plot_cat_labels + plot_cat_main +
  plot_layout(widths = c(0.15, 1))

comp_df <- plot_df %>%
  group_by(category) %>%
  mutate(category_total = sum(n)) %>%
  ungroup() %>%
  # order categories by overall frequency
  mutate(category = reorder(category, category_total, desc = TRUE))

ggplot(comp_df, aes(x = category, y = n, fill = pH)) +
  geom_col(position = position_dodge(width = 0.8)) +
  coord_flip() +
  labs(
    title = "Reaction category counts across pH (comparable baseline)",
    x = "Category",
    y = "Count"
  ) +
  scale_fill_brewer(palette = "Set3", name = "pH") +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.y = element_text(margin = margin(r = 6))
  )

plot_conv_cat <- plot_conv / plot_cat +
  plot_layout(heights = c(1, 1))

ggsave("cat_dist.svg", plot = plot_conv_cat, height = 8, width = 12)
ggsave("cat_dist.pdf", plot = plot_conv_cat, height = 8, width = 12)

# export data for Python + RDKit scaffold analysis
dir.create("input_files", showWarnings = FALSE)
write.csv(data_conv, file.path("input_files", "data_conv.csv"), row.names = FALSE)
write.csv(data_metab_filt, file.path("input_files", "data_metab.csv"), row.names = FALSE)

# create data for ML (coalesce pH)
# same end-result after filter of minor metabolites
drop_cols <- c("name", "pH", "biomim",
               "metabolite", "metabolite_gen", "invivo", "conv",
               "H_prox", "X_exc", "ox_clv",
               "sp3_sp2", "sp2_sp", "R_Ar", "Q",
               "uncat", "unk", "percent", "percent_metab")
cat_cols <- setdiff(colnames(data), c(drop_cols, "smiles"))

flags_full <- data %>%
  filter(biomim == 1, conv == 0) %>%
  select(smiles, all_of(cat_cols)) %>%
  mutate(across(all_of(cat_cols), ~ replace_na(.x, 0))) %>%
  group_by(smiles) %>%
  summarise(
    across(
      all_of(cat_cols),
      ~ as.integer(any(. == 1, na.rm = TRUE))
    ),
    .groups = "drop"
  )

all_smiles <- data %>%
  distinct(smiles)

data_ML_full <- all_smiles %>%
  left_join(flags_full, by = "smiles") %>%
  mutate(across(all_of(cat_cols), ~ replace_na(.x, 0)))

outcomes <- data_ML_full %>%
  mutate(across(2:ncol(.), as.logical)) %>%
  select(-1)

for (col in colnames(outcomes)) {
  write.csv(
    outcomes[, col, drop = FALSE],
    file = file.path("input_files", paste0(col, "_outcomes.csv")),
    row.names = FALSE
  )
}

SMILES <- data_ML_full %>%
  select(1) %>%
  rename(SMILES = smiles)

write.table(SMILES, file.path("input_files", "SMILESforGNN.csv"), sep = "", row.names = FALSE)

# export percent outcomes as well
percent_data <- data %>%
  filter(biomim == 1, conv == 1, pH == "4.5") %>%
  select(smiles, percent)

percent_data <- percent_data[!duplicated(percent_data$smiles), ]
percent_data <- percent_data[match(SMILES$SMILES, percent_data$smiles), ]
all(percent_data$smiles == SMILES$SMILES)
percent_data <- percent_data %>% 
  select(percent)
percent_data <- percent_data %>% 
  mutate(
    percent = case_when(
      percent >  10 ~ 0,
      percent <= 10 ~ 1 # reversed logic
    )
  )
percent_data <- percent_data %>% 
  mutate(percent = as.logical(percent))

write.table(percent_data, file.path("input_files", "percent_outcomes.csv"), sep = "", row.names = FALSE)
