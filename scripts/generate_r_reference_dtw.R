# Generate DTW reference data from R dtw package.
# Usage: Rscript scripts/generate_r_reference_dtw.R
#
# We use step.pattern=symmetric1 (standard DTW with equal weights for all
# 3 directions: up, left, diagonal). R's default is symmetric2, which doubles
# the diagonal cost -- a normalization convention, not the standard definition.

library(dtw)
library(jsonlite)

set.seed(42)

scenarios <- list()

# Scenario 1: basic DTW (no window constraint)
x1 <- c(1.0, 3.0, 5.0, 2.0, 8.0)
y1 <- c(2.0, 4.0, 1.0, 3.0, 6.0)
d1 <- dtw(x1, y1, step.pattern = symmetric1, distance.only = TRUE)
scenarios[[1]] <- list(
  scenario = "dtw_basic",
  x = x1,
  y = y1,
  window = NULL,
  r_dtw_distance = d1$distance
)

# Scenario 2: DTW with Sakoe-Chiba window
x2 <- rnorm(30)
y2 <- rnorm(30)
d2 <- dtw(x2, y2, step.pattern = symmetric1,
          window.type = "sakoechiba", window.size = 5, distance.only = TRUE)
scenarios[[2]] <- list(
  scenario = "dtw_with_window",
  x = x2,
  y = y2,
  window = 5,
  r_dtw_distance = d2$distance
)

# Scenario 3: different length series
x3 <- c(1.0, 2.0, 3.0)
y3 <- c(1.0, 2.0, 3.0, 4.0, 5.0)
d3 <- dtw(x3, y3, step.pattern = symmetric1, distance.only = TRUE)
scenarios[[3]] <- list(
  scenario = "dtw_different_length",
  x = x3,
  y = y3,
  window = NULL,
  r_dtw_distance = d3$distance
)

# Scenario 4: seasonal pattern
t_seq <- 1:50
x4 <- sin(2 * pi * t_seq / 12)
y4 <- sin(2 * pi * t_seq / 12 + 0.5) + rnorm(50, 0, 0.1)
d4 <- dtw(x4, y4, step.pattern = symmetric1, distance.only = TRUE)
scenarios[[4]] <- list(
  scenario = "dtw_seasonal",
  x = x4,
  y = y4,
  window = NULL,
  r_dtw_distance = d4$distance
)

# Scenario 5: control selection ranking
y_sel <- cumsum(rnorm(60))
x_very_similar <- y_sel + rnorm(60, 0, 0.3)
x_similar <- y_sel * 0.8 + cumsum(rnorm(60)) * 0.2
x_moderate <- cumsum(rnorm(60))
x_different <- cumsum(rnorm(60)) * 5 + 100

dtw_vs <- dtw(x_very_similar, y_sel, step.pattern = symmetric1, distance.only = TRUE)$distance
dtw_si <- dtw(x_similar, y_sel, step.pattern = symmetric1, distance.only = TRUE)$distance
dtw_mo <- dtw(x_moderate, y_sel, step.pattern = symmetric1, distance.only = TRUE)$distance
dtw_di <- dtw(x_different, y_sel, step.pattern = symmetric1, distance.only = TRUE)$distance

scenarios[[5]] <- list(
  scenario = "control_selection_ranking",
  y = y_sel,
  candidates = list(
    list(name = "x_very_similar", values = x_very_similar, r_dtw_distance = dtw_vs),
    list(name = "x_similar", values = x_similar, r_dtw_distance = dtw_si),
    list(name = "x_moderate", values = x_moderate, r_dtw_distance = dtw_mo),
    list(name = "x_different", values = x_different, r_dtw_distance = dtw_di)
  ),
  r_ranking = c("x_very_similar", "x_similar", "x_moderate", "x_different")[
    order(c(dtw_vs, dtw_si, dtw_mo, dtw_di))
  ]
)

json_out <- toJSON(scenarios, auto_unbox = TRUE, digits = 15, pretty = TRUE)
writeLines(json_out, "tests/fixtures/r_reference_dtw.json")
cat("Generated tests/fixtures/r_reference_dtw.json\n")
