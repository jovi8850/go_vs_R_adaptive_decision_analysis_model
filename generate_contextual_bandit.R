###############################################
# Contextual Bandit Dataset Generator (Enhanced)
# -----------------------------------------------
# Author: [Your Name]
# Description:
#   Generates training and testing contextual bandit
#   datasets with different random seeds, noise, and
#   complexity levels for robust benchmarking.
#
# Output:
#   data/contextual_bandit_train.csv
#   data/contextual_bandit_test.csv
###############################################

# ---- PARAMETERS ----

# Data set export file location (update as needed)

file_location <- "FOLLOW INSTRUCTIONS IN README.md FILE"

# Dataset sizes
n_train <- 20000
n_test  <- 5000

# Shared structure
d <- 10          # number of contextual features
k <- 5           # number of actions

# Parameter settings
train_sigma <- 1.0        # noise level (standard deviation)
test_sigma  <- 1.5        # higher noise = harder test set

train_complexity <- 2     # mild nonlinearity
test_complexity  <- 3     # stronger nonlinearity for test

# Random seeds
train_seed <- 12345
test_seed  <- 54321


# ---- Nonlinear effect function ----
nonlinear_effect <- function(x, complexity = 1) {
  if (complexity == 1) {
    return(rep(0, nrow(x))) # linear only
  } else if (complexity == 2) {
    return(0.3 * sin(x[,1] + 0.5 * x[,2]) + 0.1 * (x[,3]^2 - 1))
  } else if (complexity >= 3) {
    return(0.5 * sin(x[,1] + x[,2]) +
             0.2 * (x[,3]^2 - 1) +
             0.1 * cos(x[,4] * x[,5]))
  }
}

# ---- Bandit data generator ----
generate_bandit_data <- function(n, d, k, sigma, complexity, seed = 1234) {
  set.seed(seed)
  
  # Feature matrix
  X <- matrix(rnorm(n * d), nrow = n, ncol = d)
  colnames(X) <- paste0("x", 1:d)
  
  # Action-specific coefficients
  betas <- matrix(rnorm(k * d, 0, 1), nrow = k)
  
  # Nonlinear term
  nonlinear_term <- nonlinear_effect(X, complexity)
  
  # Expected reward per action
  expected_rewards <- matrix(0, nrow = n, ncol = k)
  for (a in 1:k) {
    expected_rewards[, a] <- X %*% betas[a, ] + nonlinear_term
  }
  
  # Observed rewards with noise
  observed_rewards <- expected_rewards + matrix(rnorm(n * k, 0, sigma), nrow = n)
  
  # Best action (true optimal)
  best_action <- apply(expected_rewards, 1, which.max)
  best_reward <- apply(expected_rewards, 1, max)
  
  # Build data frame
  df <- data.frame(X)
  df$best_action <- best_action
  df$best_reward <- best_reward
  
  # Add observed rewards
  reward_cols <- as.data.frame(observed_rewards)
  colnames(reward_cols) <- paste0("reward_a", 1:k)
  
  cb_data <- cbind(df, reward_cols)
  attr(cb_data, "betas") <- betas
  cb_data
}

# ---- Generate Training and Test Data ----
cat("Generating contextual bandit datasets...\n")

train_data <- generate_bandit_data(n_train, d, k, sigma = train_sigma,
                                   complexity = train_complexity, seed = train_seed)

test_data  <- generate_bandit_data(n_test,  d, k, sigma = test_sigma,
                                   complexity = test_complexity, seed = test_seed)

# ---- Save Outputs ----

# Make sure the folder exists
if (!dir.exists(file_location)) dir.create(file_location, recursive = TRUE)

# Save outputs (using file.path for safety)
write.csv(train_data, file.path(file_location, "contextual_bandit_train.csv"), row.names = FALSE)
write.csv(test_data,  file.path(file_location, "contextual_bandit_test.csv"),  row.names = FALSE)

# ---- Summary Log ----
cat("\n✅ Contextual Bandit Datasets Generated\n")
cat("========================================\n")
cat(sprintf("Train: %d samples | σ = %.2f | Complexity = %d | Seed = %d\n",
            n_train, train_sigma, train_complexity, train_seed))
cat(sprintf("Test : %d samples | σ = %.2f | Complexity = %d | Seed = %d\n",
            n_test,  test_sigma,  test_complexity,  test_seed))
cat(sprintf("Features: %d | Actions: %d\n", d, k))
cat("========================================\n")

# ---- Optional Visualization ----
suppressPackageStartupMessages(library(ggplot2))
sample_plot <- data.frame(
  x1 = train_data$x1[1:2000],
  x2 = train_data$x2[1:2000],
  best_action = as.factor(train_data$best_action[1:2000])
)

ggplot(sample_plot, aes(x = x1, y = x2, color = best_action)) +
  geom_point(alpha = 0.7) +
  labs(title = "Training Set: Best Action by Context",
       subtitle = sprintf("Noise σ = %.1f | Complexity = %d", train_sigma, train_complexity)) +
  theme_minimal()
