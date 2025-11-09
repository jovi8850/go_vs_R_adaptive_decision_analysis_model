# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(microbenchmark)
library(pryr)
library(openxlsx)
library(jsonlite)

# Set random seed for reproducibility
set.seed(123)

# Parameter logging structure
ParameterLogger <- R6::R6Class("ParameterLogger",
  public = list(
    parameters = list(),
    performance_metrics = list(),
    
    initialize = function() {
      self$parameters <- list(
        timestamp = Sys.time(),
        r_version = R.version.string,
        platform = R.version$platform
      )
    },
    
    log_parameters = function(model_type, params) {
      self$parameters[[model_type]] <- params
    },
    
    log_performance = function(model_type, metrics) {
      self$performance_metrics[[model_type]] <- metrics
    },
    
    export_to_excel = function(filename = "parameter_log_r.xlsx") {
      # Create workbook
      wb <- createWorkbook()
      
      # Add parameters sheet
      addWorksheet(wb, "Parameters")
      params_df <- data.frame(
        Parameter = names(unlist(self$parameters)),
        Value = unlist(self$parameters)
      )
      writeData(wb, "Parameters", params_df)
      
      # Add performance metrics sheet
      addWorksheet(wb, "Performance")
      perf_list <- list()
      for (model in names(self$performance_metrics)) {
        metrics <- self$performance_metrics[[model]]
        for (metric_name in names(metrics)) {
          perf_list[[paste(model, metric_name, sep = "_")]] <- metrics[[metric_name]]
        }
      }
      perf_df <- data.frame(
        Metric = names(perf_list),
        Value = unlist(perf_list)
      )
      writeData(wb, "Performance", perf_df)
      
      # Add experiment details sheet
      addWorksheet(wb, "Experiment_Details")
      exp_details <- data.frame(
        Item = c("Experiment Name", "Date", "Random Seed", "Data Files"),
        Value = c("Contextual Bandit Analysis", as.character(Sys.Date()), 
                 "123", "contextual_bandit_train.csv, contextual_bandit_test.csv")
      )
      writeData(wb, "Experiment_Details", exp_details)
      
      # Save workbook
      saveWorkbook(wb, filename, overwrite = TRUE)
      cat("Parameters and performance metrics exported to", filename, "\n")
    }
  )
)

# Initialize global parameter logger
param_logger <- ParameterLogger$new()

# LinUCB Contextual Bandit Model
LinUCB <- R6::R6Class("LinUCB",
  public = list(
    alpha = NULL,
    A = NULL,
    b = NULL,
    d = NULL,
    k = NULL,
    seed = NULL,
    
    initialize = function(alpha = 1.0, d = 10, k = 5, seed = 123) {
      self$alpha <- alpha
      self$d <- d
      self$k <- k
      self$seed <- seed
      set.seed(seed)
      
      self$A <- lapply(1:k, function(i) diag(d))
      self$b <- lapply(1:k, function(i) numeric(d))
      
      # Log parameters
      param_logger$log_parameters("linucb", list(
        alpha = alpha,
        context_dimension = d,
        num_arms = k,
        random_seed = seed,
        model_type = "LinUCB",
        initialization = "Identity matrix for A, zero vector for b"
      ))
    },
    
    select_arm = function(context) {
      scores <- numeric(self$k)
      
      for (arm in 1:self$k) {
        A_inv <- solve(self$A[[arm]])
        theta <- A_inv %*% self$b[[arm]]
        score <- t(theta) %*% context + self$alpha * sqrt(t(context) %*% A_inv %*% context)
        scores[arm] <- score
      }
      
      which.max(scores)
    },
    
    update = function(arm, context, reward) {
      self$A[[arm]] <- self$A[[arm]] + context %*% t(context)
      self$b[[arm]] <- self$b[[arm]] + reward * context
    }
  )
)

# Epsilon-Greedy Bandit Model
EpsilonGreedy <- R6::R6Class("EpsilonGreedy",
  public = list(
    epsilon = NULL,
    counts = NULL,
    values = NULL,
    k = NULL,
    seed = NULL,
    
    initialize = function(epsilon = 0.1, k = 5, seed = 123) {
      self$epsilon <- epsilon
      self$k <- k
      self$seed <- seed
      set.seed(seed)
      
      self$counts <- numeric(k)
      self$values <- numeric(k)
      
      # Log parameters
      param_logger$log_parameters("epsilon_greedy", list(
        epsilon = epsilon,
        num_arms = k,
        random_seed = seed,
        model_type = "Epsilon-Greedy",
        initialization = "Zero counts and values"
      ))
    },
    
    select_arm = function(context = NULL) {
      if (runif(1) < self$epsilon) {
        sample(1:self$k, 1)
      } else {
        which.max(self$values)
      }
    },
    
    update = function(arm, context = NULL, reward) {
      self$counts[arm] <- self$counts[arm] + 1
      n <- self$counts[arm]
      value <- self$values[arm]
      self$values[arm] <- ((n - 1) / n) * value + (1 / n) * reward
    }
  )
)

# Performance monitoring function
monitor_performance <- function(expression, model_name) {
  start_time <- Sys.time()
  mem_before <- pryr::mem_used()
  
  # Execute the expression
  result <- force(expression)
  
  mem_after <- pryr::mem_used()
  end_time <- Sys.time()
  
  runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
  memory_used <- mem_after - mem_before
  
  # Log performance metrics
  param_logger$log_performance(model_name, list(
    runtime_seconds = runtime,
    memory_used_bytes = memory_used,
    memory_used_mb = memory_used / 1024 / 1024,
    timestamp = Sys.time(),
    r_objects_count = length(ls(envir = .GlobalEnv))
  ))
  
  return(list(
    result = result,
    runtime = runtime,
    memory_used = memory_used
  ))
}

# Adaptive Decision Analysis
ada_model_r <- function(train_file, test_file, model_type = "linucb") {
  # Load data with error handling
  train_data <- tryCatch({
    read.csv(train_file)
  }, error = function(e) {
    stop("Error loading training data: ", e$message)
  })
  
  test_data <- tryCatch({
    read.csv(test_file)
  }, error = function(e) {
    stop("Error loading test data: ", e$message)
  })
  
  # Log data characteristics
  param_logger$log_parameters("data_info", list(
    train_samples = nrow(train_data),
    test_samples = nrow(test_data),
    features = ncol(train_data) - 6, # Excluding reward columns and best_action/best_reward
    train_file = train_file,
    test_file = test_file,
    data_loaded_time = Sys.time()
  ))
  
  # Initialize model with seeded randomness
  if (model_type == "linucb") {
    model <- LinUCB$new(alpha = 1.0, seed = 123)
  } else {
    model <- EpsilonGreedy$new(epsilon = 0.1, seed = 123)
  }
  
  # Track metrics
  history <- list(
    chosen_actions = integer(nrow(test_data)),
    rewards = numeric(nrow(test_data)),
    regrets = numeric(nrow(test_data)),
    cumulative_reward = numeric(nrow(test_data)),
    cumulative_regret = numeric(nrow(test_data)),
    context_used = matrix(0, nrow = nrow(test_data), ncol = 10)
  )
  
  # Train model on training data
  for (i in 1:nrow(train_data)) {
    row <- train_data[i, ]
    context <- as.numeric(row[1:10])
    
    if (model_type == "linucb") {
      arm <- model$select_arm(context)
      reward_col <- paste0("reward_a", arm)
      reward <- row[[reward_col]]
      model$update(arm, context, reward)
    } else {
      arm <- model$select_arm()
      reward_col <- paste0("reward_a", arm)
      reward <- row[[reward_col]]
      model$update(arm, reward = reward)
    }
  }
  
  # Test model on test data
  total_reward <- 0
  total_regret <- 0
  
  for (i in 1:nrow(test_data)) {
    row <- test_data[i, ]
    context <- as.numeric(row[1:10])
    best_reward <- row$best_reward
    
    if (model_type == "linucb") {
      arm <- model$select_arm(context)
    } else {
      arm <- model$select_arm()
    }
    
    reward_col <- paste0("reward_a", arm)
    reward <- row[[reward_col]]
    regret <- best_reward - reward
    
    total_reward <- total_reward + reward
    total_regret <- total_regret + regret
    
    history$chosen_actions[i] <- arm
    history$rewards[i] <- reward
    history$regrets[i] <- regret
    history$cumulative_reward[i] <- total_reward
    history$cumulative_regret[i] <- total_regret
    history$context_used[i, ] <- context
    
    # Update model with test data (continual learning)
    if (model_type == "linucb") {
      model$update(arm, context, reward)
    } else {
      model$update(arm, reward = reward)
    }
  }
  
  # Log final performance metrics
  param_logger$log_performance(paste0(model_type, "_results"), list(
    final_cumulative_reward = tail(history$cumulative_reward, 1),
    final_cumulative_regret = tail(history$cumulative_regret, 1),
    average_reward = mean(history$rewards),
    average_regret = mean(history$regrets),
    total_decisions = length(history$chosen_actions),
    arm_distribution = paste(table(history$chosen_actions), collapse = ",")
  ))
  
  return(history)
}

# Run analysis and create visualizations
run_analysis_r <- function() {
  cat("Starting R Adaptive Decision Analysis...\n")
  cat("Random seed set to: 123\n")
  
  results <- list()
  
  # LinUCB Model with performance monitoring
  cat("Running LinUCB model...\n")
  linucb_perf <- monitor_performance({
    ada_model_r("contextual_bandit_train.csv", "contextual_bandit_test.csv", "linucb")
  }, "linucb_model")
  results$linucb <- linucb_perf$result
  
  # Epsilon-Greedy Model with performance monitoring
  cat("Running Epsilon-Greedy model...\n")
  epsilon_perf <- monitor_performance({
    ada_model_r("contextual_bandit_train.csv", "contextual_bandit_test.csv", "epsilon")
  }, "epsilon_greedy_model")
  results$epsilon <- epsilon_perf$result
  
  # Create visualizations
  create_visualizations_r(results)
  
  # Export parameters and performance metrics
  param_logger$export_to_excel("r_analysis_parameters.xlsx")
  
  # Print summary
  cat("\n=== R ANALYSIS COMPLETED ===\n")
  cat("LinUCB - Final Reward:", tail(results$linucb$cumulative_reward, 1), "\n")
  cat("Epsilon-Greedy - Final Reward:", tail(results$epsilon$cumulative_reward, 1), "\n")
  cat("Parameters exported to: r_analysis_parameters.xlsx\n")
  
  return(list(
    results = results,
    parameters = param_logger$parameters,
    performance = param_logger$performance_metrics
  ))
}

create_visualizations_r <- function(results) {
  # Cumulative Reward Plot
  reward_data <- data.frame(
    step = 1:length(results$linucb$cumulative_reward),
    linucb = results$linucb$cumulative_reward,
    epsilon = results$epsilon$cumulative_reward
  )
  
  p1 <- ggplot(reward_data, aes(x = step)) +
    geom_line(aes(y = linucb, color = "LinUCB"), size = 1) +
    geom_line(aes(y = epsilon, color = "Epsilon-Greedy"), size = 1) +
    labs(title = "R Implementation - Cumulative Reward Over Time",
         subtitle = paste("Random Seed: 123 | Date:", Sys.Date()),
         x = "Step", y = "Cumulative Reward",
         color = "Algorithm") +
    theme_minimal()
  
  # Cumulative Regret Plot
  regret_data <- data.frame(
    step = 1:length(results$linucb$cumulative_regret),
    linucb = results$linucb$cumulative_regret,
    epsilon = results$epsilon$cumulative_regret
  )
  
  p2 <- ggplot(regret_data, aes(x = step)) +
    geom_line(aes(y = linucb, color = "LinUCB"), size = 1) +
    geom_line(aes(y = epsilon, color = "Epsilon-Greedy"), size = 1) +
    labs(title = "R Implementation - Cumulative Regret Over Time",
         subtitle = paste("Random Seed: 123 | Date:", Sys.Date()),
         x = "Step", y = "Cumulative Regret",
         color = "Algorithm") +
    theme_minimal()
  
  print(p1)
  print(p2)
  
  # Save plots with timestamp
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  ggsave(paste0("r_cumulative_reward_", timestamp, ".png"), p1, width = 8, height = 6)
  ggsave(paste0("r_cumulative_regret_", timestamp, ".png"), p2, width = 8, height = 6)
}

# Run the analysis
r_results <- run_analysis_r()