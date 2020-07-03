library(gtools)
library(rstanarm)
library(gridExtra)
library(dplyr)
library(mvtnorm)
library(ggplot2)

setwd("~/Project")
options(mc.cores = parallel::detectCores())

#################################################
##### Example 1: Logistic GLM using a single ####
##### dataset and single prior distribution #####
#################################################

#Generate dataset
set.seed(13)
n <- 10000
beta <- c(-0.5, 0.3, 0.35, .3)
df <- data.frame(x0 = rep(1, n),
                 x1 = runif(min = -2, max = 4, n= n),
                 x2 = rpois(n = n, lambda = 6),
                 x3 = rnorm(n = n, mean = -6, sd = .5))

par(mfrow=c(1,2))
probs <- inv.logit(as.matrix(df) %*% beta)
hist(as.matrix(df) %*% beta, breaks= 20)
hist(probs)
plot(probs, as.matrix(df) %*% beta)
df <- df[,-1]
df$y <- rbinom(n = n, size = 1, prob = probs)

# Add noise parameters
df$x4 <- rbinom(n = nrow(df), size = 4, prob = 0.3)
df$x5 <- rnorm(n = nrow(df))

# There are 5 different models we will be comparing. 
# Model #3 pertains to the correct data generation mechanism
# Define formula objects
f1 <<- as.formula("y ~ x1")
f2 <<- as.formula("y ~ x1 + x2")
f3 <<- as.formula("y ~ x1 + x2 + x3")
f4 <<- as.formula("y ~ x1 + x2 + x3 + x4")
f5 <<- as.formula("y ~ x1 + x2 + x3 + x4 + x5")

# Normal glm for baseline
glm1 <- glm(formula = f1, family = binomial(link = logit), data = df)
glm2 <- glm(formula = f2, family = binomial(link = logit), data = df)
glm3 <- glm(formula = f3, family = binomial(link = logit), data = df)
glm4 <- glm(formula = f4, family = binomial(link = logit), data = df)
glm5 <- glm(formula = f5, family = binomial(link = logit), data = df)

# Is it easier to pick out the correct model using just a normal glm? 
# How accurate does the prior have to be?
# We can compare just the two best priors vs regular glm for 50 and 100 samples.


#######################################################################
## Example 2: Logistic GLM using one dataset but a variety of priors ##
#######################################################################

## Function to automate generation of logistic regression stan model
# NB: Samples ARE NOT independent. This is by design.
# I.e. if n1 = 50 and n2 = 100, Sample 2 contains the same 50 records as Sample 1, plus an additional 50
logmod <- function(df, form, n = nrow(df), location, scale, link_fn = 'logit'){
  n <- max(n, nrow(df))
  df <- df[1:n,]
  num_params <- length(labels(terms(form)))
  location <- location[1:num_params]
  scale <- scale[1:num_params]
  model <- stan_glm(formula =  form, data = df, family = binomial(link = logit),
                    prior = normal(
                      location = location,
                      scale = scale)
  )
  return(model)
}


# Specify location (i.e. mean) parameters for priors arranged from most to least accurate
loc1 <- c(0.3, 0.15, .3, 0, 0) #True parameter set
loc2 <- c(0, 0, 0, 1, 1) # Close to the true parameter set
loc3 <- c(50, 0, 0, 1, 1) # Incorrect parameter set
loc4 <- c(100, 0, 0, 1, 1) # Extremely incorrect parameter set

# Specify scale (i.e variance) parameters for priors arranged from most to least informative
sca1 <- c(0.1, 0.1, 0.1, 0.1, 0.1) # Extremely informative
sca2 <- c(1, 1, 1, 1, 1) # Somewhat informative
sca3 <- c(5, 5, 5, 5, 5) # Uninformative
sca4 <- c(100, 100, 100, 100, 100) # Extremely uninformative



# We'll start by comparing WAIC and PSIS-LOO performance on a moderate case:
# 500 samples, and a moderately informative (and correct) prior
# Model format is m.(# of params).(location param set #).(scale param set #)
m1 <- logmod(df = df, form = f1, location = loc2, scale = sca2, n = 100)
m2 <- logmod(df = df, form = f2, location = loc2, scale = sca2, n = 100)
m3 <- logmod(df = df, form = f3, location = loc2, scale = sca2, n = 100)
m4 <- logmod(df = df, form = f4, location = loc2, scale = sca2, n = 100)
m5 <- logmod(df = df, form = f5, location = loc2, scale = sca2, n = 100)

# Small test of execution times for LOO vs WAIC
system.time(loo(m3))
system.time(waic(m3))

# Calculate PSIS-LOO for each model
loo1 <- loo(m1)
loo2 <- loo(m2)
loo3 <- loo(m3)
loo4 <- loo(m4)
loo5 <- loo(m5)

# Check the PSIS diagnosis plot to ensure there weren't issues with performing PSIS
par(mfrow = c(2,2))
plot(loo1)
plot(loo2)
plot(loo3)
plot(loo4)

# Plot comparing the observed outcome variable 𝑦 to simulated datasets 𝑦𝑟𝑒𝑝 from the posterior predictive distribution.
p1 <- pp_check(m1) + ggtitle("Model 1")
p2 <- pp_check(m2) + ggtitle('Model 2')
p3 <- pp_check(m3) + ggtitle('Model 3')
p4 <- pp_check(m4) + ggtitle('Model 4')
p5 <- pp_check(m5) + ggtitle('Model 5')
grid.arrange(p1, p2, p3, p4, p5, nrow = 3)

# Compare the 5 models to see which model it identifies
loo_results <- data.frame(Model = 1:5, LOO = c(loo1$estimates[3,1], loo2$estimates[3,1], loo3$estimates[3,1],
                                               loo4$estimates[3,1], loo5$estimates[3,1]))
arrange(loo_results, LOO)

# Can also use loo_compare to see the relative difference between the 5 models
loo_compare(loo1, loo2, loo3, loo4, loo5)

# PSIS-LOO incorrectly identifies Model 4 as being our model of interest. Let's see if WAIC
# does any better

waic_results <- data.frame(Model = 1:5, WAIC = c(waic(m1)$estimates[3,1], waic(m2)$estimates[3,1], waic(m3)$estimates[3,1],
                                                 waic(m4)$estimates[3,1], waic(m5)$estimates[3,1]))

arrange(waic_results, WAIC)

# Similarly, WAIC incorrectly identifies Model #4. It also throws an error message for models 3, 4 & 5.

results <- data.frame(Model = 1:5, LOO = loo_results$LOO, WAIC = waic_results$WAIC)
ggplot(results, aes(x = Model)) + 
  geom_line(aes(y = WAIC, color = 'WAIC')) + 
  geom_point(aes(y = WAIC, color = 'WAIC')) +
  geom_line(aes(y = LOO, color = 'LOO')) + 
  geom_point(aes(y = LOO, color = 'LOO')) +
  labs(color = '')
# The results from WAIC and LOO are nearly identical, even with only 100 data points.
# Let's see if re-running this experiment 100 times yields different results.

####################################################
### Example 3: Logistic GLM using many simulated ###
### datasets and a variety of priors ###############
####################################################

# Simulation function
sim_fun <- function(beta = c(-0.5, 0.3, 0.35, .3), n = 50, reps = 1, loc_param, scale_param){ 
  
  sim_df <- data.frame(iteration = 1:reps,
                       LOO_selection = NA,
                       LOO_elpd_diff = NA,
                       WAIC_selection = NA,
                       WAIC_diff = NA)
  for(i in 1:reps){
    # Create the dataset
    set.seed(i)
    df <- data.frame(x0 = rep(1, n),
                     x1 = runif(min = -2, max = 4, n= n),
                     x2 = rpois(n = n, lambda = 6),
                     x3 = rnorm(n = n, mean = -6, sd = .5))
    
    probs <- inv.logit(as.matrix(df) %*% beta)
    df <- df[,-1]
    df$y <- rbinom(n = n, size = 1, prob = probs)
    
    # Add in noise parameters
    df$x4 <- rbinom(n = nrow(df), size = 4, prob = 0.3)
    df$x5 <- rnorm(n = nrow(df))
    
    # Create the posterior models
    m1 <- logmod(df = df, form = f1, location = loc_param, scale = scale_param)
    m2 <- logmod(df = df, form = f2, location = loc_param, scale = scale_param)
    m3 <- logmod(df = df, form = f3, location = loc_param, scale = scale_param)
    m4 <- logmod(df = df, form = f4, location = loc_param, scale = scale_param)
    m5 <- logmod(df = df, form = f5, location = loc_param, scale = scale_param)
    
    # Calculate PSIS-LOO for each model
    loo1 <- loo(m1)
    loo2 <- loo(m2)
    loo3 <- loo(m3)
    loo4 <- loo(m4)
    loo5 <- loo(m5)
    comp <- loo_compare(loo1, loo2, loo3, loo4, loo5)
    
    # Calculate WAIC for each
    waic_df = data.frame(model = c('m1', 'm2', 'm3', 'm4', 'm5'),
                         waic = c(waic(m1)$estimates[3,1], waic(m2)$estimates[3,1], waic(m3)$estimates[3,1],
                                  waic(m4)$estimates[3,1], waic(m5)$estimates[3,1]))
    waic_df <- arrange(waic_df, waic)
    
    # Update dataframe with results
    sim_df$LOO_selection[i] <- row.names(comp)[1]
    sim_df$LOO_elpd_diff[i] <- comp[2,1]
    sim_df$WAIC_selection[i] <- as.character(waic_df$model[1])
    sim_df$WAIC_diff[i] <- waic_df$waic[1] - waic_df$waic[2]
  }
  return(sim_df)
}

# Now we can test how LOO and WAIC compare with different sample sizes and prior distributions
# 100 repetitions is used for each simulation

# Sims 1-4 use a moderately informative, correct prior
sim1 <- sim_fun(reps = 100, n = 50, loc_param = loc2, scale_param = sca2)
sim2 <- sim_fun(reps = 100, n = 100, loc_param = loc2, scale_param = sca2)
sim3 <- sim_fun(reps = 100, n = 500, loc_param = loc2, scale_param = sca2)
sim4 <- sim_fun(reps = 100, n = 1000, loc_param = loc2, scale_param = sca2)
### Save and load functions below to avoid having to rerun the simulation function in future investigations (since run time is long)
# save(sim1, file = 'sim1.Rd')
# save(sim2, file = 'sim2.Rd')
# save(sim3, file = 'sim3.Rd')
# save(sim4, file = 'sim4.Rd')
# load('sim1.Rd')
# load('sim2.Rd')
# load('sim3.Rd')
# load('sim4.Rd')
table(sim1$LOO_selection, sim1$WAIC_selection)
table(sim2$LOO_selection, sim2$WAIC_selection)
table(sim3$LOO_selection, sim3$WAIC_selection)
table(sim4$LOO_selection, sim4$WAIC_selection)

# Sims 5-6  use the best possible prior
sim5 <- sim_fun(reps = 100, n = 25, loc_param = loc1, scale_param = sca1)
sim6 <- sim_fun(reps = 50, n = 100, loc_param = loc1, scale_param = sca1)
# save(sim5, file = 'sim5.Rd')
# save(sim6, file = 'sim6.Rd')
# load('sim5.Rd')
# load('sim6.Rd')
table(sim5$LOO_selection, sim5$WAIC_selection)
table(sim6$LOO_selection, sim6$WAIC_selection)

# Sim 7 uses a precise but incorrect prior
sim7 <- sim_fun(reps = 50, n = 1000, loc_param = loc3, scale_param = sca3)
# save(sim7, file = 'sim7.Rd')
# load('sim7.Rd')
table(sim7$LOO_selection, sim7$WAIC_selection)

# Sim 8 use a super wide/uninformative prior
sim8 <- sim_fun(reps = 100, n = 5000, loc_param = loc2, scale_param = sca4)
# save(sim8, file = 'sim8.Rd')
# load('sim8.Rd')
table(sim8$LOO_selection, sim8$WAIC_selection)



##############################################################
### Example 4: Logistic GLM with different data generation ###
### using many simulated datasets and a variety of priors ####
##############################################################
# We'll use a different data generation function for the next set of simulations. 
# Let's plot what it looks like
n <- 10000
beta <- c(-5, 0.3, 0.6, 0.6)
df <- data.frame(x0 = rep(1, n),
                 x1 = runif(min = -2, max = 4, n= n),
                 x2 = rpois(n = n, lambda = 6),
                 x3 = rnorm(n = n, mean = -6, sd = .5))

probs <- inv.logit(as.matrix(df) %*% beta)
par(mfrow=c(1,2))

hist(probs)
plot(probs, as.matrix(df) %*% beta)


# Define 2nd simulation function (using different data generation mechanism, but still logistic regression)
sim_fun2 <- function(beta = c(-5, 0.3, 0.6, 0.6), n = 50, reps = 1, loc_param, scale_param){ 
  
  sim_df <- data.frame(iteration = 1:reps,
                       LOO_selection = NA,
                       LOO_elpd_diff = NA,
                       WAIC_selection = NA,
                       WAIC_diff = NA)
  for(i in 1:reps){
    # Create the dataset
    set.seed(i)
    df <- data.frame(x0 = rep(1, n),
                     x1 = runif(min = -2, max = 4, n= n),
                     x2 = rpois(n = n, lambda = 6),
                     x3 = rnorm(n = n, mean = -6, sd = .5))
    
    probs <- inv.logit(as.matrix(df) %*% beta)
    df <- df[,-1]
    df$y <- rbinom(n = n, size = 1, prob = probs)
    
    # Add in noise parameters
    df$x4 <- rbinom(n = nrow(df), size = 4, prob = 0.3)
    df$x5 <- rnorm(n = nrow(df))
    
    # Create the posterior models
    m1 <- logmod(df = df, form = f1, location = loc_param, scale = scale_param)
    m2 <- logmod(df = df, form = f2, location = loc_param, scale = scale_param)
    m3 <- logmod(df = df, form = f3, location = loc_param, scale = scale_param)
    m4 <- logmod(df = df, form = f4, location = loc_param, scale = scale_param)
    m5 <- logmod(df = df, form = f5, location = loc_param, scale = scale_param)
    
    # Calculate PSIS-LOO for each model
    loo1 <- loo(m1)
    loo2 <- loo(m2)
    loo3 <- loo(m3)
    loo4 <- loo(m4)
    loo5 <- loo(m5)
    comp <- loo_compare(loo1, loo2, loo3, loo4, loo5)
    
    # Calculate WAIC for each model
    waic_df = data.frame(model = c('m1', 'm2', 'm3', 'm4', 'm5'),
                         waic = c(waic(m1)$estimates[3,1], waic(m2)$estimates[3,1], waic(m3)$estimates[3,1],
                                  waic(m4)$estimates[3,1], waic(m5)$estimates[3,1]))
    waic_df <- arrange(waic_df, waic)
    
    # Update dataframe with results
    sim_df$LOO_selection[i] <- row.names(comp)[1]
    sim_df$LOO_elpd_diff[i] <- comp[2,1]
    sim_df$WAIC_selection[i] <- as.character(waic_df$model[1])
    sim_df$WAIC_diff[i] <- waic_df$waic[1] - waic_df$waic[2]
  }
  return(sim_df)
}

# Sims 1-2 use an informative, correct prior
sim1.2 <-sim_fun2(reps = , n = 20, loc_param = loc2, scale_param = sca2)
sim2.2 <-sim_fun2(reps = 5, n = 50, loc_param = loc2, scale_param = sca2)
# save(sim1.2, file = 'sim1_2.Rd')
# save(sim2.2, file = 'sim2_2.Rd')
# load('sim1_2.Rd')
# load('sim2_2.Rd')
table(sim1.2$LOO_selection, sim1.2$WAIC_selection)
table(sim2.2$LOO_selection, sim2.2$WAIC_selection)

# Sim 3  use the best possible prior
sim3.2 <- sim_fun2(reps = 1, n = 25, loc_param = loc1, scale_param = sca1)
# save(sim3.2, file = 'sim3_2.Rd')
# load('sim3_2.Rd')
table(sim3.2$LOO_selection, sim3.2$WAIC_selection)

# Sim 4 use a super wide/uninformative prior
sim4.2 <- sim_fun2(reps = 10, n = 5000, loc_param = loc2, scale_param = sca4)
# save(sim4.2, file = 'sim4_2.Rd')
# load('sim4_2.Rd')
table(sim4.2$LOO_selection, sim4.2$WAIC_selection)




###################################################################################
## Example 5: Linear Model using many simulated datasets and a variety of priors ##
###################################################################################

linmod <- function(df, form, n = nrow(df), location, scale, iter = 2000){
  n <- max(n, nrow(df))
  df <- df[1:n,]
  num_params <- length(labels(terms(form)))
  location <- location[1:num_params]
  scale <- scale[1:num_params]
  model <- stan_glm(formula = form, data = df, family = gaussian(),
                    prior = normal(
                      location = location,
                      scale = scale),
                    prior_intercept = normal(location = -5, scale = 2),
                    iter = iter)
  return(model)
}


#Try a few models to see if serious issues with e.g. chains mixing
beta = c(-5, 0.3, 0.15, 0.3)
n = 50
df <- data.frame(x0 = rep(1, n),
                 x1 = rnorm(n = n, mean = 4, sd = 1),
                 x2 = rnorm(n = n, mean = 16, sd = 3),
                 x3 = rnorm(n = n, mean = -6, sd = .5))

df$y <- as.matrix(df) %*% beta
df <- df[,-1]

# Add in noise parameters
df$x4 <- rbinom(n = nrow(df), size = 4, prob = 0.3)
df$x5 <- rnorm(n = nrow(df))

summary(lm(data = df, form = f5))

location <- loc1
scale <- sca1
loc_param <- loc1
scale_param <- sca1
m1 <- linmod(df = df, form = f1, location = loc_param, scale = scale_param)
m2 <- linmod(df = df, form = f2, location = loc_param, scale = scale_param)
m3 <- linmod(df = df, form = f3, location = loc_param, scale = scale_param)
m4 <- linmod(df = df, form = f4, location = loc_param, scale = scale_param)
m5 <- linmod(df = df, form = f5, location = loc_param, scale = scale_param)



# 3rd Simulation function (using different data generation mechanism and linear regression)
sim_fun3 <- function(beta = c(-5, 0.3, 0.15, 0.3), 
                     n = 50, reps = 1, loc_param, scale_param, iter = 2000){ 
  sim_df <- data.frame(iteration = 1:reps,
                       LOO_selection = NA,
                       LOO_elpd_diff = NA,
                       WAIC_selection = NA,
                       WAIC_diff = NA)
  for(i in 1:reps){
    # Create the dataset
    set.seed(i)
    df <- data.frame(x0 = rep(1, n),
                     x1 = rnorm(n = n, mean = 4, sd = 1),
                     x2 = rnorm(n = n, mean = 16, sd = 3),
                     x3 = rnorm(n = n, mean = -6, sd = .5))
    
    df$y <- as.matrix(df) %*% beta
    df <- df[,-1]
    
    # Add in noise parameters
    df$x4 <- rbinom(n = nrow(df), size = 4, prob = 0.3)
    df$x5 <- rnorm(n = nrow(df))
    
    # Create the posterior models
    m1 <- linmod(df = df, form = f1, location = loc_param, scale = scale_param)
    m2 <- linmod(df = df, form = f2, location = loc_param, scale = scale_param)
    m3 <- linmod(df = df, form = f3, location = loc_param, scale = scale_param)
    m4 <- linmod(df = df, form = f4, location = loc_param, scale = scale_param)
    m5 <- linmod(df = df, form = f5, location = loc_param, scale = scale_param)
    
    # Calculate PSIS-LOO for each model
    loo1 <- loo(m1)
    loo2 <- loo(m2)
    loo3 <- loo(m3)
    loo4 <- loo(m4)
    loo5 <- loo(m5)
    comp <- loo_compare(loo1, loo2, loo3, loo4, loo5)
    
    # Calculate WAIC for each
    waic_df = data.frame(model = c('m1', 'm2', 'm3', 'm4', 'm5'),
                         waic = c(waic(m1)$estimates[3,1], waic(m2)$estimates[3,1], waic(m3)$estimates[3,1],
                                  waic(m4)$estimates[3,1], waic(m5)$estimates[3,1]))
    waic_df <- arrange(waic_df, waic)
    
    # Update dataframe with results
    sim_df$LOO_selection[i] <- row.names(comp)[1]
    sim_df$LOO_elpd_diff[i] <- comp[2,1]
    sim_df$WAIC_selection[i] <- as.character(waic_df$model[1])
    sim_df$WAIC_diff[i] <- waic_df$waic[1] - waic_df$waic[2]
  }
  return(sim_df)
}

# Sim 1 uses an informative, correct prior
sim1.3 <- sim_fun3(reps = 100, n = 20, loc_param = loc2, scale_param = c(2,2,2,2,2))
# save(sim1.3, file = 'sim1_3.Rd')
# load('sim1_3.Rd')
table(sim1.3$LOO_selection, sim1.3$WAIC_selection)

# Sim 2 uses an uninformative, correct prior
sim2.3 <- sim_fun3(reps = 100, n = 10, loc_param = loc2, scale_param = c(10,10,10,10,10))
# save(sim2.3, file = 'sim2_3.Rd')
# load('sim2_3.Rd')
table(sim2.3$LOO_selection, sim2.3$WAIC_selection)

# Sim 3 uses the true parameters for a prior
sim3.3 <- sim_fun3(reps = 100, n = 10, loc_param = loc1, scale_param = c(2,2,2,2,2))
# save(sim3.3, file = 'sim3_3.Rd')
# load('sim3_3.Rd')
table(sim3.3$LOO_selection, sim3.3$WAIC_selection)
