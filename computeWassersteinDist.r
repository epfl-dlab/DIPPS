# Compute the wassersetein distance between two distribution

args <- commandArgs(trailingOnly = TRUE)

src_file <- args[1]
tgt_file <- args[2]
src_weights_file <- args[3]
tgt_weights_file <- args[4]
experiment_dir <- args[5]
src_file <- paste(experiment_dir, src_file, sep = "/")
tgt_file <- paste(experiment_dir, tgt_file, sep = "/")
src_weights_file <- paste(experiment_dir, src_weights_file, sep = "/")
tgt_weights_file <- paste(experiment_dir, tgt_weights_file, sep = "/")
prefix <- strsplit(src_file, "/")[[1]]
prefix <- strsplit(prefix[length(prefix)], "_")[[1]][1]
warning(prefix)
library(transport)
library(MASS)

# Read in opt and non-opt
src <- read.csv(src_file)
tgt <- read.csv(tgt_file)

src_weights <- read.csv(src_weights_file)
tgt_weights <- read.csv(tgt_weights_file)
src_weights <- as.matrix(src_weights) 
src_weights <- as.numeric(src_weights)
tgt_weights <- as.matrix(tgt_weights) 
tgt_weights <- as.numeric(tgt_weights)

# Type conversion
wpp_src <- wpp(src, src_weights)
wpp_tgt <- wpp(tgt, tgt_weights)

plan <- transport(wpp_src, wpp_tgt, p = 1, method = 'networkflow', threads = 1)
res <- wasserstein(wpp_src, wpp_tgt, p = 1, tplan = plan, prob = TRUE)
print(res)

plan_file <- paste(prefix, "plan.csv", sep = "_")
plan_file <- paste(experiment_dir, plan_file, sep = "/")
warning(plan_file)
write.csv(plan, plan_file, row.names = FALSE)
