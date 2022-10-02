args <- commandArgs(trailingOnly = TRUE)
print(args)

src_file <- args[1]
tgt_file <- args[2]
weights_file <- args[3]
print(src_file)
library(transport)
library(MASS)

num_attributes = 10
# Read in opt and non-opt
src <- read.csv(src_file)
tgt <- read.csv(tgt_file)

# Normalize
weights_src <- as.matrix(read.csv(weights_file))
weights_src <- weights_src / norm(as.matrix(weights_src))

weights_tgt <- replicate(nrow(tgt), 1)
weights_tgt <- weights_tgt / norm(as.matrix(weights_tgt))

# Type conversion
wpp_opt <- wpp(src, weights_src)
wpp_all <- wpp(tgt, weights_tgt)

plan <- transport(wpp_opt, wpp_all, p = 2, method = 'networkflow')
val <- wasserstein(wpp_opt, wpp_all, p = 2, tplan = plan)
print(val)