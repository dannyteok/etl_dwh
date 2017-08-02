setwd("/Projects/smartfocus-olympus-rnd-dataprocessing/")
rm(list = ls())
graphics.off()

df = read.csv("data.csv")

df = df[df$rssi < 0,]

n <- boxplot(rssi ~ beacon, data = df, boxwex=0.25)
# make a list of your data per group
a <- split(df, df$beacon)
# Go through the list and exclude the outliers
a <- lapply(1:nlevels(df$beacon), function(i,x)
  subset(x[[i]], rssi < n$stats[4, i] & rssi > n$stats[2, i]), a)
# Transform to a data.frame again
df_filtered <- do.call(rbind, a)

boxplot(rssi ~ beacon, data = df_filtered, add=T, col=2, at =1:nlevels(df$beacon) + 0.2, boxwex=0.25)

# boxplot(df$rssi ~ df$beacon)
#
# library(doBy)
#
# print(summaryBy(rssi~beacon, data = df, FUN = c(mean, median, sd)))
#
# library(plyr)
#
# df_quantile = do.call("rbind", tapply(df$rssi, df$beacon, quantile))
#
# filtered = function(x) {
#   lowerq = quantile(x)[2]
#   upperq = quantile(x)[4]
#   iqr = upperq - lowerq
#
#   mild.threshold.upper = (iqr * 1.5) + upperq
#   mild.threshold.lower = lowerq - (iqr * 1.5)
#
#   extreme.threshold.upper = (iqr * 3) + upperq
#   extreme.threshold.lower = lowerq - (iqr * 3)
#
#   x = x[x > extreme.threshold.lower & x < extreme.threshold.upper]
#   return(x)
# }
#
# filtData = tapply(df$rssi, df$beacon, filtered)
