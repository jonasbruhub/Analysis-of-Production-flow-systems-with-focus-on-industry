library("evmix")
library("pracma")

x <- as.matrix(rbind(
  c(0.01, 0.4, 0.5, 0.6, 0.99),
  c(0.01, 0.6, 0.5, 0.4, 0.99)
))

# x <- data.frame(
#   x1 = c(0.000, 0.4, 0.5, 0.6, 0.99),
#   x2 = c(0.000, 0.6, 0.5, 0.4, 0.99)
# )

print(x[1, ])
print(x)

print(as.matrix(cbind(c(0.4), c(0.5))))

# xx <- seq(0, 1, 0.001)

# plot(xx, dbckden(xx, x, lambda = 0.3, bcmethod = "copula", xmax = 1))

# plot(xx, dbckden(xx, x, lambda = 0.3, bcmethod = "simple", ul = 1))

# plot(xx, pbckden(xx, x, lambda = 0.1, bcmethod = "copula", xmax = 1))


# print(
#   dbckden(
#     as.matrix(cbind(c(0.4), c(0.5))),
#     kerncentres = x,
#     lambda = 0.3,
#     bcmethod = "simple"
#   )
# )




print(
  dbckden(
    c(0.4, 0.5),
    kerncentres = x,
    lambda = 0.3,
    bcmethod = "copula",
    xmax = 1
  )
)


persp(x, y, z, col = "blue")


meshgrid_uniform <- meshgrid(seq(0, 1, 0.1), seq(0, 1, 0.1))
# meshgrid_uniform$X
# meshgrid_uniform$Y
# z <- meshgrid_uniform$X^2 + meshgrid_uniform$Y^2
print(as.matrix(rbind(
  as.vector(meshgrid_uniform$X),
  as.vector(meshgrid_uniform$Y)
)))
kerncentres <- as.matrix(rbind(
  as.vector(meshgrid_uniform$X),
  as.vector(meshgrid_uniform$Y)
))


kerncentres <- list(
  list(0.01, 0.4, 0.5, 0.6, 0.99),
  list(0.01, 0.6, 0.5, 0.4, 0.99)
)
print(kerncentres)

z <- dbckden(
  c(0.5),
  kerncentres = kerncentres,
  lambda = 0.3,
  bcmethod = "copula",
  xmax = 1
)
print(z)
print(dim(z))

persp(
  seq(0, 1, 0.01),
  seq(0, 1, 0.01),
  z,
  col = "blue"
)
