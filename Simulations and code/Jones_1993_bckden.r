if(!require('pracma')) {
  install.packages('pracma')
  library('pracma')
}

if(!require('evmix')) {
  install.packages('evmix')
  library('evmix')
}


x = linspace(0,300,100)
y = x^2
plot(x,y)


kern_center = c(0)
bcmethod = "simple"
proper = FALSE


# y = dbckden(x,kerncentres = kern_center, bcmethod = bcmethod, proper = proper, bw = 40)

# # plot.new()
# plot(x,y, type = "l")




ka0_m <- function(truncpoint, kernel = "gaussian") {

#   check.kernel(kernel)
  
  kernel = ifelse(kernel == "rectangular", "uniform", kernel)
  kernel = ifelse(kernel == "normal", "gaussian", kernel)

  switch(kernel,
    gaussian = kpgaussian(truncpoint),
    uniform = kpuniform(truncpoint),
    triangular = kptriangular(truncpoint),
    epanechnikov = kpepanechnikov(truncpoint),
    biweight = kpbiweight(truncpoint),
    triweight = kptriweight(truncpoint),
    tricube = kptricube(truncpoint),
    parzen = kpparzen(truncpoint),
    cosine = kpcosine(truncpoint),
    optcosine = kpoptcosine(truncpoint))
}

ka0_m(0)

# bckdenxsimple(50,0,40)

truncpoint = x / 40
a0 = kpgaussian(truncpoint)
a1 = -dnorm(truncpoint)
a2 = (a0 + truncpoint*a1)


denom = (a2*a0 - a1^2)
lx = a2/denom
mx = a1/denom

x_i = 2
u = (x - x_i) / 40

y = (lx - mx*u)*(dnorm(u)) / 40

plot(x,y)



lambda = 40

truncpoint = x/lambda

kdgaussian(u)

# ka1 <- function(truncpoint, kernel = "gaussian") {
  
#   check.kernel(kernel)
  
#   kernel = ifelse(kernel == "rectangular", "uniform", kernel)
#   kernel = ifelse(kernel == "normal", "gaussian", kernel)

#   if (kernel != "gaussian") truncpoint = pmax(pmin(truncpoint, 1), -1)
  
#   switch(kernel,
#     gaussian = -dnorm(truncpoint),
#     uniform = (truncpoint^2 - 1)/4,
#     triangular = ifelse(truncpoint <= 0, (3*truncpoint^2 + 2*truncpoint^3 - 1)/6, (3*truncpoint^2 - 2*truncpoint^3 - 1)/6),
#     epanechnikov = 3*(2*truncpoint^2 - truncpoint^4 - 1)/16,
#     biweight = 5*(3*truncpoint^2 - 3*truncpoint^4 + truncpoint^6 - 1)/32,
#     triweight = 35*(4*truncpoint^2 - 6*truncpoint^4 + 4*truncpoint^6 - truncpoint^8 - 1)/256,
#     tricube = ifelse(truncpoint <= 0, 70*(truncpoint^2/2 + 3*truncpoint^5/5 + 3*truncpoint^8/8 + truncpoint^11/11 + 0.1 - 3/8 + 1/11)/81,
#       70*(truncpoint^2/2 - 3*truncpoint^5/5 + 3*truncpoint^8/8 - truncpoint^11/11 + 0.1 - 3/8 + 1/11)/81),
#     parzen = ifelse(truncpoint < -0.5, 32*truncpoint^5 + 120*truncpoint^4 + 160*truncpoint^3 + 80*truncpoint^2 - 8,
#       ifelse((truncpoint >= -0.5) & (truncpoint < 0), -96*truncpoint^5 - 120*truncpoint^4 + 40*truncpoint^2 - 7,
#       ifelse((truncpoint >= 0) & (truncpoint < 0.5), 96*truncpoint^5 - 120*truncpoint^4 + 40*truncpoint^2 - 7,
#       -32*truncpoint^5 + 120*truncpoint^4 - 160*truncpoint^3 + 80*truncpoint^2 - 8)))/60,
#     cosine = (truncpoint^2/4 - 0.25 + truncpoint*sin(pi*truncpoint)/2/pi + (cos(pi*truncpoint) + 1)/2/pi/pi),
#     optcosine = (truncpoint*sin(pi*truncpoint/2) - 1)/2 + cos(pi*truncpoint/2)/pi)
# }


# ka1(0)


# ka2 <- function(truncpoint, kernel = "gaussian") {
  
#   check.kernel(kernel)
  
#   kernel = ifelse(kernel == "rectangular", "uniform", kernel)
#   kernel = ifelse(kernel == "normal", "gaussian", kernel)

#   if (kernel != "gaussian") truncpoint = pmax(pmin(truncpoint, 1), -1)

#   switch(kernel,
#     gaussian = (ka0(truncpoint) + truncpoint*ka1(truncpoint)),
#     uniform = (truncpoint^3 + 1)/6,
#     triangular = ifelse(truncpoint <= 0, (4*truncpoint^3 + 3*truncpoint^4 + 1)/12, (4*truncpoint^3 - 3*truncpoint^4 + 1)/12),
#     epanechnikov = (5*truncpoint^3 - 3*truncpoint^5 + 2)/20,
#     biweight = (5*truncpoint^3 - 6*truncpoint^5 + 15*truncpoint^7/7 + 8/7)/16,
#     triweight = (35*truncpoint^3/3 - 21*truncpoint^5 + 15*truncpoint^7 - 35*truncpoint^9/9 + 16/9)/32,
#     tricube = ifelse(truncpoint <= 0, 70*(truncpoint^3/3 + truncpoint^6/2 + truncpoint^9/3 + truncpoint^12/12 + 1/12)/81,
#       70*(truncpoint^3/3 - truncpoint^6/2 + truncpoint^9/3 - truncpoint^12/12 + 1/12)/81),
#     parzen = ifelse(truncpoint < -0.5, 2/45 + 8*truncpoint^3/9 + 2*truncpoint^4 + 8*truncpoint^5/5 + 4*truncpoint^6/9,
#       ifelse((truncpoint >= -0.5) & (truncpoint < 0), 1/24 + 4*truncpoint^3/9 - 8*truncpoint^5/5 - 4*truncpoint^6/3,
#       ifelse((truncpoint >= 0) & (truncpoint < 0.5), 1/24 + 4*truncpoint^3/9 - 8*truncpoint^5/5 + 4*truncpoint^6/3,
#       7/180 + 8*truncpoint^3/9 - 2*truncpoint^4 + 8*truncpoint^5/5 - 4*truncpoint^6/9))),
#     cosine = ((truncpoint^3 + 1)/6 + truncpoint^2*sin(pi*truncpoint)/2/pi + (truncpoint*cos(pi*truncpoint) - 1)/pi/pi - sin(pi*truncpoint)/pi/pi/pi),
#     optcosine = (truncpoint^2*sin(pi*truncpoint/2) + 1)/2 + 2*truncpoint*cos(pi*truncpoint/2)/pi - 4*(sin(pi*truncpoint/2) + 1)/pi/pi)
# }


# ka2(0)

