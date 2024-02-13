binary_search <- function(sorted_array, x) {
    # return idx, value s.t. sortedArray[idx-1] < x <= sortedArray[idx]
    # If no such idx exists, return (0,x)

    n <- length(sorted_array)

    if (x <= sorted_array[1]) {
        return(c(1, sorted_array[1]))
    }

    if (x > sorted_array[n]) {
        return(c(0, x))
    }

    a <- 1
    b <- n

    idx <- (a + b) %/% 2

    while (a + 1 < b) {
        idx <- (a + b) %/% 2
        if (x <= sorted_array[idx]) {
            b <- idx
        } else {
            a <- idx
        }
    }

    return(c(b, sorted_array[b]))
}


# Initialize stuff
n <- 1000 # Number of simulations
t_arrival <- NULL

r_process_centrifuge <- 2 # [kg/s]




simulation <- function() {
    # Monte-carlo simulation
    # Draw rate and for how long, production from fermentation
    t_ferment <- rgamma(1, 10, 1)
    r_ferment <- rgamma(1, 100, 2)

    m_batch <- t_ferment * r_ferment



    # after starting receiving product, it is handled by a mixer
    # until it can be processed.
    # Assuming that no problem has occured before, it can directly be processed

    # the product is then handled by a centrifuge, splitting the product each
    # part handled separately. One part by a manual worker and other part by
    # robots, until joined back together

    # The next part of the process is the centrifuge. At any point, it can break
    # down, resulting in some down time t_down until processing again. After
    # restart, must reboot before can process.

    # At any time t since last reboot, has some distribution f, assuming
    # exponential initially due to memorylessnes

    # in the limit, the remanining time untill break is distributed with
    # lim_[t -> oo] P(gamma_t < x) = int_0^x (1-F(u)) du



    # calculate remaining time until finished production (of batch)
    t_remaining_centrifuge <- m_batch / r_process_centrifuge

    # How many break downs will happen
    t_break_centrifuge <- rexp(n = 100, rate = 100)
    temp <- cumsum(t_break_centrifuge)
    temp2 <- binary_search(temp, t_remaining_centrifuge)

    while (temp2[1] == 0) {
        t_break_centrifuge <- c(t_break_centrifuge, rexp(n = 10, rate = 100))
        temp <- cumsum(t_break_centrifuge)
        temp2 <- binary_search(temp, t_remaining_centrifuge)
    }


    return(temp2[2])

    # If more time than until breaking, simulate restart and breaking of
    # system
    # if (t_remaining_centrifuge >= t_break_centrifuge) {

    # }
}

n_breakdowns <- replicate(n, simulation())

hist(n_breakdowns)
