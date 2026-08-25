# log sum exponential function -> solves the over flow issue of a number
import math
def log_sum_exp(vector):
  
  # find the max value in the vector
  maximum = vector[0]
  for number in vector:
    if number > maximum:
      maximum = number
  
  # find exp^(xi - max)
  total = 0
  for number in vector:
    total += math.exp(number - maximum)
  
  return (maximum + total)

result = log_sum_exp([100, 200, 300, 400, 500])
print(result)
