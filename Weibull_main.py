import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Example failure data (time to failure in hours)
failure_times = np.array([50, 120, 200, 250, 310, 400, 500, 600, 750, 900])

# Fit Weibull distribution to the data
shape, loc, scale = stats.weibull_min.fit(failure_times, floc=0)  # floc=0 forces the location to 0


# Generate Weibull probability plot data
sorted_data = np.sort(failure_times)
weibull_cdf = stats.weibull_min.cdf(sorted_data, shape, loc, scale)  # CDF of fitted Weibull distribution

# Convert CDF to Weibull probability scale (ln(-ln(1 - CDF)))
weibull_y = np.log(-np.log(1 - weibull_cdf))

# Convert time to log scale for Weibull plot
weibull_x = np.log(sorted_data)

# Plot Weibull Probability Plot
plt.figure(figsize=(8, 6))
plt.scatter(weibull_x, weibull_y, label="Failure Data", color='blue', marker='o')
plt.xlabel("ln(Time to Failure)")
plt.ylabel("ln(-ln(1 - CDF))")
plt.title("Weibull Probability Plot")
plt.grid()
plt.legend()
plt.show()

# Print Weibull parameters
print(f"Shape Parameter (β): {shape:.2f}")
print(f"Scale Parameter (η): {scale:.2f}")
