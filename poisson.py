import marimo

__generated_with = "0.1.64"
app = marimo.App()


@app.cell
def __(mo):
    mo.md("# Dimensioning internet servers")
    return


@app.cell
def __(mo):
    mo.md("The poisson distribution models the probability of arrivals based on average arrival statistics and the assumption that each arrival event are independent. There are more sophisticated models for internet traffic, but the poisson model is usefull as a simple approximation. [See more on wikipedia](https://en.wikipedia.org/wiki/Poisson_distribution).")
    return


@app.cell
def __(mo):
    mo.md("![poisson function](https://wikimedia.org/api/rest_v1/media/math/render/svg/7a0693ecaa606e3878dfa9a85552d357c690ffb9)")
    return


@app.cell
def __(factorial, np, plt):
    t = np.arange(0, 100, 0.1)
    d = np.exp(-10)*np.power(10, t)/factorial(t)

    plt.plot(t, d)
    plt.gca()
    return d, t


@app.cell
def __(mo):
    mo.md("For larger numbers this equation becomes unruly because of the factorial. Instead we can approximate it with the gaussian distribution with the mean still being the mean number of arrivals and the standard deviation being the square root of the mean number of arrivals")
    return


@app.cell
def __(mo):
    mo.md("Specify max users per hour and sigma. Sigma is the number of standard deviations from average you want to be able to handle. See the uptime calculation below.")
    return


@app.cell
def __(mo):
    userprhour = mo.ui.slider(start=500000, stop=10000000, step=500000, label="users per hour")
    sigma = mo.ui.slider(start=0, stop=6, step=0.5, label="sigma")

    [userprhour, sigma]
    return sigma, userprhour


@app.cell
def __(np, plt, sigma, userprhour):
    # Parameters for the Gaussian distribution
    mean = userprhour.value / 3600
    std_deviation = np.sqrt(mean)

    # Generating points on the x axis between -4 and 4
    x = np.linspace(0, 3300, 1000)

    # Calculating the Gaussian distribution values for each x
    y = (1 / (np.sqrt(2 * np.pi) * std_deviation)) * np.exp(
        -0.5 * ((x - mean) / std_deviation) ** 2
    )

    # Creating the plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Gaussian Distribution\nMean = 0, Std Dev = 1")
    plt.vlines(
        mean + np.sqrt(mean) * sigma.value, 0, 0.03, colors="k", linestyles="solid"
    )
    plt.title("Gaussian Distribution")
    plt.xlabel("Requests per second")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.gca()
    return mean, std_deviation, x, y


@app.cell
def __(mean, mo):
    mo.md(f"Users per second (mean): {mean}")
    return


@app.cell
def __(mean, mo, sigma, std_deviation):
    mo.md(f"Users per second (mean + sigma): {mean + sigma.value * std_deviation}")
    return


@app.cell
def __():
    import marimo as mo

    from scipy.special import factorial
    import numpy as np
    import matplotlib.pyplot as plt
    return factorial, mo, np, plt


if __name__ == "__main__":
    app.run()
