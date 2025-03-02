import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from losses import get_loss_fn

params = {
    "lines.linewidth": 1.5,
    "legend.fontsize": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
}
matplotlib.rcParams.update(params)

plt.rc("font", family="Times New Roman")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--noise_scale", type=float, default=2.0, help="Scale of the heavy-tailed noise"
)
parser.add_argument(
    "--noise_prob",
    type=float,
    default=0.2,
    help="Probability of adding heavy-tailed noise",
)
parser.add_argument(
    "--noise_threshold",
    type=float,
    default=1.5,
    help="X threshold for increased noise probability",
)
parser.add_argument(
    "--n_samples", type=int, default=10, help="Number of training samples"
)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)


def ground_truth(x):
    """Ground truth function"""
    return x**2


def generate_heavy_tailed_noise(
    x_values, scale=1.0, base_prob=0.2, threshold=1.5, seed=None
):
    """Generate heavy-tailed noise using Student's t-distribution with position-dependent sign"""
    if seed is not None:
        np.random.seed(seed)

    size = len(x_values)

    rng = np.random.RandomState()
    noise_mask = rng.random(size) < base_prob

    noise = np.abs(np.random.standard_t(df=2, size=size)) * scale
    noise[~noise_mask] = 0

    noise[x_values > threshold] *= -1
    return noise


def generate_dataset(n_samples, noise_scale, noise_prob, noise_threshold, seed=None):

    if seed is not None:
        np.random.seed(seed)

    x = np.random.uniform(0, 3, n_samples)

    y_true = ground_truth(x)

    noise = generate_heavy_tailed_noise(
        x, scale=noise_scale, base_prob=noise_prob, threshold=noise_threshold, seed=seed
    )
    y_noisy = y_true + noise

    return list(zip(x, y_noisy))


COLORS = {
    "true": "#2C3E50",
    "data": "#E74C3C",
    "mse": "#3498DB",
    "rankcosine": "#2ECC71",
}


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def train_model(data_points, loss_fn, learning_rate=0.001, epochs=1000):
    x = torch.tensor([[x] for x, _ in data_points], dtype=torch.float32)
    y = torch.tensor([[y] for _, y in data_points], dtype=torch.float32)

    model = LinearModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    w = model.linear.weight.item()
    b = model.linear.bias.item()

    return model, w, b


def mse_closed_form(data_points):
    X = np.array([[x] for x, _ in data_points])
    y = np.array([y for _, y in data_points])

    X_b = np.c_[np.ones(X.shape[0]), X]

    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    b, w = theta[0], theta[1]

    model = LinearModel()
    with torch.no_grad():
        model.linear.weight.data = torch.tensor([[w]], dtype=torch.float32)
        model.linear.bias.data = torch.tensor([b], dtype=torch.float32)

    return model, w, b


def plot_models(data, models_dict, equations):
    plt.figure(figsize=(8, 6))
    plt.rc("font", family="Times New Roman")

    plt.gcf().patch.set_facecolor("white")
    plt.gca().set_facecolor("white")

    plt.grid(True, linestyle="--", color="black", alpha=0.2)

    x_true = np.linspace(0, 3, 100)
    y_true = ground_truth(x_true)
    plt.plot(
        x_true,
        y_true,
        "-",
        color=COLORS["true"],
        label="Ground Truth: f(x) = x²",
        linewidth=2.5,
        alpha=0.8,
    )

    x_data, y_data = zip(*data)
    plt.scatter(
        x_data,
        y_data,
        c=COLORS["data"],
        s=100,
        label="Training Data",
        alpha=0.7,
        edgecolors="white",
    )

    x_pred = np.linspace(0, 3, 100)
    x_pred_tensor = torch.tensor([[x] for x in x_pred], dtype=torch.float32)

    for name, model in models_dict.items():
        with torch.no_grad():
            y_pred = model(x_pred_tensor).numpy()
        plt.plot(
            x_pred,
            y_pred,
            "-",
            color=COLORS[name.lower()],
            label=f"{name}: {equations[name]}",
            linewidth=2,
            alpha=0.8,
        )

    plt.xlabel("$x$")
    plt.ylabel("$\hat{f}(x)$")
    title = (
        f"Comparison of Different Loss Functions\n"
        f"(noise scale={args.noise_scale}, base_prob={args.noise_prob}, "
        f"threshold={args.noise_threshold}, n={args.n_samples})"
    )
    plt.title(title)

    plt.legend(frameon=True, fancybox=True, shadow=True)

    plt.xlim(0, 3.1)
    y_min = (
        min(
            min(
                min(
                    [
                        model(torch.tensor([[3.0]])).item()
                        for model in models_dict.values()
                    ]
                ),
                min(y_data),
            ),
            0,
        )
        - 1
    )
    y_max = (
        max(
            max(
                max(
                    [
                        model(torch.tensor([[3.0]])).item()
                        for model in models_dict.values()
                    ]
                ),
                max(y_data),
            ),
            9,
        )
        + 1
    )
    plt.ylim(y_min, y_max)

    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    plt.tight_layout()
    plt.savefig(
        f"Simulation_noise{args.noise_scale}_prob{args.noise_prob}_thresh{args.noise_threshold}_n{args.n_samples}.png"
    )
    print(
        f"Simulation_noise{args.noise_scale}_prob{args.noise_prob}_thresh{args.noise_threshold}_n{args.n_samples}.png"
    )


def format_equation(w, b):
    if b >= 0:
        return f"y = {w:.2f}x + {b:.2f}"
    else:
        return f"y = {w:.2f}x - {abs(b):.2f}"


if __name__ == "__main__":
    data = generate_dataset(
        n_samples=args.n_samples,
        noise_scale=args.noise_scale,
        noise_prob=args.noise_prob,
        noise_threshold=args.noise_threshold,
        seed=args.seed,
    )

    mse_loss = nn.MSELoss()
    rankcosine_loss = get_loss_fn("rankcosine")
    listnet_loss = get_loss_fn("listnet")

    print("Training with MSE loss:")
    # model_mse, w_mse, b_mse = train_model(data, mse_loss)
    # mse_eq = format_equation(w_mse, b_mse)
    # print(f"MSE model: {mse_eq}\n")
    model_mse_closed, w_mse_closed, b_mse_closed = mse_closed_form(data)
    mse_closed_eq = format_equation(w_mse_closed, b_mse_closed)
    print(f"MSE closed-form solution: {mse_closed_eq}\n")

    print("Training with RankCosine loss:")
    model_rankcosine, w_rc, b_rc = train_model(data, rankcosine_loss)
    rc_eq = format_equation(w_rc, b_rc)
    print(f"RankCosine model: {rc_eq}\n")

    models = {
        "MSE": model_mse_closed,
        "RankCosine": model_rankcosine,
    }

    equations = {
        "MSE": mse_closed_eq,
        "RankCosine": rc_eq,
    }

    plot_models(data, models, equations)
