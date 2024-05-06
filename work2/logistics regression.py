import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def make_moons_3d(n_samples=500, noise=0.1):
    # Generate the original 2D make_moons data
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # Adding a sinusoidal variation in the third dimension

    # Concatenating the positive and negative moons with an offset and noise
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # Adding Gaussian noise
    X += np.random.normal(scale=noise, size=X.shape)

    return X, y


# Generate the data (1000 datapoints)
X, labels = make_moons_3d(n_samples=1000, noise=0.2)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Logistic
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Generate the data (500 datapoints)
X1, labels1 = make_moons_3d(n_samples=500, noise=0.2)

y1_pred = logreg.predict(X1)
accuracy_test = accuracy_score(labels1, y1_pred)
print("Accuracy:", accuracy_test)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")


ax.scatter(
    X1[y1_pred == 0][:, 0],
    X1[y1_pred == 0][:, 1],
    X1[y1_pred == 0][:, 2],
    c="b",
    label="Class 0",
)

ax.scatter(
    X1[y1_pred == 1][:, 0],
    X1[y1_pred == 1][:, 1],
    X1[y1_pred == 1][:, 2],
    c="g",
    label="Class 1",
)

wrong_indices = (y1_pred != labels1).nonzero()[0]
ax.scatter(
    X1[wrong_indices, 0],
    X1[wrong_indices, 1],
    X1[wrong_indices, 2],
    c="red",
    marker="x",
    label="Misclassified Points",
)

ax.legend()
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Scatter Plot")
plt.show()
