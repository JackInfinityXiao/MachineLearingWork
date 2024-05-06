import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
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


svm_model1 = svm.SVC(C=1.0, kernel="linear", gamma="scale")
svm_model2 = svm.SVC(C=1.0, kernel="poly", gamma="scale")
svm_model3 = svm.SVC(C=1.0, kernel="rbf", gamma="scale")

# Generate the data (1000 datapoints)
X, labels = make_moons_3d(n_samples=1000, noise=0.2)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)


model1 = svm_model1
model1.fit(X_train, y_train)
model2 = svm_model2
model2.fit(X_train, y_train)
model3 = svm_model3
model3.fit(X_train, y_train)

X1, labels1 = make_moons_3d(n_samples=500, noise=0.2)

predictions1 = model1.predict(X1)
accuracy_test1 = accuracy_score(labels1, predictions1)
print("Accuracy1:", accuracy_test1)


predictions2 = model2.predict(X1)
accuracy_test2 = accuracy_score(labels1, predictions2)
print("Accuracy2:", accuracy_test2)

predictions3 = model3.predict(X1)
accuracy_test3 = accuracy_score(labels1, predictions3)
print("Accuracy3:", accuracy_test3)


def plot_3d_scatter(X, y, predictions, title, ax):

    ax.scatter(
        X1[predictions == 0][:, 0],
        X1[predictions == 0][:, 1],
        X1[predictions == 0][:, 2],
        c="b",
        label="Class 0",
    )

    ax.scatter(
        X1[predictions == 1][:, 0],
        X1[predictions == 1][:, 1],
        X1[predictions == 1][:, 2],
        c="g",
        label="Class 1",
    )

    wrong_indices = (predictions != y).nonzero()[0]

    ax.scatter(
        X[wrong_indices, 0],
        X[wrong_indices, 1],
        X[wrong_indices, 2],
        c="red",
        marker="x",
        label="Misclassified Points",
    )

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(1, 3, 1, projection="3d")
plot_3d_scatter(X1, labels1, predictions1, "Linear Kernel", ax1)
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
plot_3d_scatter(X1, labels1, predictions2, "Polynomial Kernel", ax2)
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
plot_3d_scatter(X1, labels1, predictions3, "RBF Kernel", ax3)
plt.tight_layout()
plt.show()
