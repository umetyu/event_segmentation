from pathlib import Path

import matplotlib.pyplot as plt


loss_file = Path("loss_30.txt")

epochs = []
losses = []

for line in loss_file.read_text().splitlines():
    timestamp, loss, step = line.split()
    epochs.append(int(step) + 1)
    losses.append(float(loss))

best_index = min(range(len(losses)), key=losses.__getitem__)

plt.figure(figsize=(9, 5))
plt.plot(epochs, losses, marker="o", linewidth=2, label="Training loss")

plt.scatter(
    epochs[best_index],
    losses[best_index],
    color="red",
    zorder=3,
    label=f"Best: {losses[best_index]:.4f}",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("EV-LayerSegNet Training Loss — 30 Epochs")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve_30.png", dpi=200)
plt.show()

print(f"Initial loss: {losses[0]:.6f}")
print(f"Final loss:   {losses[-1]:.6f}")
print(f"Best loss:    {losses[best_index]:.6f}")
print(f"Best epoch:   {epochs[best_index]}")