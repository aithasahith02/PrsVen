import matplotlib.pyplot as plt
import json

with open("scripts/api/sample_output.json", "r") as f:
    output = json.load(f)

times, probs = zip(*output["survival_curve"])
plt.plot(times, probs)
plt.xlabel("Time (Months)")
plt.ylabel("Survival Probability")
plt.title(f"Predicted Survival Curve\n(Expected Survival: {output['expected_survival_months']} months)")
plt.grid(True)
plt.show()
