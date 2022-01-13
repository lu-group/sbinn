import numpy as np
import pytorchsbinn 

gluc_data = np.hsplit(np.loadtxt("glucose.dat"),[1])
meal_data = np.hsplit(np.loadtxt("meal.dat"),[4])

t = gluc_data[0]
y = gluc_data[1]
meal_t = meal_data[0]
meal_q = meal_data[1]

pytorchsbinn.pinn(t[:1800], y[:1800], meal_t, meal_q)