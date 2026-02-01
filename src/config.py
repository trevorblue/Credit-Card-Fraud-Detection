#----Configutation Flags----
RESOURCE_MODE = True # Optimise for 8gb RAM. Set to False if  hardware is upgraded

QML_ENABLED = True # For future use. Set to False to disable QML interface

SEED = 42 
# --- Reproducibility with SEED ---
# Why SEED = 42?
# --------------------------------
# In machine learning, many steps involve randomness:
# - Splitting data into train/test sets
# - Shuffling rows or sampling subsets
# - Initializing model weights (especially in neural nets / XGBoost)
# - Generating synthetic data (e.g., SMOTE)
# - Randomized hyperparameter searches
#
# Without fixing a seed, these steps will behave slightly differently
# each time you run the code → accuracy, loss curves, or splits will
# change unpredictably. This makes debugging and comparing models painful.
#
# By setting SEED = 42, we make runs reproducible.
#   • Data Splitting:
#       from sklearn.model_selection import train_test_split
#       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
#       → Ensures the same rows always go into train/test sets.
#
#   • Shuffling / Sampling:
#       import random; random.seed(SEED)
#       data.sample(5, random_state=SEED)
#       → Guarantees the same 5 rows are picked every time.
#
#   • Model Initialization:
#       import tensorflow as tf; tf.random.set_seed(SEED)
#       → Neural nets start with identical initial weights → stable accuracy/loss graphs.
#
#   • Oversampling (SMOTE):
#       from imblearn.over_sampling import SMOTE
#       sm = SMOTE(random_state=SEED)
#       → Produces the same synthetic samples across runs.
#
#   • Hyperparameter Tuning:
#       from sklearn.model_selection import GridSearchCV
#       grid = GridSearchCV(model, param_grid, cv=5, random_state=SEED)
#       → Ensures CV folds are shuffled consistently.
#
# ✅ Summary:
# SEED = 42 ensures reproducibility in ALL steps where randomness appears.
# That means: data splitting, shuffling, weight initialization,
# oversampling, and hyperparameter search.
# With it → running today, tomorrow, or next month gives identical results.
#
# ("42" is just a fun tradition from sci-fi; any number works.)

import numpy as np
import tensorflow as tf #for deep learning models e.e neural networks, GPU
import random #Python's biult in random number generator    
import os # for interactng with computer operating system

# Set all seeds for full reproducibility
#Random Seed Intuition ---
# Think of the random generator as a machine with gears inside.
# The gears are controlled by a mathematical formula.
# If you put the machine in exactly the same gear position, it will always
# churn out the same sequence of "random" numbers.
#
# The seed value (42, 123, etc.) is just a code you give the machine
# to set its internal gears in a certain way.
#
# So:
#   Seed = 42  → gears arranged one way → sequence A comes out.
#   Seed = 123 → gears arranged differently → sequence B comes out.
#
# IMPORTANT: The number itself (42, 123, etc.) does NOT show up in the output.
np.random.seed(SEED)#numpy's  built-in random module is completely separate from
tf.random.set_seed(SEED)#TensorFlow has its own random number generator
random.seed(SEED) #Python's built-in random module is separate from numpy's  
os.environ['PYTHONHASHSEED'] = str(SEED)
# 🔹 PYTHONHASHSEED (often confusing 😅):
#
# Background:
#   - In Python 3.3+, "hash randomization" was added for security.
#   - This means Python may randomize the order of things like dictionary keys.
#
# Example:
#   my_dict = {"a":1, "b":2, "c":3}
#   print(my_dict.keys())
#   Run 1 → ['a','b','c']
#   Run 2 → ['c','a','b']
#   (Same dict, but different order shown.)
#
# Problem:
#   - If your code depends on dictionary order (e.g., configs, batching data),
#     then results may differ between runs, even with NumPy/TF seeds set.
#
# What this line does:
#   os.environ['PYTHONHASHSEED'] = str(SEED)
#   👉 Forces Python to always use the same hashing method (no randomization).
#   👉 Guarantees consistent dictionary order and other hash-based operations.


#If PyTorch is added to this project later, it also has its own random generator.

#----OPTIONAL: ENVIRONEMT CHECK----
try:
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"[INFO] System RAM: {ram_gb:.2f} GB")
    if RESOURCE_MODE and ram_gb < 12:
        print("[INFO] RESOURCE MODE IS ON. Models will be optimised for lower memort usage.")
except ImportError:
    pass
print("Configuration loaded successfully.")

# 🔹 try–except block:
#    - Code inside "try" runs normally.
#    - If it fails (e.g., psutil not installed), "except" prevents a crash and just skips.
#
# 🔹 psutil:
#    - A library for system monitoring (CPU, memory, etc.).
#    - Here it checks how much RAM your system has.
#
# 🔹 ram_gb = psutil.virtual_memory().total / (1024 ** 3):
#    - Gets total RAM in bytes.
#    - Divides by (1024 ** 3) → converts bytes → gigabytes (GB).
#
# 🔹 The if statement:
#    - If your RAM < 12 GB AND RESOURCE_MODE is True →
#      print a warning that models will run in "lighter mode"
#      (useful for low-memory systems).
