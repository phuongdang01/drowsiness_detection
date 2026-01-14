import mediapipe as mp
print("dir(mp):", dir(mp))
try:
    import mediapipe.python.solutions as solutions
    print("Explicit import of solutions worked")
    print("solutions dir:", dir(solutions))
except ImportError as e:
    print("Explicit import failed:", e)

try:
    print("mp.solutions:", mp.solutions)
except AttributeError:
    print("mp.solutions not found")
