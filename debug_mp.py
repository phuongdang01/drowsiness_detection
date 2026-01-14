import mediapipe
print("dir(mediapipe):", dir(mediapipe))
try:
    import mediapipe.python.solutions
    print("Imported mediapipe.python.solutions successfully")
except ImportError as e:
    print("Failed to import mediapipe.python.solutions:", e)

try:
    print("mediapipe.solutions:", mediapipe.solutions)
except AttributeError as e:
    print("mediapipe.solutions access failed:", e)
