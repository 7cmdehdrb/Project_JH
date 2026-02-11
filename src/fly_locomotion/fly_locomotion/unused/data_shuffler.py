import random

file = "/home/min/7cmdehdrb/fuck_flight/src/fly_locomotion/fly_locomotion/dataskeleton_data.txt"

with open(file, "r") as f:
    lines = f.readlines()

random.shuffle(lines)

output_file = file.replace(".txt", "_shuffled.txt")
with open(output_file, "w") as f:
    f.writelines(lines)

print(f"Shuffled {len(lines)} lines -> {output_file}")
