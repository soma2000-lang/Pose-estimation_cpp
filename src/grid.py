import cv2

# Load the undistorted image
img = cv2.imread("/home/smajumder/solutions/Solution/solution/debug/find_coordinates_here.png")
if img is None:
    print("[ERROR] Could not find debug/find_coordinates_here.png. Run your frame extraction first.")
    exit()

h, w = img.shape[:2]

# Drawing vertical lines & numbers (X-axis) every 50 pixels
for x in range(0, w, 50):
    is_major = (x % 100 == 0)
    color = (0, 0, 255) if is_major else (0, 80, 180) # Red for 100s, orange for 50s
    cv2.line(img, (x, 0), (x, h), color, 1)
    if is_major:
        cv2.putText(img, str(x), (x + 3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

# Draw horizontal lines & numbers (Y-axis) every 50 pixels
for y in range(0, h, 50):
    is_major = (y % 100 == 0)
    color = (0, 255, 0) if is_major else (0, 180, 80) # Bright green for 100s, dark green for 50s
    cv2.line(img, (0, y), (w, y), color, 1)
    if is_major:
        cv2.putText(img, str(y), (5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

# Saving the grid-mapped reference image
output_path = "debug/image_with_grid1.png"
cv2.imwrite(output_path, img)
print(f"[SUCCESS] Grid image created at: {output_path}")