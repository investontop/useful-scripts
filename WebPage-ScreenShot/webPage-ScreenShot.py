import os
import time
from datetime import datetime
import pyautogui
from PIL import Image
import pygetwindow as gw

# === Config ===
scroll_times = 5
scroll_pause = 1.0
save_path = r'D:\FAMILY\1 Ramanathan\My work\S&S\01 Hard Times'
base_name = 'HSEC'
today_str = datetime.now().strftime('%Y%m%d')
file_name = f'{base_name}-{today_str}.jpeg'
output_file = os.path.join(save_path, file_name)

# === Locate Brave Window ===
print("Looking for Brave browser window...")
brave_windows = [w for w in gw.getWindowsWithTitle('Brave') if w.visible]

if not brave_windows:
    raise Exception("Brave browser window not found. Please open Brave and load your page.")

brave_win = brave_windows[0]
brave_win.activate()
time.sleep(2)

# Get window coordinates
left, top = brave_win.left, brave_win.top
width, height = brave_win.width, brave_win.height

# Optional crop: reduce top to skip title bar (adjust if needed)
crop_top = 80  # Usually title bar height
region = (left, top + crop_top, width, height - crop_top)

print(f"Using region: {region}")

# === Screenshot logic ===
if not os.path.exists(save_path):
    os.makedirs(save_path)

print("Focus on the Brave window. Capturing in 5 seconds...")
time.sleep(5)

screenshots = []

for i in range(scroll_times):
    print(f"Capturing screenshot {i+1} of {scroll_times}...")
    screenshot = pyautogui.screenshot(region=region)
    temp_file = os.path.join(save_path, f"temp_{i}.png")
    screenshot.save(temp_file)
    screenshots.append(temp_file)

    pyautogui.scroll(-800)
    time.sleep(scroll_pause)

# === Stitch vertically ===
print("Stitching screenshots...")
images = [Image.open(img) for img in screenshots]
stitched_height = sum(img.height for img in images)
stitched_img = Image.new("RGB", (images[0].width, stitched_height))

y_offset = 0
for img in images:
    stitched_img.paste(img, (0, y_offset))
    y_offset += img.height

stitched_img.save(output_file, "JPEG", quality=90)
print(f"Saved final screenshot to: {output_file}")

# Cleanup
for img in screenshots:
    os.remove(img)
