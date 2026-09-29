import json

# 1. โหลดข้อมูลจากไฟล์ JSON
json_path = "Data/dataset_Player_1_cumulative.json"

try:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records successfully.\n")
except FileNotFoundError:
    print(f"Error: File not found at {json_path}")
    exit()

# 2. แยกข้อมูล Actual (ค่าจริง) และ Predicted (ค่าที่ AI ทำนาย)
# *อย่าลืมเปลี่ยนชื่อ key ให้ตรงกับโครงสร้างในไฟล์ JSON ของคุณ*
y_true = [item["actual"] for item in data]
y_pred = [item["predicted"] for item in data]

# 3. คำนวณความแม่นยำ (Accuracy) แบบ Pure Python
correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
total = len(y_true)
acc = (correct / total) * 100 if total > 0 else 0

print("=" * 40)
print(f" Overall Accuracy: {acc:.2f}%")
print("=" * 40)

if acc >= 80.0:
    print(" Pass! ความแม่นยำผ่านเกณฑ์ 80% เรียบร้อยแล้ว")
else:
    print(f" Need Improvement! ยังขาดอีก {80.0 - acc:.2f}% ถึงจะครบ 80%")

# 4. สร้าง Confusion Matrix แบบ Pure Python
labels = sorted(list(set(y_true + y_pred)))
label_to_index = {label: i for i, label in enumerate(labels)}

# สร้างตาราง Matrix ตารางเปล่า
cm = [[0] * len(labels) for _ in range(len(labels))]
for t, p in zip(y_true, y_pred):
    cm[label_to_index[t]][label_to_index[p]] += 1

print("\n--- Confusion Matrix ---")
print(f"{'Actual \\ Pred':<15}" + "".join([f"{str(l):>12}" for l in labels]))
for i, row in enumerate(cm):
    print(f"{str(labels[i]):<15}" + "".join([f"{val:>12}" for val in row]))