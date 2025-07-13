import os
import csv
from collections import Counter
from datetime import datetime

REPORT_FOLDER = "reports"

def load_reports():
    if not os.path.exists(REPORT_FOLDER):
        return []

    report_data = []

    for file in os.listdir(REPORT_FOLDER):
        if not file.endswith(".csv"):
            continue

        reg_id = file[:-4]  # Remove .csv extension
        file_path = os.path.join(REPORT_FOLDER, file)

        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                records = [row for row in reader if len(row) >= 6]

            if not records:
                continue

            last_10 = records[-10:]
            emotions = [row[1] for row in last_10]
            posture_flags = [row[3] for row in last_10]
            attentions = [row[5] for row in last_10]
            confs = [float(row[2]) for row in last_10]

            emotion_counter = Counter(emotions)
            posture_issues = posture_flags.count("Slouching")
            inattention = attentions.count("Inattentive")

            report_data.append({
                "id": reg_id,
                "student_name": reg_id,
                "registration_id": reg_id,
                "report_date": last_10[-1][0],
                "sessions_count": len(records),
                "avg_attention": attention_score(attentions),
                "avg_emotion_score": sum(confs) / len(confs),
                "posture_issues": posture_issues,
                "download_url_csv": f"/download/{reg_id}.csv",
                "download_url_pdf": f"/download/{reg_id}.pdf",  # Placeholder for future PDF support
            })
        except Exception as e:
            print(f"[ERROR] Couldn't read report for {reg_id}: {e}")

    return sorted(report_data, key=lambda x: x["report_date"], reverse=True)


def attention_score(attentions: list[str]):
    if not attentions:
        return 0
    focused_count = attentions.count("Focused")
    return round((focused_count / len(attentions)) * 100, 1)
