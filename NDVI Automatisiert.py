import csv
import http.cookiejar
import os
import re
import shutil
import smtplib
import sys
from datetime import datetime
from email.message import EmailMessage
from io import BytesIO

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.colorbar as colorbar
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# -----------------------------------------------------------------------------
# Konfiguration
# -----------------------------------------------------------------------------
SOURCE_FOLDER_CANDIDATES = [
    r"C:\Users\trapp\OneDrive - Hochschule für Wirtschaft und Umwelt\Noir nachts",
    r"C:\Users\trapp\Hochschule für Wirtschaft und Umwelt\Langenbucher, Claudia - BraikeNoIRnachts",
]
SOURCE_FOLDER = next(
    (
        path
        for path in SOURCE_FOLDER_CANDIDATES
        if os.path.isdir(path) and os.listdir(path)
    ),
    SOURCE_FOLDER_CANDIDATES[0],
)
OUTPUT_FOLDER = r"C:\Users\trapp\Desktop\NDVI 2 Bilder beleuchtet"
AVG_LOG_PATH = os.path.join(OUTPUT_FOLDER, "ndvi_history.csv")
ALERT_EMAIL_TO = "......."
START_DATE = datetime.strptime("2026-09-07", "%Y-%m-%d").date()

# Telegram-Setup für Warnmeldungen
TELEGRAM_BOT_TOKEN = "8615026139:AAHsREFTKmL3Af-beWeLiCK4q01WIRLc9Xs"
TELEGRAM_CHAT_ID = "994328852"

# -----------------------------------------------------------------------------
# NDVI-Konfiguration (identisch zum bestehenden Skript)
# -----------------------------------------------------------------------------
NIR_CORRECTION_FACTOR = 0.9
RED_CORRECTION_FACTOR = 1.1
POSITIVE_SCALE_FACTOR = 2.0
NDVI_BINS = np.arange(-1.0, 1.0001, 0.02)
NDVI_LABELS = [f"{a:.2f}..{b:.2f}" for a, b in zip(NDVI_BINS[:-1], NDVI_BINS[1:])]
GRAY = np.array([0.6, 0.6, 0.6])
POS_COLORS = [
    (128 / 255, 0, 128 / 255), (1, 0, 0), (1, 1, 0),
    (0, 1, 0), (0, 0, 1),
]
CMAP_POS = LinearSegmentedColormap.from_list("ndvi_pos", POS_COLORS)


def ensure_output_folder():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def list_local_images():
    """Liefert alle Bilddateien aus dem lokal synchronisierten OneDrive-Ordner."""
    if not os.path.isdir(SOURCE_FOLDER):
        raise RuntimeError(
            f"Der lokale OneDrive-Ordner wurde nicht gefunden. Gesucht wurde: {SOURCE_FOLDER}. "
            "Bitte den Ordner synchronisieren oder den Pfad in SOURCE_FOLDER_CANDIDATES anpassen."
        )

    files = []
    for entry in os.listdir(SOURCE_FOLDER):
        full_path = os.path.join(SOURCE_FOLDER, entry)
        if os.path.isfile(full_path) and entry.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            files.append(entry)
    return sorted(files)


def parse_file_time(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2})[ _-]*(\d{1,2})[.:_-]*(\d{2})", filename, flags=re.I)
    if not match:
        return None
    date_value, hour, minute = match.groups()
    try:
        return datetime.strptime(f"{date_value} {int(hour):02d}:{int(minute):02d}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def build_target_names_by_time(files):
    """Sucht die Dateien für 23:20 und 23:30 und ordnet sie den Zielnamen zu."""
    targets = {}
    for file in files:
        name = os.path.basename(file)
        dt = parse_file_time(name)
        if dt is None:
            continue
        hour_minute = dt.strftime("%H:%M")
        if hour_minute == "23:20":
            targets["23:20"] = name
        elif hour_minute == "23:30":
            targets["23:30"] = name
    return targets


def copy_daily_images():
    """Kopiert alle gültigen Paarbilder ab dem 07.09.2026 in den Zielordner."""
    files = list_local_images()
    by_date = {}
    for file_name in files:
        dt = parse_file_time(file_name)
        if dt is None:
            continue
        date = dt.strftime("%Y-%m-%d")
        by_date.setdefault(date, {})[dt.strftime("%H:%M")] = file_name

    valid_dates = [
        date
        for date, times in by_date.items()
        if datetime.strptime(date, "%Y-%m-%d").date() >= START_DATE
        and "23:20" in times and "23:30" in times
    ]
    if not valid_dates:
        raise RuntimeError(
            "Im lokal synchronisierten Ordner wurden keine passenden Paare mit 23:20 und 23:30 ab dem 07.09.2026 gefunden. "
            "Prüfe die Dateinamen im Ordner."
        )

    for existing_file in os.listdir(OUTPUT_FOLDER):
        full_existing = os.path.join(OUTPUT_FOLDER, existing_file)
        if os.path.isfile(full_existing) and (
            existing_file.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
            or "_RED" in existing_file.upper()
            or "_NIR" in existing_file.upper()
            or "NDVI" in existing_file.upper()
        ):
            os.remove(full_existing)

    copied_paths = []
    for date in sorted(valid_dates):
        red_name = by_date[date]["23:20"]
        nir_name = by_date[date]["23:30"]
        red_target = os.path.join(OUTPUT_FOLDER, f"{date}_2320_RED.jpg")
        nir_target = os.path.join(OUTPUT_FOLDER, f"{date}_2330_NIR.jpg")
        shutil.copy2(os.path.join(SOURCE_FOLDER, red_name), red_target)
        shutil.copy2(os.path.join(SOURCE_FOLDER, nir_name), nir_target)
        copied_paths.extend([red_target, nir_target])

    return copied_paths


def download_daily_images():
    """Kompatibilitäts-Wrapper für das alte Skript."""
    return copy_daily_images()


def find_images_by_date(folder, keywords):
    images_by_date = {}
    date_pattern = re.compile(r"(20\d{2}-\d{2}-\d{2})")
    for filename in os.listdir(folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            continue
        name = filename.upper()
        date_match = date_pattern.search(filename)
        if date_match and any(keyword in name for keyword in keywords):
            images_by_date.setdefault(date_match.group(1), []).append(filename)
    return images_by_date


def class_index(ndvi):
    index = np.digitize(ndvi, NDVI_BINS, right=False) - 1
    return np.clip(index, 0, len(NDVI_LABELS) - 1)


def class_color(label):
    low, high = [float(value) for value in label.split("..")]
    if (low + high) / 2 < 0:
        return GRAY
    return CMAP_POS((low + high) / 2)[:3]


def scaled_ndvi(ndvi):
    adjusted = np.maximum(ndvi, 0.0)
    return np.clip(adjusted * POSITIVE_SCALE_FACTOR, 0.0, 1.0)


def create_false_color(ndvi):
    scaled = scaled_ndvi(ndvi)
    image = np.zeros((*ndvi.shape, 3), dtype=float)
    image[ndvi < 0] = GRAY
    positive = ndvi >= 0
    image[positive] = CMAP_POS(scaled[positive])[:, :3]
    return (image * 255).astype(np.uint8)


def create_ndvi_colorbar_image(height, width=180, avg_ndvi=None):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    cb_ax = fig.add_axes([0.35, 0.25, 0.3, 0.70])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb = colorbar.ColorbarBase(cb_ax, cmap=CMAP_POS, norm=norm, orientation="vertical")
    cb.set_label("NDVI", fontsize=12, labelpad=8)
    ticks = np.linspace(0.0, 1.0, 6)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.1f}" for t in ticks])
    cb.ax.tick_params(labelsize=16)

    gray_ax = fig.add_axes([0.35, 0.08, 0.3, 0.12])
    gray_ax.add_patch(matplotlib.patches.Rectangle((0, 0), 1, 1, color=GRAY))
    gray_ax.text(0.5, 0.5, "< 0", ha="center", va="center", fontsize=12)
    gray_ax.set_xticks([])
    gray_ax.set_yticks([])
    gray_ax.axis("off")

    if avg_ndvi is not None:
        fig.text(0.5, 0.02, f"Durchschn. NDVI: {avg_ndvi:.2f}", ha="center", va="bottom", fontsize=14, color="black")

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def save_diagram(ndvis_by_date, diagram_path):
    dates = [date for date, _ in ndvis_by_date]
    labels = [datetime.strptime(date, "%Y-%m-%d").strftime("%d.%m.") for date in dates]
    percentages_by_date = []
    average_ndvis = []
    for _, ndvi in ndvis_by_date:
        indices = class_index(ndvi).ravel()
        percentages = np.bincount(indices, minlength=len(NDVI_LABELS))
        percentages = percentages / percentages.sum() * 100
        percentages_by_date.append(percentages)
        average_ndvis.append(float(np.mean(scaled_ndvi(ndvi))))

    percentages_by_date = np.array(percentages_by_date)
    positions = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(max(10, len(labels) * 0.8), 7))
    bottom = np.zeros(len(labels))
    negative_percentage = percentages_by_date[:, [float(label.split("..")[-1]) <= 0 for label in NDVI_LABELS]].sum(axis=1)
    if np.any(negative_percentage > 0):
        ax1.bar(positions, negative_percentage, bottom=bottom, color=GRAY, edgecolor="black", linewidth=0.3, label="< 0")
        bottom += negative_percentage

    positive_handles = []
    positive_labels = []
    for index, label in enumerate(NDVI_LABELS):
        if float(label.split("..")[-2]) < 0:
            continue
        bar = ax1.bar(positions, percentages_by_date[:, index], bottom=bottom, color=class_color(label), edgecolor="black", linewidth=0.3, label=label)
        positive_handles.append(bar[0])
        positive_labels.append(label)
        bottom += percentages_by_date[:, index]

    ax1.set_xlabel("Tag", fontsize=12)
    ax1.set_ylabel("NDVI-Klassen [%]", fontsize=12)
    ax1.set_title("NDVI-Klassenverteilung pro Tag mit NDVI-Summe", fontsize=14)
    ax1.set_xticks(positions, labels)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="x", rotation=45, labelsize=10)
    ax1.tick_params(axis="y", labelsize=10)

    ax2 = ax1.twinx()
    ax2.plot(positions, average_ndvis, color="red", marker="o", linewidth=3, label="NDVI Durchschnitt", zorder=10)
    ax2.set_ylabel("NDVI Durchschnitt", color="red", fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor="red", labelsize=10)

    handles1 = positive_handles[::-1]
    labels1 = positive_labels[::-1]
    if np.any(negative_percentage > 0):
        handles1.append(matplotlib.patches.Patch(color=GRAY, label="< 0"))
        labels1.append("< 0")
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, title="Legende", fontsize=6, title_fontsize=8, labelspacing=0.15, handlelength=1.2, handleheight=0.7, bbox_to_anchor=(1.22, 0.5), loc="center left", borderaxespad=0.0, borderpad=0.3)

    fig.subplots_adjust(right=0.78, bottom=0.18)
    formula = r"$\mathrm{NDVI_{Durchschnitt}} = \mathrm{mean}(\max(NDVI, 0) \times 2)$"
    fig.text(0.86, 0.05, formula, ha="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.4", facecolor="whitesmoke", edgecolor="gray", alpha=0.9))
    fig.savefig(diagram_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def persist_average_ndvi(avg_ndvi):
    row = {"date": datetime.now().strftime("%Y-%m-%d"), "avg_ndvi": f"{avg_ndvi:.6f}"}
    exists = os.path.exists(AVG_LOG_PATH)
    with open(AVG_LOG_PATH, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["date", "avg_ndvi"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def send_telegram_warning(avg_ndvi):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Warnung: Kein Telegram-Bot-Token oder Chat-ID gesetzt. Telegram-Nachricht konnte nicht verschickt werden.")
        return

    text = (
        "NDVI-Warnung: Durchschnitt unter 0,1\n\n"
        f"Aktueller Durchschnitt: {avg_ndvi:.3f}\n"
        f"Zeitpunkt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    response = requests.post(
        url,
        json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True},
        timeout=20,
    )

    if response.ok:
        print("Warnung per Telegram-Bot gesendet.")
    else:
        print(f"Telegram-Fehler: {response.status_code} - {response.text}")


def main():
    ensure_output_folder()
    downloaded_paths = download_daily_images()

    if not downloaded_paths:
        raise RuntimeError("Keine Bilddateien für den ausgewählten Zeitraum wurden kopiert.")

    red_by_date = find_images_by_date(OUTPUT_FOLDER, ("RED", "ROT", "_R.", "_R_", "_R-"))
    nir_by_date = find_images_by_date(OUTPUT_FOLDER, ("NIR", "INFRA"))
    dates = sorted(
        date for date in set(red_by_date) | set(nir_by_date)
        if datetime.strptime(date, "%Y-%m-%d").date() >= START_DATE
    )

    if not dates:
        raise RuntimeError("Keine passenden RED-/NIR-Bilder im Zielordner gefunden.")

    ndvis_by_date = []
    for date in dates:
        red_candidates = red_by_date.get(date, [])
        nir_candidates = nir_by_date.get(date, [])
        if len(red_candidates) != 1 or len(nir_candidates) != 1:
            raise RuntimeError(
                f"Für {date} wurden nicht genau ein RED- und ein NIR-Bild gefunden: "
                f"RED={red_candidates}, NIR={nir_candidates}"
            )

        red_name = red_candidates[0]
        nir_name = nir_candidates[0]
        red = cv2.imread(os.path.join(OUTPUT_FOLDER, red_name), cv2.IMREAD_GRAYSCALE)
        nir = cv2.imread(os.path.join(OUTPUT_FOLDER, nir_name), cv2.IMREAD_GRAYSCALE)

        if red is None or nir is None:
            raise RuntimeError(f"Bilder für {date} konnten nicht geladen werden.")
        if red.shape != nir.shape:
            raise RuntimeError(f"Die Bildgrößen für {date} stimmen nicht überein: {red.shape} und {nir.shape}")

        nir_corrected = nir.astype(float) * NIR_CORRECTION_FACTOR
        red_corrected = red.astype(float) * RED_CORRECTION_FACTOR
        ndvi = (nir_corrected - red_corrected) / (nir_corrected + red_corrected + 1e-6)
        ndvi = np.clip(ndvi, -1, 1)
        ndvis_by_date.append((date, ndvi))

        output_path = os.path.join(OUTPUT_FOLDER, f"{date}_NDVI_ohne_negative_Bewertung.png")
        output_image = Image.fromarray(create_false_color(ndvi)).convert("RGB")
        avg_ndvi = float(np.mean(scaled_ndvi(ndvi)))
        legend_image = create_ndvi_colorbar_image(output_image.height, avg_ndvi=avg_ndvi)

        combined = Image.new("RGB", (output_image.width + legend_image.width, output_image.height), "white")
        combined.paste(output_image, (0, 0))
        combined.paste(legend_image, (output_image.width, 0))
        combined.save(output_path)
        print(f"{date}: NDVI-Bild gespeichert: {output_path}")
        print(f"  Durchschnittlicher NDVI (positive Werte x{POSITIVE_SCALE_FACTOR}): {avg_ndvi:.3f}")

        persist_average_ndvi(avg_ndvi)
        if avg_ndvi < 0.1:
            send_telegram_warning(avg_ndvi)

    diagram_path = os.path.join(OUTPUT_FOLDER, "NDVI_Diagramm_ohne_negative_Bewertung.png")
    save_diagram(ndvis_by_date, diagram_path)
    print(f"Sammeldiagramm gespeichert: {diagram_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fehler: {exc}")
        raise
