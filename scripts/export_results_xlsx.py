"""Exporte les resultats XLSX de FedCompress vers un classeur XLSX compact.

"""

from __future__ import annotations

import csv
import datetime as dt
import math
import sys
import zipfile
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.xlsx_tools import lire_dicts_xlsx

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "fedcompress_results_summary.xlsx"


SUMMARY_COLUMNS = [
    "family",
    "model_key",
    "model_id",
    "experiment",
    "partition",
    "scenario",
    "rounds",
    "final_accuracy",
    "accuracy_pct",
    "final_loss",
    "accuracy_loss_pt",
    "communication_total_mb",
    "communication_reduction_pct",
    "communication_round_mb",
    "uplink_round_mb",
    "downlink_round_mb",
    "total_seconds",
    "train_examples",
    "test_examples",
    "communicated_parameters",
    "source_file",
    "notes",
]


SMALL_CNN_FILES = [
    ("fedavg_float16_comparison_summary.xlsx", "iid_float16", "iid"),
    ("fedavg_sparsification_summary.xlsx", "iid_sparsification", "iid"),
    ("fedavg_non_iid_summary.xlsx", "non_iid", "non_iid"),
    ("fedavg_non_iid_20rounds_summary.xlsx", "non_iid_20rounds", "non_iid"),
]


HF_EXPERIMENTS = [
    ("iid_baseline", "iid", "none", 3),
    ("iid_float16", "iid", "none,float16", 3),
    ("iid_sparsification", "iid", "none,topk_50,topk_20,topk_10", 3),
    ("non_iid", "non_iid", "none,float16,topk_20,topk_10", 3),
    ("non_iid_20rounds", "non_iid", "none,float16,topk_20,topk_10", 20),
]


SMALL_CNN_COMMANDS = [
    ("iid_baseline", "iid", "none", 20),
    ("iid_float16", "iid", "none,float16", 20),
    ("iid_sparsification", "iid", "none,topk_50,topk_20,topk_10", 20),
    ("non_iid", "non_iid", "none,float16,topk_20,topk_10", 20),
    ("non_iid_20rounds", "non_iid", "none,float16,topk_20,topk_10", 20),
]


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_result_dicts(path: Path) -> list[dict[str, str]]:
    """Lit un resultat XLSX, avec fallback CSV pour les anciens fichiers."""

    if path.exists() and path.suffix.lower() == ".xlsx":
        return lire_dicts_xlsx(path)
    if path.exists() and path.suffix.lower() == ".csv":
        return read_csv_dicts(path)

    ancien_csv = path.with_suffix(".csv")
    if ancien_csv.exists():
        return read_csv_dicts(ancien_csv)
    return []


def to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def to_int(value: object) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    return int(round(number))


def round_count(row: dict[str, object], fallback: int | None = None) -> int | None:
    total = to_float(row.get("communication_total_mb"))
    per_round = to_float(row.get("communication_round_mb"))
    if total is not None and per_round not in (None, 0):
        return int(round(total / per_round))
    return fallback


def normalize_row(
    row: dict[str, str],
    *,
    family: str,
    model_key: str,
    model_id: str,
    experiment: str,
    partition: str,
    source_file: str,
    notes: str,
) -> dict[str, object]:
    accuracy = to_float(row.get("final_accuracy"))
    loss = to_float(row.get("final_loss"))
    out: dict[str, object] = {
        "family": family,
        "model_key": model_key,
        "model_id": model_id,
        "experiment": experiment,
        "partition": partition,
        "scenario": row.get("scenario", ""),
        "rounds": round_count(row),
        "final_accuracy": accuracy,
        "accuracy_pct": accuracy * 100 if accuracy is not None else None,
        "final_loss": loss,
        "accuracy_loss_pt": None,
        "communication_total_mb": to_float(row.get("communication_total_mb")),
        "communication_reduction_pct": None,
        "communication_round_mb": to_float(row.get("communication_round_mb")),
        "uplink_round_mb": to_float(row.get("uplink_round_mb")),
        "downlink_round_mb": to_float(row.get("downlink_round_mb")),
        "total_seconds": to_float(row.get("total_seconds")),
        "train_examples": to_int(row.get("train_examples")),
        "test_examples": to_int(row.get("test_examples")),
        "communicated_parameters": to_int(row.get("communicated_parameters")),
        "source_file": source_file,
        "notes": notes,
    }
    out["rounds"] = round_count(out)
    return out


def load_small_cnn_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    baseline_path = RESULTS_DIR / "fedavg_simple_baseline.xlsx"
    baseline_rows = read_result_dicts(baseline_path)
    if baseline_rows:
        last = baseline_rows[-1]
        total_seconds = sum(
            value for value in (to_float(row.get("round_seconds")) for row in baseline_rows) if value is not None
        )
        row = {
            "scenario": "none",
            "final_accuracy": last.get("test_accuracy"),
            "final_loss": last.get("test_loss"),
            "total_seconds": total_seconds,
            "communication_total_mb": last.get("communication_total_mb"),
            "communication_round_mb": last.get("communication_round_mb"),
            "uplink_round_mb": "",
            "downlink_round_mb": "",
        }
        normalized = normalize_row(
            row,
            family="small_cnn",
            model_key="small_cnn",
            model_id="Small CNN",
            experiment="iid_baseline",
            partition="iid",
            source_file=baseline_path.name,
            notes="Full small CNN trained and communicated.",
        )
        rows.append(normalized)

    for filename, experiment, partition in SMALL_CNN_FILES:
        path = RESULTS_DIR / filename
        for source_row in read_result_dicts(path):
            rows.append(
                normalize_row(
                    source_row,
                    family="small_cnn",
                    model_key="small_cnn",
                    model_id="Small CNN",
                    experiment=experiment,
                    partition=partition,
                    source_file=filename,
                    notes="Full small CNN trained and communicated.",
                )
            )

    return rows


def load_hf_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    summary_paths = sorted(RESULTS_DIR.glob("*/hf_*_summary.xlsx"))
    if not summary_paths:
        summary_paths = [RESULTS_DIR / "hf_models_all_summary.xlsx"]

    for path in summary_paths:
        for source_row in read_result_dicts(path):
            rows.append(
                normalize_row(
                    source_row,
                    family="hugging_face",
                    model_key=source_row.get("model_key", ""),
                    model_id=source_row.get("model_id", ""),
                    experiment=source_row.get("experiment", ""),
                    partition=source_row.get("partition", ""),
                    source_file=path.name,
                    notes="Backbone frozen; classification head trained and communicated.",
                )
            )
    return rows


def add_relative_metrics(rows: list[dict[str, object]]) -> None:
    baselines: dict[tuple[object, object], dict[str, object]] = {}
    for row in rows:
        if row.get("scenario") == "none":
            baselines[(row.get("model_key"), row.get("experiment"))] = row

    for row in rows:
        baseline = baselines.get((row.get("model_key"), row.get("experiment")))
        if not baseline:
            continue
        base_acc = to_float(baseline.get("final_accuracy"))
        acc = to_float(row.get("final_accuracy"))
        base_comm = to_float(baseline.get("communication_total_mb"))
        comm = to_float(row.get("communication_total_mb"))
        if base_acc is not None and acc is not None:
            row["accuracy_loss_pt"] = (base_acc - acc) * 100
        if base_comm not in (None, 0) and comm is not None:
            row["communication_reduction_pct"] = (1 - comm / base_comm) * 100


def make_summary_rows(rows: Iterable[dict[str, object]]) -> list[list[object]]:
    table = [SUMMARY_COLUMNS]
    for row in rows:
        table.append([row.get(column) for column in SUMMARY_COLUMNS])
    return table


def make_readme_rows(total_rows: int) -> list[list[object]]:
    return [
        ["Item", "Value"],
        ["Classeur", "Resume des resultats FedCompress"],
        ["Genere le", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Racine projet", str(PROJECT_ROOT)],
        ["Lignes exportees", total_rows],
        ["Objectif", "Regrouper les resultats XLSX et les commandes de reproduction dans un XLSX."],
        ["Note git", "Le dossier results et les fichiers XLSX sont ignores par git par defaut."],
    ]


def make_command_rows() -> list[list[object]]:
    rows: list[list[object]] = [["Group", "Model", "Experiment", "Command", "Comment"]]
    py = r".\.venv\Scripts\python.exe"
    rows.append(
        [
            "export",
            "all",
            "xlsx",
            rf"{py} scripts\export_results_xlsx.py",
            "Regenere ce classeur a partir des XLSX existants.",
        ]
    )
    rows.append(
        [
            "dataset",
            "fashion_mnist",
            "inspect",
            rf"{py} scripts\inspect_dataset.py --save-grid",
            "Inspecte Fashion-MNIST et sauvegarde une grille d'exemples.",
        ]
    )
    rows.append(
        [
            "benchmark",
            "hugging_face",
            "candidate_models",
            rf"{py} scripts\benchmark_hf_models.py --models resnet18,convnext_tiny",
            "Relance le benchmark des modeles Hugging Face.",
        ]
    )
    rows.append(
        [
            "small_cnn_suite",
            "small_cnn",
            "all",
            rf"{py} scripts\run_cnn_suite.py",
            "Lance toute la suite du petit CNN.",
        ]
    )
    rows.append(
        [
            "hf_suite",
            "both",
            "all",
            rf"{py} scripts\run_hf_suite.py --model both",
            "Lance toute la suite Hugging Face pour ResNet-18 et ConvNeXT-Tiny.",
        ]
    )
    rows.append(
        [
            "hf_suite",
            "both",
            "all_without_long",
            rf"{py} scripts\run_hf_suite.py --model both --skip-long",
            "Lance les experiences standard sauf le non-IID long de 20 rounds.",
        ]
    )
    rows.append(
        [
            "hf_suite",
            "both",
            "dry_run",
            rf"{py} scripts\run_hf_suite.py --model both --dry-run",
            "Affiche le plan sans lancer l'entrainement.",
        ]
    )

    for model in ("resnet18", "convnext_tiny"):
        for experiment, partition, scenarios, rounds in HF_EXPERIMENTS:
            command = (
                rf"{py} scripts\run_hf_experiment.py --model {model} "
                rf"--experiment {experiment} --partition {partition} "
                rf"--scenarios {scenarios} --rounds {rounds}"
            )
            rows.append(
                [
                    "hf_single",
                    model,
                    experiment,
                    command,
                    "Backbone gele par defaut; seuls les parametres entrainables sont communiques.",
                ]
            )

    for experiment, partition, scenarios, rounds in SMALL_CNN_COMMANDS:
        rows.append(
            [
                "small_cnn",
                "small_cnn",
                experiment,
                (
                    rf"{py} scripts\run_cnn_experiment.py --experiment {experiment} "
                    rf"--partition {partition} --scenarios {scenarios} --rounds {rounds}"
                ),
                "Relance l'equivalent Python propre de l'ancien notebook.",
            ]
        )

    return rows


def make_best_rows(rows: list[dict[str, object]]) -> list[list[object]]:
    table: list[list[object]] = [
        [
            "Question",
            "Best current answer",
            "Evidence",
        ]
    ]

    def row_for(model_key: str, experiment: str, scenario: str) -> dict[str, object] | None:
        for row in rows:
            if (
                row.get("model_key") == model_key
                and row.get("experiment") == experiment
                and row.get("scenario") == scenario
            ):
                return row
        return None

    conv = row_for("convnext_tiny", "non_iid_20rounds", "float16")
    if conv:
        table.append(
            [
                "Best Hugging Face compromise",
                "ConvNeXT-Tiny with float16",
                (
                    f"{to_float(conv.get('accuracy_pct')):.2f}% accuracy, "
                    f"{to_float(conv.get('communication_reduction_pct')):.2f}% less communication."
                ),
            ]
        )

    small = row_for("small_cnn", "iid_float16", "float16")
    if small:
        table.append(
            [
                "Best simple compression",
                "Small CNN with float16",
                (
                    f"{to_float(small.get('accuracy_loss_pt')):.2f} accuracy point loss, "
                    f"{to_float(small.get('communication_reduction_pct')):.2f}% less communication."
                ),
            ]
        )

    table.append(
        [
            "Main scientific warning",
            "Repeat runs with several seeds",
            "The current workbook summarizes one seed for most experiments.",
        ]
    )
    return table


def column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def xml_text(value: object) -> str:
    return escape(str(value), {'"': "&quot;"})


def cell_xml(row_idx: int, col_idx: int, value: object, style: int) -> str:
    ref = f"{column_name(col_idx)}{row_idx}"
    style_attr = f' s="{style}"' if style else ""
    if value is None or value == "":
        return f'<c r="{ref}"{style_attr}/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"{style_attr}><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return f'<c r="{ref}"{style_attr}/>'
        return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{xml_text(value)}</t></is></c>'


def column_widths(rows: list[list[object]]) -> list[float]:
    if not rows:
        return []
    max_cols = max(len(row) for row in rows)
    widths: list[float] = []
    for col_idx in range(max_cols):
        longest = 0
        for row in rows[:500]:
            value = row[col_idx] if col_idx < len(row) else ""
            longest = max(longest, len(str(value)))
        widths.append(min(max(longest + 2, 10), 55))
    return widths


def sheet_xml(rows: list[list[object]]) -> str:
    max_cols = max((len(row) for row in rows), default=1)
    max_rows = max(len(rows), 1)
    dimension = f"A1:{column_name(max_cols)}{max_rows}"
    cols = "".join(
        f'<col min="{idx}" max="{idx}" width="{width:.1f}" customWidth="1"/>'
        for idx, width in enumerate(column_widths(rows), start=1)
    )
    row_items = []
    for row_idx, row in enumerate(rows, start=1):
        cells = "".join(
            cell_xml(row_idx, col_idx, value, 1 if row_idx == 1 else 0)
            for col_idx, value in enumerate(row, start=1)
        )
        row_items.append(f'<row r="{row_idx}">{cells}</row>')
    auto_filter = f'<autoFilter ref="{dimension}"/>' if max_rows > 1 and max_cols > 1 else ""
    freeze = (
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/><selection pane="bottomLeft" '
        'activeCell="A2" sqref="A2"/></sheetView></sheetViews>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="{dimension}"/>'
        f"{freeze}"
        f"<cols>{cols}</cols>"
        f"<sheetData>{''.join(row_items)}</sheetData>"
        f"{auto_filter}"
        "</worksheet>"
    )


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = "".join(
        f'<sheet name="{xml_text(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<bookViews><workbookView/></bookViews>"
        f"<sheets>{sheets}</sheets>"
        "</workbook>"
    )


def workbook_rels_xml(sheet_count: int) -> str:
    rels = []
    for idx in range(1, sheet_count + 1):
        rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    rels.append(
        f'<Relationship Id="rId{sheet_count + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{''.join(rels)}</Relationships>"
    )


def root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
        'Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
        'Target="docProps/app.xml"/>'
        "</Relationships>"
    )


def content_types_xml(sheet_count: int) -> str:
    overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        '<Override PartName="/docProps/core.xml" '
        'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for idx in range(1, sheet_count + 1):
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f"{''.join(overrides)}</Types>"
    )


def styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2">'
        '<font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font>'
        '<font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/><family val="2"/></font>'
        "</fonts>"
        '<fills count="3">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF17324D"/><bgColor indexed="64"/></patternFill></fill>'
        "</fills>"
        '<borders count="2">'
        "<border><left/><right/><top/><bottom/><diagonal/></border>"
        '<border><left style="thin"><color rgb="FFD9DEE8"/></left>'
        '<right style="thin"><color rgb="FFD9DEE8"/></right>'
        '<top style="thin"><color rgb="FFD9DEE8"/></top>'
        '<bottom style="thin"><color rgb="FFD9DEE8"/></bottom><diagonal/></border>'
        "</borders>"
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" '
        'applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1">'
        '<alignment horizontal="center" vertical="center" wrapText="1"/></xf>'
        "</cellXfs>"
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '<dxfs count="0"/>'
        '<tableStyles count="0" defaultTableStyle="TableStyleMedium2" defaultPivotStyle="PivotStyleLight16"/>'
        "</styleSheet>"
    )


def core_xml() -> str:
    timestamp = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:title>FedCompress results summary</dc:title>"
        "<dc:creator>FedCompress export script</dc:creator>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>'
        "</cp:coreProperties>"
    )


def app_xml(sheet_names: list[str]) -> str:
    titles = "".join(f"<vt:lpstr>{xml_text(name)}</vt:lpstr>" for name in sheet_names)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>Microsoft Excel</Application>"
        "<DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop>"
        '<HeadingPairs><vt:vector size="2" baseType="variant">'
        '<vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant>'
        f'<vt:variant><vt:i4>{len(sheet_names)}</vt:i4></vt:variant>'
        "</vt:vector></HeadingPairs>"
        f'<TitlesOfParts><vt:vector size="{len(sheet_names)}" baseType="lpstr">{titles}</vt:vector></TitlesOfParts>'
        "</Properties>"
    )


def write_xlsx(sheets: dict[str, list[list[object]]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = list(sheets)
    with zipfile.ZipFile(output_file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml(len(sheet_names)))
        archive.writestr("_rels/.rels", root_rels_xml())
        archive.writestr("xl/workbook.xml", workbook_xml(sheet_names))
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(len(sheet_names)))
        archive.writestr("xl/styles.xml", styles_xml())
        archive.writestr("docProps/core.xml", core_xml())
        archive.writestr("docProps/app.xml", app_xml(sheet_names))
        for idx, rows in enumerate(sheets.values(), start=1):
            archive.writestr(f"xl/worksheets/sheet{idx}.xml", sheet_xml(rows))


def main() -> None:
    small_rows = load_small_cnn_rows()
    hf_rows = load_hf_rows()
    all_rows = small_rows + hf_rows
    add_relative_metrics(all_rows)

    sheets = {
        "README": make_readme_rows(len(all_rows)),
        "Best_Results": make_best_rows(all_rows),
        "Summary": make_summary_rows(all_rows),
        "Small_CNN": make_summary_rows(small_rows),
        "HuggingFace": make_summary_rows(hf_rows),
        "Commands": make_command_rows(),
    }
    write_xlsx(sheets, OUTPUT_FILE)
    print(f"Exported {len(all_rows)} result rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
