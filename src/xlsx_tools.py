"""Petits outils XLSX sans dependance externe.

Le projet n'a pas besoin d'un gros moteur Excel. On veut seulement ecrire des
tables de resultats lisibles et, parfois, les relire pour construire un classeur
de synthese. Ce module cree donc des fichiers `.xlsx` simples avec une feuille
ou plusieurs feuilles.
"""

from __future__ import annotations

import datetime as dt
import math
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS_DOC_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def table_depuis_dicts(lignes: list[dict[str, object]], colonnes: list[str] | None = None) -> list[list[object]]:
    """Transforme une liste de dictionnaires en table XLSX.

    Entrees:
        lignes: lignes de resultats.
        colonnes: ordre optionnel des colonnes.

    Sortie:
        Table avec la premiere ligne comme entete.
    """

    if colonnes is None:
        colonnes = []
        for ligne in lignes:
            for nom_colonne in ligne:
                if nom_colonne not in colonnes:
                    colonnes.append(nom_colonne)
    return [colonnes, *[[ligne.get(colonne, "") for colonne in colonnes] for ligne in lignes]]


def ecrire_dicts_xlsx(
    chemin: Path,
    lignes: list[dict[str, object]],
    colonnes: list[str] | None = None,
    nom_feuille: str = "Resultats",
) -> None:
    """Ecrit une table de dictionnaires dans un fichier XLSX.

    Entrees:
        chemin: fichier `.xlsx` a creer.
        lignes: donnees a ecrire.
        colonnes: ordre optionnel des colonnes.
        nom_feuille: nom de l'onglet Excel.

    Sortie:
        Aucune sortie Python. Le fichier est cree sur disque.
    """

    ecrire_xlsx({nom_feuille: table_depuis_dicts(lignes, colonnes)}, chemin)


def ecrire_xlsx(feuilles: dict[str, list[list[object]]], chemin: Path) -> None:
    """Cree un classeur XLSX avec une ou plusieurs feuilles.

    Entrees:
        feuilles: dictionnaire `nom_feuille -> table`.
        chemin: fichier `.xlsx` de sortie.

    Sortie:
        Aucune sortie Python. Le classeur est ecrit sur disque.
    """

    chemin.parent.mkdir(parents=True, exist_ok=True)
    noms_feuilles = list(feuilles)
    with zipfile.ZipFile(chemin, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml(len(noms_feuilles)))
        archive.writestr("_rels/.rels", _root_rels_xml())
        archive.writestr("xl/workbook.xml", _workbook_xml(noms_feuilles))
        archive.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml(len(noms_feuilles)))
        archive.writestr("xl/styles.xml", _styles_xml())
        archive.writestr("docProps/core.xml", _core_xml())
        archive.writestr("docProps/app.xml", _app_xml(noms_feuilles))
        for index, lignes in enumerate(feuilles.values(), start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _sheet_xml(lignes))


def lire_dicts_xlsx(chemin: Path, nom_feuille: str | None = None) -> list[dict[str, str]]:
    """Lit la premiere table d'un fichier XLSX simple.

    Entrees:
        chemin: fichier `.xlsx`.
        nom_feuille: feuille a lire. Si absent, on lit la premiere feuille.

    Sortie:
        Liste de dictionnaires, comme `csv.DictReader`.
    """

    lignes = lire_table_xlsx(chemin, nom_feuille)
    if not lignes:
        return []

    entetes = [str(valeur) for valeur in lignes[0]]
    sorties = []
    for ligne in lignes[1:]:
        sortie = {}
        for index, entete in enumerate(entetes):
            sortie[entete] = str(ligne[index]) if index < len(ligne) and ligne[index] is not None else ""
        sorties.append(sortie)
    return sorties


def lire_table_xlsx(chemin: Path, nom_feuille: str | None = None) -> list[list[object]]:
    """Lit une feuille XLSX creee par ce projet.

    Entrees:
        chemin: fichier `.xlsx`.
        nom_feuille: feuille a lire. Si absent, on lit la premiere.

    Sortie:
        Table de valeurs.
    """

    if not chemin.exists():
        return []

    with zipfile.ZipFile(chemin, "r") as archive:
        chemins_feuilles = _chemins_feuilles(archive)
        if not chemins_feuilles:
            return []
        if nom_feuille is None:
            chemin_feuille = next(iter(chemins_feuilles.values()))
        else:
            chemin_feuille = chemins_feuilles.get(nom_feuille)
            if chemin_feuille is None:
                return []
        shared_strings = _shared_strings(archive)
        xml = archive.read(chemin_feuille)
    return _parse_sheet_xml(xml, shared_strings)


def _column_name(index: int) -> str:
    nom = ""
    while index:
        index, reste = divmod(index - 1, 26)
        nom = chr(65 + reste) + nom
    return nom


def _column_index(cell_ref: str) -> int:
    lettres = re.match(r"([A-Z]+)", cell_ref)
    if not lettres:
        return 1
    index = 0
    for caractere in lettres.group(1):
        index = index * 26 + (ord(caractere) - 64)
    return index


def _xml_text(valeur: object) -> str:
    return escape(str(valeur), {'"': "&quot;"})


def _cell_xml(row_idx: int, col_idx: int, valeur: object, style: int) -> str:
    ref = f"{_column_name(col_idx)}{row_idx}"
    style_attr = f' s="{style}"' if style else ""
    if valeur is None or valeur == "":
        return f'<c r="{ref}"{style_attr}/>'
    if isinstance(valeur, bool):
        return f'<c r="{ref}" t="b"{style_attr}><v>{1 if valeur else 0}</v></c>'
    if isinstance(valeur, (int, float)) and not isinstance(valeur, bool):
        nombre = float(valeur)
        if math.isnan(nombre) or math.isinf(nombre):
            return f'<c r="{ref}"{style_attr}/>'
        return f'<c r="{ref}"{style_attr}><v>{valeur}</v></c>'
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{_xml_text(valeur)}</t></is></c>'


def _column_widths(lignes: list[list[object]]) -> list[float]:
    if not lignes:
        return [12.0]
    max_cols = max(len(ligne) for ligne in lignes)
    largeurs = []
    for col_idx in range(max_cols):
        longueur = 0
        for ligne in lignes[:500]:
            valeur = ligne[col_idx] if col_idx < len(ligne) else ""
            longueur = max(longueur, len(str(valeur)))
        largeurs.append(min(max(longueur + 2, 10), 55))
    return largeurs


def _sheet_xml(lignes: list[list[object]]) -> str:
    if not lignes:
        lignes = [[""]]
    max_cols = max((len(ligne) for ligne in lignes), default=1)
    max_rows = max(len(lignes), 1)
    dimension = f"A1:{_column_name(max_cols)}{max_rows}"
    cols = "".join(
        f'<col min="{idx}" max="{idx}" width="{largeur:.1f}" customWidth="1"/>'
        for idx, largeur in enumerate(_column_widths(lignes), start=1)
    )
    row_items = []
    for row_idx, ligne in enumerate(lignes, start=1):
        cellules = "".join(
            _cell_xml(row_idx, col_idx, valeur, 1 if row_idx == 1 else 0)
            for col_idx, valeur in enumerate(ligne, start=1)
        )
        row_items.append(f'<row r="{row_idx}">{cellules}</row>')
    auto_filter = f'<autoFilter ref="{dimension}"/>' if max_rows > 1 and max_cols > 1 else ""
    freeze = (
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/><selection pane="bottomLeft" '
        'activeCell="A2" sqref="A2"/></sheetView></sheetViews>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<worksheet xmlns="{NS_MAIN}" xmlns:r="{NS_DOC_REL}">'
        f'<dimension ref="{dimension}"/>'
        f"{freeze}"
        f"<cols>{cols}</cols>"
        f"<sheetData>{''.join(row_items)}</sheetData>"
        f"{auto_filter}"
        "</worksheet>"
    )


def _workbook_xml(noms_feuilles: list[str]) -> str:
    feuilles = "".join(
        f'<sheet name="{_xml_text(nom)}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, nom in enumerate(noms_feuilles, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<workbook xmlns="{NS_MAIN}" xmlns:r="{NS_DOC_REL}">'
        "<bookViews><workbookView/></bookViews>"
        f"<sheets>{feuilles}</sheets>"
        "</workbook>"
    )


def _workbook_rels_xml(sheet_count: int) -> str:
    relations = []
    for idx in range(1, sheet_count + 1):
        relations.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    relations.append(
        f'<Relationship Id="rId{sheet_count + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{NS_REL}">'
        f"{''.join(relations)}</Relationships>"
    )


def _root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{NS_REL}">'
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


def _content_types_xml(sheet_count: int) -> str:
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


def _styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<styleSheet xmlns="{NS_MAIN}">'
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


def _core_xml() -> str:
    timestamp = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:title>FedCompress results</dc:title>"
        "<dc:creator>FedCompress</dc:creator>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>'
        "</cp:coreProperties>"
    )


def _app_xml(noms_feuilles: list[str]) -> str:
    titres = "".join(f"<vt:lpstr>{_xml_text(nom)}</vt:lpstr>" for nom in noms_feuilles)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>Microsoft Excel</Application>"
        "<DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop>"
        '<HeadingPairs><vt:vector size="2" baseType="variant">'
        '<vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant>'
        f'<vt:variant><vt:i4>{len(noms_feuilles)}</vt:i4></vt:variant>'
        "</vt:vector></HeadingPairs>"
        f'<TitlesOfParts><vt:vector size="{len(noms_feuilles)}" baseType="lpstr">{titres}</vt:vector></TitlesOfParts>'
        "</Properties>"
    )


def _chemins_feuilles(archive: zipfile.ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relations = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in rels.findall(f"{{{NS_REL}}}Relationship")
    }

    chemins = {}
    for feuille in workbook.findall(f".//{{{NS_MAIN}}}sheet"):
        nom = feuille.attrib["name"]
        rid = feuille.attrib[f"{{{NS_DOC_REL}}}id"]
        cible = relations[rid]
        chemins[nom] = f"xl/{cible}" if not cible.startswith("xl/") else cible
    return chemins


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    racine = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    valeurs = []
    for item in racine.findall(f"{{{NS_MAIN}}}si"):
        textes = [node.text or "" for node in item.findall(f".//{{{NS_MAIN}}}t")]
        valeurs.append("".join(textes))
    return valeurs


def _parse_sheet_xml(xml: bytes, shared_strings: list[str]) -> list[list[object]]:
    racine = ET.fromstring(xml)
    lignes = []
    for row in racine.findall(f".//{{{NS_MAIN}}}row"):
        valeurs: list[object] = []
        for cell in row.findall(f"{{{NS_MAIN}}}c"):
            ref = cell.attrib.get("r", "A1")
            col_idx = _column_index(ref)
            while len(valeurs) < col_idx - 1:
                valeurs.append("")
            valeurs.append(_cell_value(cell, shared_strings))
        lignes.append(valeurs)
    return lignes


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> object:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        textes = [node.text or "" for node in cell.findall(f".//{{{NS_MAIN}}}t")]
        return "".join(textes)

    value_node = cell.find(f"{{{NS_MAIN}}}v")
    if value_node is None or value_node.text is None:
        return ""
    valeur = value_node.text

    if cell_type == "s":
        index = int(valeur)
        return shared_strings[index] if 0 <= index < len(shared_strings) else ""
    if cell_type == "b":
        return "TRUE" if valeur == "1" else "FALSE"
    return valeur
