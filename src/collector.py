import os
import re
import requests
import mwparserfromhell
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

MEDIAWIKI_API = "https://en.wikipedia.org/w/api.php"

DECORATIVE_FIELDS = {
    "image_flag", "image_flag2", "image_coat", "image_map", "image_map2",
    "image_map_caption", "image_map2_caption", "image_map_alt", "image_map2_alt",
    "alt_flag", "alt_flag2", "alt_coat", "flag_border", "flag_caption",
    "coat_alt", "coat_caption", "symbol_type", "national_anthem",
    "map_caption", "map_caption2", "image_map_size", "image_map2_size",
    "coa_size", "flag_width", "footnote_a", "footnote_b", "footnote_c",
    "footnote_d", "footnote_e", "footnote_f", "footnotes",
}

UN_MEMBER_STATES = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda",
    "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain",
    "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan",
    "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria",
    "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada",
    "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros",
    "Congo", "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czech Republic",
    "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt",
    "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia",
    "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana",
    "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti",
    "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland",
    "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati",
    "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya",
    "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia",
    "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico",
    "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique",
    "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua",
    "Niger", "Nigeria", "North Korea", "North Macedonia", "Norway", "Oman", "Pakistan",
    "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland",
    "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis",
    "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino",
    "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles",
    "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia",
    "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan",
    "Suriname", "Sweden", "Switzerland", "Syria", "Tajikistan", "Tanzania", "Thailand",
    "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey",
    "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates",
    "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela",
    "Vietnam", "Yemen", "Zambia", "Zimbabwe",
]

def _fetch_wikitext(country_name: str) -> str:
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
        "titles": country_name,
        "redirects": 1,
    }
    headers = {"User-Agent": "WikipediaTEDProject/1.0 (COE543 LAU; Academic Research) python-requests"}
    response = requests.get(MEDIAWIKI_API, params=params, headers=headers, timeout=15)
    response.raise_for_status()
    data = response.json()

    pages = data["query"]["pages"]
    page = next(iter(pages.values()))

    if "missing" in page:
        raise ValueError(f"Wikipedia page not found: {country_name}")

    return page["revisions"][0]["slots"]["main"]["*"]


def _extract_infobox(wikitext: str) -> dict[str, str]:
    parsed = mwparserfromhell.parse(wikitext)
    templates = parsed.filter_templates()

    infobox = None
    for template in templates:
        name = template.name.strip().lower()
        if "infobox country" in name or "infobox former country" in name:
            infobox = template
            break

    if infobox is None:
        raise ValueError("No infobox country template found.")

    result: dict[str, str] = {}
    for param in infobox.params:
        key = param.name.strip()
        value = param.value

        tag = _sanitize_tag(key)
        if not tag or tag in DECORATIVE_FIELDS:
            continue

        clean = _sanitize_value(value)
        if clean:
            result[tag] = clean

    return result


def _sanitize_tag(key: str) -> str:
    tag = key.strip().lower()
    tag = re.sub(r"[^\w]", "_", tag)
    tag = re.sub(r"_+", "_", tag)
    tag = tag.strip("_")
    if tag and tag[0].isdigit():
        tag = "field_" + tag
    return tag or ""


def _sanitize_value(value) -> str:
    stripped = value.strip_code(
        normalize=True,
        collapse=True,
        keep_template_params=False,
    )
    stripped = re.sub(r"\[\d+\]", "", stripped)
    stripped = re.sub(r"<[^>]+>", "", stripped)
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped.strip()


def _infobox_to_xml(country_name: str, infobox: dict[str, str]) -> ElementTree:
    root = Element("country")
    name_el = SubElement(root, "name")
    name_el.text = country_name

    for key, value in infobox.items():
        if not value:
            continue
        child = SubElement(root, key)
        child.text = value

    tree = ElementTree(root)
    indent(tree, space="  ")
    return tree


def _output_path(country_name: str) -> str:
    filename = country_name.lower().replace(" ", "_") + ".xml"
    return os.path.join(DATA_DIR, filename)


def collect_country(country_name: str, overwrite: bool = False) -> str:
    output_path = _output_path(country_name)

    if not overwrite and os.path.exists(output_path):
        return output_path

    wikitext = _fetch_wikitext(country_name)
    infobox = _extract_infobox(wikitext)

    tree = _infobox_to_xml(country_name, infobox)

    os.makedirs(DATA_DIR, exist_ok=True)
    tree.write(output_path, encoding="unicode", xml_declaration=True)

    return output_path


def collect_all(overwrite: bool = False) -> dict[str, str]:
    results: dict[str, str] = {}

    for country in UN_MEMBER_STATES:
        try:
            path = collect_country(country, overwrite=overwrite)
            results[country] = path
            print(f"[OK]    {country} -> {path}")
        except Exception as e:
            results[country] = f"ERROR: {e}"
            print(f"[FAIL]  {country}: {e}")

    return results


if __name__ == "__main__":
    collect_all()