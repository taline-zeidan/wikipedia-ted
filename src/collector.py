import os
import re
import wptools
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

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


def _sanitize_tag(key: str) -> str:
    tag = key.strip().lower()
    tag = re.sub(r"[^\w]", "_", tag)
    tag = re.sub(r"_+", "_", tag)
    tag = tag.strip("_")
    if tag and tag[0].isdigit():
        tag = "field_" + tag
    return tag or "field"


def _sanitize_value(value: str) -> str:
    value = re.sub(r"\[\[.*?\]\]", lambda m: m.group(0).split("|")[-1].strip("[]"), value)
    value = re.sub(r"{{.*?}}", "", value)
    value = re.sub(r"<.*?>", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _infobox_to_xml(country_name: str, infobox: dict) -> ElementTree:
    root = Element("country")
    name_el = SubElement(root, "name")
    name_el.text = country_name

    for key, value in infobox.items():
        tag = _sanitize_tag(key)
        if not tag:
            continue

        raw_value = str(value) if not isinstance(value, str) else value
        clean_value = _sanitize_value(raw_value)

        if not clean_value:
            continue

        child = SubElement(root, tag)
        child.text = clean_value

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

    page = wptools.page(country_name, silent=True)
    page.get_parse()

    infobox = page.data.get("infobox")
    if not infobox:
        raise ValueError(f"No infobox found for: {country_name}")

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