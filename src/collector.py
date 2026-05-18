"""
Wikipedia Infobox Collector (Project 1)
=======================================================
Scrapes Wikipedia country infoboxes and saves them as XML files.

Country name resolution:
  Wikipedia page titles sometimes differ from the canonical names we use
  (e.g. "Micronesia" → "Federated States of Micronesia"). The
  WIKIPEDIA_NAME_OVERRIDES dict maps our canonical name to the exact
  Wikipedia page title that holds the infobox.

  When running collect_all(), the UN member list is fetched live from
  Wikipedia's "Member states of the United Nations" page. A hardcoded
  fallback is used if that fetch fails.
"""

import argparse
import os
import re
import time
import requests
import mwparserfromhell
from bs4 import BeautifulSoup
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

MEDIAWIKI_API = "https://en.wikipedia.org/w/api.php"

UN_MEMBERS_PAGE = "https://en.wikipedia.org/wiki/Member_states_of_the_United_Nations"

SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

API_HEADERS = {
    "User-Agent": "WikipediaTEDProject/1.0 (COE543 LAU; Academic Research) python-requests"
}

# Fields that carry no structural or semantic value for comparison
DECORATIVE_FIELDS = {
    "image_flag", "image_flag2", "image_coat", "image_map", "image_map2",
    "image_map_caption", "image_map2_caption", "image_map_alt", "image_map2_alt",
    "alt_flag", "alt_flag2", "alt_coat", "flag_border", "flag_caption",
    "coat_alt", "coat_caption", "symbol_type", "national_anthem",
    "map_caption", "map_caption2", "image_map_size", "image_map2_size",
    "coa_size", "flag_width", "footnote_a", "footnote_b", "footnote_c",
    "footnote_d", "footnote_e", "footnote_f", "footnotes",
}

INFOBOX_PATTERNS = [
    "infobox country",
    "infobox former country",
    "infobox sovereign state",
    "infobox nation",
    # Denmark and the Netherlands use this on their main country articles.
    "infobox political division",
]

# ---------------------------------------------------------------------------
# Rate limiting
#
# Wikipedia's etiquette for unauthenticated API access is ~1 request/second.
# We enforce this proactively (between every request) rather than only
# reacting to HTTP 429 after we've already been throttled.
# ---------------------------------------------------------------------------

MIN_REQUEST_INTERVAL = 1.0  # seconds between successive Wikipedia requests
_last_request_time: float = 0.0


def _throttle() -> None:
    """Block until at least MIN_REQUEST_INTERVAL has passed since the last call."""
    global _last_request_time
    now = time.monotonic()
    elapsed = now - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.monotonic()

# ---------------------------------------------------------------------------
# Wikipedia name overrides
#
# Maps our canonical country name → the exact Wikipedia page title.
# Only needed when the Wikipedia title differs from what we use.
# ---------------------------------------------------------------------------

WIKIPEDIA_NAME_OVERRIDES: dict[str, str] = {
    # Disambiguation pages — UN list uses the short common name
    "Georgia":                  "Georgia (country)",
    "Ireland":                  "Republic of Ireland",
    # Wikipedia uses the full official name
    "Micronesia":               "Federated States of Micronesia",
    "Sao Tome and Principe":    "São Tomé and Príncipe",
    "São Tomé and Príncipe":    "São Tomé and Príncipe",
    # Wikipedia uses "Democratic Republic of the Congo" consistently
    "Congo, DR":                "Democratic Republic of the Congo",
    # Wikipedia uses "Republic of the Congo" for Brazzaville
    "Congo":                    "Republic of the Congo",
    # Palestine is listed as "State of Palestine" on Wikipedia
    "Palestine":                "State of Palestine",
    # Laos full name on Wikipedia
    "Laos":                     "Laos",
    # Brunei full official name redirects fine; kept explicit for safety
    "Brunei":                   "Brunei",
    # Ivory Coast redirects fine but keeping explicit
    "Ivory Coast":              "Ivory Coast",
}

# ---------------------------------------------------------------------------
# Hardcoded fallback used if the live Wikipedia scrape fails
# ---------------------------------------------------------------------------

_FALLBACK_UN_MEMBER_STATES: list[str] = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola",
    "Antigua and Barbuda", "Argentina", "Armenia", "Australia", "Austria",
    "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus",
    "Belgium", "Belize", "Benin", "Bhutan", "Bolivia",
    "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria",
    "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon",
    "Canada", "Central African Republic", "Chad", "Chile", "China",
    "Colombia", "Comoros", "Republic of the Congo", "Costa Rica", "Croatia",
    "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica",
    "Dominican Republic", "Ecuador", "Egypt", "El Salvador",
    "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia",
    "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany",
    "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau",
    "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India",
    "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica",
    "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait",
    "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia",
    "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar",
    "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands",
    "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco",
    "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia",
    "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger",
    "Nigeria", "North Korea", "North Macedonia", "Norway", "Oman",
    "Pakistan", "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru",
    "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia",
    "Rwanda", "Saint Kitts and Nevis", "Saint Lucia",
    "Saint Vincent and the Grenadines", "Samoa", "San Marino",
    "São Tomé and Príncipe", "Saudi Arabia", "Senegal", "Serbia",
    "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia",
    "Solomon Islands", "Somalia", "South Africa", "South Korea",
    "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden",
    "Switzerland", "Syria", "Tajikistan", "Tanzania", "Thailand",
    "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia",
    "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine",
    "United Arab Emirates", "United Kingdom", "United States", "Uruguay",
    "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", "Yemen", "Zambia",
    "Zimbabwe",
    # UN members added after original collector list
    "Democratic Republic of the Congo",
    "Ivory Coast",
    "Palestine",
]


# ---------------------------------------------------------------------------
# Live UN member name scraping
# ---------------------------------------------------------------------------

def _fetch_un_wikipedia_names() -> list[str]:
    """
    Scrape the list of UN member state names directly from Wikipedia.

    Returns the names exactly as they appear on the Wikipedia table,
    which are the correct page titles for fetching infoboxes.

    Falls back to the hardcoded list if the page cannot be reached.
    """
    try:
        response = requests.get(UN_MEMBERS_PAGE, headers=SCRAPE_HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        target_table = None
        for table in soup.find_all("table"):
            if "Member state" in table.text:
                target_table = table
                break

        if target_table is None:
            raise ValueError("Could not locate the UN members table on Wikipedia.")

        names = []
        for tr in target_table.find_all("tr"):
            first_cell = tr.find(["th", "td"])
            if not first_cell or "Member state" in first_cell.text:
                continue
            # Strip footnotes, parentheticals, and alternate names after commas
            name = first_cell.text.split("[")[0].split(",")[0].split("(")[0].strip()
            if name:
                names.append(name)

        if not names:
            raise ValueError("Scraped name list is empty.")

        print(f"[collector] Fetched {len(names)} UN member names from Wikipedia.")
        return names

    except Exception as exc:
        print(
            f"[collector] WARNING: Could not fetch UN members from Wikipedia ({exc}). "
            "Using hardcoded fallback list."
        )
        return list(_FALLBACK_UN_MEMBER_STATES)


def _resolve_wikipedia_title(canonical_name: str) -> str:
    """
    Return the Wikipedia page title to use when fetching a country's wikitext.

    Checks WIKIPEDIA_NAME_OVERRIDES first; falls back to the canonical name.
    """
    return WIKIPEDIA_NAME_OVERRIDES.get(canonical_name, canonical_name)


# ---------------------------------------------------------------------------
# Build the authoritative country list at import time
# ---------------------------------------------------------------------------

def _build_un_member_states() -> list[str]:
    """
    Return the deduplicated list of UN member state canonical names.

    We prefer the live Wikipedia list. When a Wikipedia name has an entry
    in WIKIPEDIA_NAME_OVERRIDES (inverted), we replace it with our
    canonical form so the rest of the codebase stays consistent.
    """
    wiki_names = _fetch_un_wikipedia_names()

    # Build reverse map: wikipedia_title → canonical_name
    wiki_to_canonical = {v: k for k, v in WIKIPEDIA_NAME_OVERRIDES.items()}

    canonical = []
    seen = set()
    for name in wiki_names:
        canonical_name = wiki_to_canonical.get(name, name)
        if canonical_name not in seen:
            seen.add(canonical_name)
            canonical.append(canonical_name)

    return canonical


# Module-level list, built once when the module is imported
UN_MEMBER_STATES: list[str] = _build_un_member_states()


# ---------------------------------------------------------------------------
# Wikipedia wikitext fetching
# ---------------------------------------------------------------------------

def _fetch_wikitext(country_name: str) -> str:
    """
    Fetch the raw wikitext for a country page from the Wikipedia API.

    Uses WIKIPEDIA_NAME_OVERRIDES to resolve the correct page title when
    our canonical name differs from Wikipedia's.

    Enforces proactive rate limiting (1 req/s) and handles HTTP 429
    by honoring the Retry-After header before raising.
    """
    wiki_title = _resolve_wikipedia_title(country_name)
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
        "titles": wiki_title,
        "redirects": 1,
    }

    _throttle()
    response = requests.get(
        MEDIAWIKI_API, params=params, headers=API_HEADERS, timeout=15
    )

    # Handle explicit rate-limiting: respect Wikipedia's Retry-After header,
    # then signal failure so collect_all's retry loop kicks in.
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "5")
        try:
            wait = float(retry_after)
        except ValueError:
            wait = 5.0
        print(f"[collector] 429 for '{wiki_title}', server asked to wait {wait}s")
        time.sleep(wait)
        raise requests.HTTPError(
            f"429 Too Many Requests for '{wiki_title}' (waited {wait}s)"
        )

    response.raise_for_status()
    data = response.json()

    pages = data["query"]["pages"]
    page = next(iter(pages.values()))

    if "missing" in page:
        raise ValueError(f"Wikipedia page not found: '{wiki_title}' (canonical: '{country_name}')")

    return page["revisions"][0]["slots"]["main"]["*"]


# ---------------------------------------------------------------------------
# Infobox extraction
# ---------------------------------------------------------------------------

def _normalize_template_name(name: str) -> str:
    return re.sub(r"[\s_]+", " ", name.strip().lower())


def _extract_infobox(wikitext: str) -> dict[str, str]:
    """Parse wikitext and extract all fields from the first country infobox."""
    parsed = mwparserfromhell.parse(wikitext)
    templates = parsed.filter_templates(recursive=True)

    infobox = None
    for template in templates:
        normalized = _normalize_template_name(str(template.name))
        for pattern in INFOBOX_PATTERNS:
            if pattern in normalized:
                infobox = template
                break
        if infobox is not None:
            break

    if infobox is None:
        raise ValueError("No infobox country template found.")

    result: dict[str, str] = {}
    for param in infobox.params:
        tag = _sanitize_tag(param.name.strip())
        if not tag or tag in DECORATIVE_FIELDS:
            continue
        clean = _sanitize_value(param.value)
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


# ---------------------------------------------------------------------------
# XML serialization
# ---------------------------------------------------------------------------

def _infobox_to_xml(country_name: str, infobox: dict[str, str]) -> ElementTree:
    """Convert a flat infobox dict to an ElementTree rooted at <country>."""
    root = Element("country")
    name_el = SubElement(root, "name")
    name_el.text = country_name

    for key, value in infobox.items():
        if value:
            child = SubElement(root, key)
            child.text = value

    tree = ElementTree(root)
    indent(tree, space="  ")
    return tree


def _output_path(country_name: str) -> str:
    filename = country_name.lower().replace(" ", "_") + ".xml"
    return os.path.join(DATA_DIR, filename)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_country(country_name: str, overwrite: bool = False) -> str:
    """
    Scrape and save the Wikipedia infobox for one country.

    Args:
        country_name: Canonical country name (key in WIKIPEDIA_NAME_OVERRIDES
                      or a name that matches the Wikipedia title directly).
        overwrite:    Re-scrape even if a local XML file already exists.

    Returns:
        Path to the saved XML file.
    """
    output_path = _output_path(country_name)

    if not overwrite and os.path.exists(output_path):
        return output_path

    wikitext = _fetch_wikitext(country_name)
    infobox = _extract_infobox(wikitext)

    tree = _infobox_to_xml(country_name, infobox)

    os.makedirs(DATA_DIR, exist_ok=True)
    tree.write(output_path, encoding="unicode", xml_declaration=True)

    return output_path


def list_unscraped() -> list[str]:
    """
    Return UN member names that do not yet have a file in data/raw/.

    A country is considered scraped when its XML exists (same path rule as
    collect_country). Failed scrapes with no file on disk are included.
    """
    return [
        country
        for country in UN_MEMBER_STATES
        if not os.path.exists(_output_path(country))
    ]


def _collect_countries(countries: list[str], overwrite: bool = False) -> dict[str, str]:
    """
    Scrape a given list of countries with retry/backoff.

    See collect_all() for throttling and return-value semantics.
    """
    results: dict[str, str] = {}
    max_retries = 5
    base_delay = 5  # seconds; doubles each retry: 5, 10, 20, 40, 80

    for country in countries:
        last_error = None

        for attempt in range(max_retries):
            try:
                path = collect_country(country, overwrite=overwrite)
                results[country] = path
                print(f"[OK]    {country} -> {path}")
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(
                        f"[RETRY] {country} (attempt {attempt + 1}/{max_retries}): "
                        f"{exc}, retrying in {delay}s"
                    )
                    time.sleep(delay)

        if last_error is not None:
            results[country] = f"ERROR: {last_error}"
            print(f"[FAIL]  {country}: {last_error}")

    return results


def collect_missing() -> dict[str, str]:
    """
    Scrape only UN members that have no XML in data/raw yet.

    Skips countries that already have a file (no Wikipedia request).
    Use this to fill gaps after a partial run without re-fetching all 193.
    """
    missing = list_unscraped()
    if not missing:
        print(
            f"[collector] Nothing to do: all {len(UN_MEMBER_STATES)} countries "
            f"already have XML under {DATA_DIR}."
        )
        return {}

    print(
        f"[collector] Scraping {len(missing)} missing of "
        f"{len(UN_MEMBER_STATES)} countries..."
    )
    return _collect_countries(missing, overwrite=False)


def collect_countries(countries: list[str], overwrite: bool = False) -> dict[str, str]:
    """
    Scrape a specific list of UN member states (with retry/backoff).

    Args:
        countries: Canonical names as in UN_MEMBER_STATES (e.g. "France",
                   "Congo, DR"). Unknown names raise ValueError.
        overwrite: Re-scrape even when a local XML file already exists.

    Returns:
        Dict mapping country name → file path or "ERROR: …".
    """
    if not countries:
        return {}

    valid = set(UN_MEMBER_STATES)
    unknown = [c for c in countries if c not in valid]
    if unknown:
        raise ValueError(
            f"Unknown country name(s): {unknown}. "
            "Names must match UN_MEMBER_STATES exactly "
            "(run list(UN_MEMBER_STATES) or --list-missing)."
        )

    # Preserve caller order while dropping duplicates
    seen: set[str] = set()
    ordered: list[str] = []
    for name in countries:
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    print(f"[collector] Scraping {len(ordered)} selected countries...")
    return _collect_countries(ordered, overwrite=overwrite)


def collect_all(overwrite: bool = False) -> dict[str, str]:
    """
    Scrape infoboxes for all UN member states.

    Uses exponential backoff (up to 5 attempts) to handle Wikipedia
    rate-limiting (HTTP 429). Countries that fail after all retries are
    recorded with an "ERROR: …" value in the returned dict.

    Proactive 1 req/s throttling is enforced by _throttle() inside
    _fetch_wikitext; backoff here is the fallback when 429s slip through
    or other transient errors occur.

    When overwrite=False, countries that already have XML are skipped
    without calling Wikipedia (same as collect_missing for those files).

    Args:
        overwrite: Re-scrape countries that already have a local XML file.

    Returns:
        Dict mapping canonical country name → file path or "ERROR: …".
    """
    return _collect_countries(UN_MEMBER_STATES, overwrite=overwrite)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Wikipedia country infoboxes to data/raw/*.xml",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--missing",
        action="store_true",
        help="Only countries without an XML file yet",
    )
    mode.add_argument(
        "--list-missing",
        action="store_true",
        help="Print countries missing from data/raw and exit",
    )
    mode.add_argument(
        "--countries",
        "-c",
        nargs="+",
        metavar="NAME",
        help='Specific UN members, e.g. -c France Germany "Congo, DR"',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-fetch even when XML already exists (default: skip existing)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()

    if args.list_missing:
        missing = list_unscraped()
        print(f"{len(missing)} missing (of {len(UN_MEMBER_STATES)}):")
        for name in missing:
            print(f"  {name}")
    elif args.missing:
        collect_missing()
    elif args.countries:
        collect_countries(args.countries, overwrite=args.overwrite)
    else:
        collect_all(overwrite=args.overwrite)