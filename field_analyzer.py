import os
import re
import json
import time
import xml.etree.ElementTree as ET
import requests
import mwparserfromhell
from collections import defaultdict


MEDIAWIKI_API = "https://en.wikipedia.org/w/api.php"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "data", "field_analysis.json")

# Delay between remote requests (seconds). Increase if still getting 429s.
BASE_DELAY_SECONDS = 3.0
MAX_RETRIES = 4

SAMPLE_COUNTRIES = [
    "Lebanon", "Syria", "Jordan", "Egypt", "Saudi Arabia", "Turkey", "Iran", "Iraq",
    "France", "Germany", "United Kingdom", "Italy", "Spain", "Switzerland",
    "Netherlands", "Sweden", "Poland", "Portugal", "Belgium", "Austria",
    "United States", "Canada", "Mexico", "Brazil", "Argentina", "Colombia", "Chile",
    "China", "Japan", "India", "South Korea", "Indonesia", "Pakistan", "Bangladesh",
    "Nigeria", "South Africa", "Kenya", "Ethiopia", "Ghana", "Tanzania",
    "Australia", "New Zealand",
    "Russia", "Ukraine",
    "Morocco", "Algeria", "Tunisia",
    "Thailand", "Vietnam", "Malaysia",
]

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
    "infobox political division",
]


def _local_xml_path(country_name: str) -> str:
    filename = country_name.lower().replace(" ", "_") + ".xml"
    return os.path.join(DATA_DIR, filename)


def _read_local_fields(country_name: str) -> dict[str, str] | None:
    path = _local_xml_path(country_name)
    if not os.path.exists(path):
        return None
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        fields: dict[str, str] = {}
        for child in root:
            tag = child.tag.strip()
            text = (child.text or "").strip()
            if tag and text:
                fields[tag] = text
        return fields
    except Exception as e:
        print(f"    [WARN] Could not parse local XML for {country_name}: {e}")
        return None


def _sanitize_tag(key: str) -> str:
    tag = key.strip().lower()
    tag = re.sub(r"[^\w]", "_", tag)
    tag = re.sub(r"_+", "_", tag)
    tag = tag.strip("_")
    if tag and tag[0].isdigit():
        tag = "field_" + tag
    return tag or ""


def _sanitize_value(value) -> str:
    stripped = value.strip_code(normalize=True, collapse=True, keep_template_params=False)
    stripped = re.sub(r"\[\d+\]", "", stripped)
    stripped = re.sub(r"<[^>]+>", "", stripped)
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped.strip()


def _normalize_template_name(name: str) -> str:
    return re.sub(r"[\s_]+", " ", name.strip().lower())


def _fetch_wikitext_with_retry(country_name: str) -> str | None:
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
        "titles": country_name,
        "redirects": 1,
    }
    headers = {
        "User-Agent": "WikipediaTEDProject/1.0 (COE543 LAU; Academic Research) python-requests"
    }

    delay = BASE_DELAY_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(MEDIAWIKI_API, params=params, headers=headers, timeout=20)
            if response.status_code == 429:
                wait = delay * attempt
                print(f"    [429] Rate limited. Waiting {wait:.0f}s before retry {attempt}/{MAX_RETRIES}...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()
            pages = data["query"]["pages"]
            page = next(iter(pages.values()))
            if "missing" in page:
                print(f"    [FAIL] {country_name}: page not found on Wikipedia")
                return None
            return page["revisions"][0]["slots"]["main"]["*"]
        except requests.exceptions.HTTPError as e:
            if attempt < MAX_RETRIES:
                wait = delay * attempt
                print(f"    [RETRY {attempt}] {country_name}: {e}. Waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"    [FAIL] {country_name}: {e} after {MAX_RETRIES} attempts")
                return None
        except Exception as e:
            print(f"    [FAIL] {country_name}: {e}")
            return None

    return None


def _fetch_remote_fields(country_name: str) -> dict[str, str] | None:
    wikitext = _fetch_wikitext_with_retry(country_name)
    if wikitext is None:
        return None

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
        print(f"    [FAIL] {country_name}: no country infobox found")
        return None

    fields: dict[str, str] = {}
    for param in infobox.params:
        tag = _sanitize_tag(str(param.name))
        if not tag or tag in DECORATIVE_FIELDS:
            continue
        value = _sanitize_value(param.value)
        if value:
            fields[tag] = value

    return fields


def _get_fields(country_name: str, fetch_missing: bool) -> dict[str, str] | None:
    local = _read_local_fields(country_name)
    if local is not None:
        return local
    if not fetch_missing:
        return None
    print(f"    [remote] {country_name} not cached locally, fetching...")
    time.sleep(BASE_DELAY_SECONDS)
    return _fetch_remote_fields(country_name)


def _looks_like_noise(value: str) -> bool:
    noise_patterns = [
        r"^https?://",
        r"\.svg$",
        r"\.png$",
        r"\.jpg$",
        r"\.jpeg$",
        r"\.gif$",
    ]
    lower = value.lower()
    return any(re.search(p, lower) for p in noise_patterns)


def _detect_aliases(field_counts: dict) -> dict[str, list[str]]:
    known_alias_groups = [
        ["population_estimate", "pop_est", "population"],
        ["area_km2", "area", "total_area_km2"],
        ["gdp_ppp", "gdp_purchasing_power_parity"],
        ["gdp_nominal", "gdp_real"],
        ["government_type", "government"],
        ["official_languages", "official_language", "languages_official"],
        ["ethnic_groups", "ethnicity", "ethnicities"],
        ["calling_code", "calling_codes"],
        ["utc_offset", "timezone_offset"],
        ["percent_water", "water_percent"],
    ]

    detected: dict[str, list[str]] = {}
    all_seen = set(field_counts.keys())

    for group in known_alias_groups:
        present = [f for f in group if f in all_seen]
        if len(present) > 1:
            detected[present[0]] = present[1:]

    return detected


def run_analysis(fetch_missing: bool = True) -> None:
    local_count = sum(
        1 for c in SAMPLE_COUNTRIES if os.path.exists(_local_xml_path(c))
    )
    remote_count = len(SAMPLE_COUNTRIES) - local_count

    print(f"Local XML files found : {local_count}/{len(SAMPLE_COUNTRIES)}")
    print(f"To fetch from Wikipedia: {remote_count}")
    if remote_count > 0 and fetch_missing:
        est = remote_count * BASE_DELAY_SECONDS
        print(f"Estimated fetch time   : ~{est:.0f}s ({est/60:.1f} min) (may be longer with retries)")
    print()

    field_counts: dict[str, int] = defaultdict(int)
    field_samples: dict[str, list[str]] = defaultdict(list)
    failed: list[str] = []
    total = len(SAMPLE_COUNTRIES)

    for i, country in enumerate(SAMPLE_COUNTRIES, 1):
        is_local = os.path.exists(_local_xml_path(country))
        source_tag = "local" if is_local else "remote"
        print(f"  [{i:02d}/{total}] {country} ({source_tag})")

        fields = _get_fields(country, fetch_missing)

        if fields is None:
            failed.append(country)
            continue

        for tag, value in fields.items():
            field_counts[tag] += 1
            if len(field_samples[tag]) < 5:
                field_samples[tag].append(f"{country}: {value[:80]}")

    success_count = total - len(failed)

    report_entries = []
    for tag, count in sorted(field_counts.items(), key=lambda x: -x[1]):
        samples = field_samples[tag]
        is_noise = bool(samples) and all(
            _looks_like_noise(s.split(": ", 1)[-1]) for s in samples
        )
        report_entries.append({
            "field": tag,
            "count": count,
            "frequency_pct": round(count / success_count * 100, 1),
            "noise": is_noise,
            "samples": samples,
        })

    aliases = _detect_aliases(field_counts)

    report = {
        "sample_size": success_count,
        "failed": failed,
        "alias_groups": aliases,
        "fields": report_entries,
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _print_summary(report, success_count)


def _print_summary(report: dict, success_count: int) -> None:
    print(f"\n{'='*62}")
    print(f"FIELD ANALYSIS REPORT: {success_count} countries analyzed")
    if report["failed"]:
        print(f"Failed ({len(report['failed'])}): {', '.join(report['failed'])}")
    print(f"{'='*62}\n")

    print(f"{'Field':<42} {'Count':>5} {'Freq%':>6}  Noise")
    print("-" * 62)
    for entry in report["fields"]:
        noise = "YES" if entry["noise"] else ""
        print(f"{entry['field']:<42} {entry['count']:>5} {entry['frequency_pct']:>5.1f}%  {noise}")

    print(f"\n{'='*62}")
    print("SAMPLE VALUES (top 30 fields, up to 3 samples each)")
    print("-" * 62)
    for entry in report["fields"][:30]:
        print(f"\n  {entry['field']} ({entry['frequency_pct']}%):")
        for sample in entry["samples"][:3]:
            print(f"    {sample}")

    if report["alias_groups"]:
        print(f"\n{'='*62}")
        print("DETECTED ALIAS GROUPS")
        print("-" * 62)
        for canonical, aliases in report["alias_groups"].items():
            print(f"  {canonical}  <-  {', '.join(aliases)}")

    print(f"\nFull report saved to: {REPORT_PATH}")
    print("\nNext steps:")
    print("  1. Fields >= 80% frequency -> whitelist candidates")
    print("  2. Read sample values -> comparable vs non-comparable")
    print("  3. Noise fields -> exclude")
    print("  4. Alias groups -> FIELD_NAME_MAP in preprocessor.py")


if __name__ == "__main__":
    run_analysis(fetch_missing=True)