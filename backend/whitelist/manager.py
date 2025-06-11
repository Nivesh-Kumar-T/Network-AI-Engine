from pathlib import Path

# Paths to the whitelist files
IP_LIST = Path("whitelist/ip_list.txt")
COUNTRY_LIST = Path("whitelist/country_list.txt")


# === Generic Helpers ===
def _read_list(path: Path) -> set:
    if path.exists():
        with open(path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def _write_list(path: Path, items: set):
    with open(path, "w") as f:
        for item in sorted(items):
            f.write(item + "\n")


# === IP Whitelist Functions ===
def add_ip(ip: str) -> bool:
    items = _read_list(IP_LIST)
    if ip not in items:
        items.add(ip)
        _write_list(IP_LIST, items)
        return True
    return False

def remove_ip(ip: str) -> bool:
    items = _read_list(IP_LIST)
    if ip in items:
        items.remove(ip)
        _write_list(IP_LIST, items)
        return True
    return False

def list_ips() -> list:
    return sorted(_read_list(IP_LIST))


# === Country Whitelist Functions ===
def add_country(code: str) -> bool:
    code = code.upper()
    items = _read_list(COUNTRY_LIST)
    if code not in items:
        items.add(code)
        _write_list(COUNTRY_LIST, items)
        return True
    return False

def remove_country(code: str) -> bool:
    code = code.upper()
    items = _read_list(COUNTRY_LIST)
    if code in items:
        items.remove(code)
        _write_list(COUNTRY_LIST, items)
        return True
    return False

def list_countries() -> list:
    return sorted(_read_list(COUNTRY_LIST))