from pathlib import Path

# Blacklist file path
BLACKLIST_IPS = Path("blacklist/ip_list.txt")

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

# === Blacklist Functions ===
def add_blacklist_ip(ip: str) -> bool:
    items = _read_list(BLACKLIST_IPS)
    if ip not in items:
        items.add(ip)
        _write_list(BLACKLIST_IPS, items)
        return True
    return False

def remove_blacklist_ip(ip: str) -> bool:
    items = _read_list(BLACKLIST_IPS)
    if ip in items:
        items.remove(ip)
        _write_list(BLACKLIST_IPS, items)
        return True
    return False

def is_blacklisted(ip: str) -> bool:
    return ip in _read_list(BLACKLIST_IPS)

def list_blacklisted_ips() -> list:
    return sorted(_read_list(BLACKLIST_IPS))
