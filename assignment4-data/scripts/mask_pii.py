import re
import phonenumbers

EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)



US_PHONE_RE = re.compile(r"""
    (?<![\w+])
    (?:\+?1[\s.-]?)?
    (?:\(\d{3}\)|\d{3})               # area code: any 3 digits
    [\s.-]?
    \d{3}                             # exchange: any 3 digits
    [\s.-]?
    \d{4}
    (?:\s*(?:extension|ext\.?|x)\s*\d{1,6})?
    (?!\w)
""", re.VERBOSE | re.IGNORECASE)

OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4_RE = re.compile(rf"(?<!\d)(?<!\d\.){OCTET}(?:\.{OCTET}){{3}}(?!\.?\d)")

def mask_email(inp) -> tuple[str, int]:
    return EMAIL_RE.subn("|||EMAIL_ADDRESS|||", inp)

def mask_phone(inp) -> tuple[str, int]:
    return US_PHONE_RE.subn("|||PHONE_NUMBER|||", inp)

def mask_ip(inp) -> tuple[str, int]:
    return IPV4_RE.subn("|||IP_ADDRESS|||", inp)

def all_masks(inp):
    inp, _ = mask_email(inp)
    inp, _ = mask_phone(inp)
    inp, _ = mask_ip(inp)
    return inp